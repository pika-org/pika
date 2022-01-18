//  Copyright (c) 2011-2014 Bryce Adelstein-Lelbach
//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2013-2014 Thomas Heller
//  Copyright (c) 2013-2014 Patricia Grubel
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/barrier.hpp>
#include <pika/local/functional.hpp>
#include <pika/local/init.hpp>
#include <pika/local/thread.hpp>
#include <pika/modules/format.hpp>

#include "htts2.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

template <typename BaseClock = std::chrono::steady_clock>
struct pika_driver : htts2::driver
{
    pika_driver(int argc, char** argv)
      : htts2::driver(argc, argv, true)
    //      , count_(0)
    {
    }

    void run()
    {
        std::vector<std::string> const cfg = {
            "pika.os_threads=" + std::to_string(osthreads_),
            "pika.run_pika_main!=0", "pika.commandline.allow_unknown!=1"};

        pika::util::function_nonser<int(
            pika::program_options::variables_map & vm)>
            f;
        pika::program_options::options_description desc;

        pika::local::init_params init_args;
        init_args.cfg = cfg;
        init_args.desc_cmdline = desc;

        using pika::util::placeholders::_1;

        pika::local::init(
            std::function<int(pika::program_options::variables_map&)>(
                pika::util::bind(&pika_driver::run_impl, std::ref(*this), _1)),
            argc_, argv_, init_args);
    }

private:
    int run_impl(pika::program_options::variables_map&)
    {
        // Cold run
        //kernel();

        // Hot run
        results_type results = kernel();
        print_results(results);

        return pika::local::finalize();
    }

    pika::threads::thread_result_type payload_thread_function(
        pika::threads::thread_restart_state =
            pika::threads::thread_restart_state::signaled)
    {
        htts2::payload<BaseClock>(this->payload_duration_ /* = p */);
        //++count_;
        return pika::threads::thread_result_type(
            pika::threads::thread_schedule_state::terminated,
            pika::threads::invalid_thread_id);
    }

    void stage_tasks(std::uint64_t target_osthread)
    {
        std::uint64_t const this_osthread = pika::get_worker_thread_num();

        // This branch is very rarely taken (I've measured); this only occurs
        // if we are unlucky enough to be stolen from our intended queue.
        if (this_osthread != target_osthread)
        {
            // Reschedule in an attempt to correct.
            pika::threads::thread_init_data data(
                pika::threads::make_thread_function_nullary(
                    pika::util::bind(&pika_driver::stage_tasks, std::ref(*this),
                        target_osthread)),
                nullptr    // No pika-thread name.
                ,
                pika::threads::thread_priority::normal
                // Place in the target OS-thread's queue.
                ,
                pika::threads::thread_schedule_hint(target_osthread));
            pika::threads::register_work(data);
        }

        for (std::uint64_t i = 0; i < this->tasks_; ++i)
        {
            using pika::util::placeholders::_1;
            pika::threads::thread_init_data data(
                pika::util::bind(
                    &pika_driver::payload_thread_function, std::ref(*this), _1),
                nullptr    // No pika-thread name.
                ,
                pika::threads::thread_priority::normal
                // Place in the target OS-thread's queue.
                ,
                pika::threads::thread_schedule_hint(target_osthread));
            pika::threads::register_work(data);
        }
    }

    void wait_for_tasks(pika::lcos::local::barrier& finished)
    {
        std::uint64_t const pending_count =
            get_thread_count(pika::threads::thread_priority::normal,
                pika::threads::thread_schedule_state::pending);

        if (pending_count == 0)
        {
            std::uint64_t const all_count =
                get_thread_count(pika::threads::thread_priority::normal);

            if (all_count != 1)
            {
                pika::threads::thread_init_data data(
                    pika::threads::make_thread_function_nullary(
                        pika::util::bind(&pika_driver::wait_for_tasks,
                            std::ref(*this), std::ref(finished))),
                    nullptr, pika::threads::thread_priority::low);
                register_work(data);
                return;
            }
        }

        finished.wait();
    }

    typedef double results_type;

    results_type kernel()
    {
        ///////////////////////////////////////////////////////////////////////

        //count_ = 0;

        results_type results;

        std::uint64_t const this_osthread = pika::get_worker_thread_num();

        htts2::timer<BaseClock> t;

        ///////////////////////////////////////////////////////////////////////
        // Warmup Phase
        for (std::uint64_t i = 0; i < this->osthreads_; ++i)
        {
            if (this_osthread == i)
                continue;

            pika::threads::thread_init_data data(
                pika::threads::make_thread_function_nullary(pika::util::bind(
                    &pika_driver::stage_tasks, std::ref(*this), i)),
                nullptr    // No pika-thread name.
                ,
                pika::threads::thread_priority::normal
                // Place in the target OS-thread's queue.
                ,
                pika::threads::thread_schedule_hint(i));
            pika::threads::register_work(data);
        }

        stage_tasks(this_osthread);

        ///////////////////////////////////////////////////////////////////////
        // Compute + Cooldown Phase

        // The use of an atomic and live waiting here does not add any noticeable
        // overhead, as compared to the more complicated continuation-style
        // detection method that checks the threadmanager internal counters
        // (I've measured). Using this technique is preferable as it is more
        // comparable to the other implementations (especially qthreads).
        //        do {
        //            pika::this_thread::suspend();
        //        } while (count_ < (this->tasks_ * this->osthreads_));

        // Schedule a low-priority thread; when it is executed, it checks to
        // make sure all the tasks (which are normal priority) have been
        // executed, and then it
        pika::lcos::local::barrier finished(2);

        pika::threads::thread_init_data data(
            pika::threads::make_thread_function_nullary(
                pika::util::bind(&pika_driver::wait_for_tasks, std::ref(*this),
                    std::ref(finished))),
            nullptr, pika::threads::thread_priority::low);
        register_work(data);

        finished.wait();

        // w_M [nanoseconds]
        results = static_cast<double>(t.elapsed());

        return results;

        ///////////////////////////////////////////////////////////////////////
    }

    void print_results(results_type results) const
    {
        if (this->io_ == htts2::csv_with_headers)
            std::cout
                << "OS-threads (Independent Variable),"
                << "Tasks per OS-thread (Control Variable) [tasks/OS-threads],"
                << "Payload Duration (Control Variable) [nanoseconds],"
                << "Total Walltime [nanoseconds]"
                << "\n";

        pika::util::format_to(std::cout, "{},{},{},{:.14g}\n", this->osthreads_,
            this->tasks_, this->payload_duration_, results);
    }

    //    std::atomic<std::uint64_t> count_;
};

int main(int argc, char** argv)
{
    pika_driver<> d(argc, argv);

    d.run();

    return 0;
}
