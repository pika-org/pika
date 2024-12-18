//  Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
//  Copyright (c) 2007-2021 Hartmut Kaiser
//  Copyright (c)      2013 Thomas Heller
//  Copyright (c)      2013 Patricia Grubel
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// FIXME: Calling the tasks "workers" overloads the term worker-thread (which
// refers to OS-threads).

#include <pika/pika.hpp>
#include <pika/pika_init.hpp>

#include <pika/functional/bind.hpp>
#include <pika/string_util/classification.hpp>
#include <pika/string_util/split.hpp>
#include <pika/testing.hpp>

#include <fmt/ostream.h>
#include <fmt/printf.h>

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "activate_counters.hpp"
#include "worker_timed.hpp"

char const* benchmark_name = "Homogeneous Timed Task Spawn - pika";

using pika::program_options::options_description;
using pika::program_options::value;
using pika::program_options::variables_map;

using pika::get_os_thread_count;

using pika::threads::detail::make_thread_function_nullary;
using pika::threads::detail::register_work;
using pika::threads::detail::thread_init_data;

using pika::detail::get_runtime;
using pika::this_thread::suspend;

using pika::chrono::detail::high_resolution_timer;

using pika::reset_active_counters;
using pika::stop_active_counters;

using std::cout;
using std::flush;

///////////////////////////////////////////////////////////////////////////////
// Command-line variables.
std::uint64_t tasks = 500000;
std::uint64_t suspended_tasks = 0;
std::uint64_t delay = 0;
bool header = true;
bool csv_header = false;
std::string scaling("weak");
std::string distribution("static-balanced-stackbased");

std::uint64_t suspend_step = 0;
std::uint64_t no_suspend_step = 1;

///////////////////////////////////////////////////////////////////////////////
std::string format_build_date()
{
    std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();

    std::time_t current_time = std::chrono::system_clock::to_time_t(now);

    std::string ts = std::ctime(&current_time);
    ts.resize(ts.size() - 1);    // remove trailing '\n'
    return ts;
}

///////////////////////////////////////////////////////////////////////////////
void print_results(std::uint64_t cores, double walltime, double warmup_estimate,
    std::vector<std::string> const& counter_shortnames,
    std::shared_ptr<pika::util::activate_counters> ac)
{
    std::vector<pika::performance_counters::counter_value> counter_values;

    if (ac) counter_values = ac->evaluate_counters(pika::launch::sync);

    if (csv_header)
    {
        header = false;
        cout << "Delay,Tasks,STasks,OS_Threads,Execution_Time_sec,Warmup_sec";

        for (auto const& counter_shortname : counter_shortnames)
        {
            cout << "," << counter_shortname;
        }
        cout << "\n";
    }

    if (header)
    {
        cout << "# BENCHMARK: " << benchmark_name << " (" << scaling << " scaling, " << distribution
             << " distribution)\n";

        cout << "# VERSION: " << PIKA_HAVE_GIT_COMMIT << " " << format_build_date() << "\n"
             << "#\n";

        // Note that if we change the number of fields above, we have to
        // change the constant that we add when printing out the field # for
        // performance counters below (e.g. the last_index part).
        cout << "## 0:DELAY:Delay [micro-seconds] - Independent Variable\n"
                "## 1:TASKS:# of Tasks - Independent Variable\n"
                "## 2:STASKS:# of Tasks to Suspend - Independent Variable\n"
                "## 3:OSTHRDS:OS-threads - Independent Variable\n"
                "## 4:WTIME:Total Walltime [seconds]\n"
                "## 5:WARMUP:Total Walltime [seconds]\n";

        std::uint64_t const last_index = 5;

        for (std::uint64_t i = 0; i < counter_shortnames.size(); ++i)
        {
            cout << "## " << (i + 1 + last_index) << ":" << counter_shortnames[i] << ":"
                 << ac->name(i);

            if (!ac->unit_of_measure(i).empty()) cout << " [" << ac->unit_of_measure(i) << "]";

            cout << "\n";
        }
    }

    fmt::print(cout, "{}, {}, {}, {}, {:.14g}, {:.14g}", delay, tasks, suspended_tasks, cores,
        walltime, warmup_estimate);

    if (ac)
    {
        for (std::uint64_t i = 0; i < counter_shortnames.size(); ++i)
            fmt::print(cout, ", {:.14g}", counter_values[i].get_value<double>());
    }

    cout << "\n";
}

///////////////////////////////////////////////////////////////////////////////
void wait_for_tasks(pika::barrier& finished, std::uint64_t suspended_tasks)
{
    std::uint64_t const pending_count = get_runtime().get_thread_manager().get_thread_count(
        pika::execution::thread_priority::normal,
        pika::threads::detail::thread_schedule_state::pending);

    if (pending_count == 0)
    {
        std::uint64_t const all_count = get_runtime().get_thread_manager().get_thread_count(
            pika::execution::thread_priority::normal);

        if (all_count != suspended_tasks + 1)
        {
            thread_init_data data(make_thread_function_nullary(pika::util::detail::bind(
                                      &wait_for_tasks, std::ref(finished), suspended_tasks)),
                "wait_for_tasks", pika::execution::thread_priority::low);
            register_work(data);
            return;
        }
    }

    finished.wait();
}

///////////////////////////////////////////////////////////////////////////////
pika::threads::detail::thread_result_type invoke_worker_timed_no_suspension(
    pika::threads::detail::thread_restart_state ex =
        pika::threads::detail::thread_restart_state::signaled)
{
    worker_timed(delay * 1000);
    return pika::threads::detail::thread_result_type(
        pika::threads::detail::thread_schedule_state::terminated,
        pika::threads::detail::invalid_thread_id);
}

pika::threads::detail::thread_result_type invoke_worker_timed_suspension(
    pika::threads::detail::thread_restart_state ex =
        pika::threads::detail::thread_restart_state::signaled)
{
    worker_timed(delay * 1000);

    pika::error_code ec(pika::throwmode::lightweight);
    pika::this_thread::suspend(
        pika::threads::detail::thread_schedule_state::suspended, "suspend", ec);

    return pika::threads::detail::thread_result_type(
        pika::threads::detail::thread_schedule_state::terminated,
        pika::threads::detail::invalid_thread_id);
}

///////////////////////////////////////////////////////////////////////////////
using stage_worker_function = void (*)(std::uint64_t, bool);

void stage_worker_static_balanced_stackbased(std::uint64_t target_thread, bool suspend)
{
    if (suspend)
    {
        pika::threads::detail::thread_init_data data(&invoke_worker_timed_suspension,
            "invoke_worker_timed_suspension", pika::execution::thread_priority::normal,
            pika::execution::thread_schedule_hint(static_cast<std::int16_t>(target_thread)));
        pika::threads::detail::register_work(data);
    }
    else
    {
        pika::threads::detail::thread_init_data data(&invoke_worker_timed_no_suspension,
            "invoke_worker_timed_no_suspension", pika::execution::thread_priority::normal,
            pika::execution::thread_schedule_hint(static_cast<std::int16_t>(target_thread)));
        pika::threads::detail::register_work(data);
    }
}

void stage_worker_static_balanced_stackless(std::uint64_t target_thread, bool suspend)
{
    if (suspend)
    {
        pika::threads::detail::thread_init_data data(&invoke_worker_timed_suspension,
            "invoke_worker_timed_suspension", pika::execution::thread_priority::normal,
            pika::execution::thread_schedule_hint(static_cast<std::int16_t>(target_thread)),
            pika::execution::thread_stacksize::nostack);
        pika::threads::detail::register_work(data);
    }
    else
    {
        pika::threads::detail::thread_init_data data(&invoke_worker_timed_no_suspension,
            "invoke_worker_timed_no_suspension", pika::execution::thread_priority::normal,
            pika::execution::thread_schedule_hint(static_cast<std::int16_t>(target_thread)),
            pika::execution::thread_stacksize::nostack);
        pika::threads::detail::register_work(data);
    }
}

void stage_worker_static_imbalanced(std::uint64_t target_thread, bool suspend)
{
    if (suspend)
    {
        pika::threads::detail::thread_init_data data(&invoke_worker_timed_suspension,
            "invoke_worker_timed_suspension", pika::execution::thread_priority::normal,
            pika::execution::thread_schedule_hint(0));
        pika::threads::detail::register_work(data);
    }
    else
    {
        pika::threads::detail::thread_init_data data(&invoke_worker_timed_no_suspension,
            "invoke_worker_timed_no_suspension", pika::execution::thread_priority::normal,
            pika::execution::thread_schedule_hint(0));
        pika::threads::detail::register_work(data);
    }
}

void stage_worker_round_robin(std::uint64_t target_thread, bool suspend)
{
    if (suspend)
    {
        pika::threads::detail::thread_init_data data(
            &invoke_worker_timed_suspension, "invoke_worker_timed_suspension");
        pika::threads::detail::register_work(data);
    }
    else
    {
        pika::threads::detail::thread_init_data data(
            &invoke_worker_timed_no_suspension, "invoke_worker_timed_no_suspension");
        pika::threads::detail::register_work(data);
    }
}

void stage_workers(
    std::uint64_t target_thread, std::uint64_t local_tasks, stage_worker_function stage_worker)
{
    std::uint64_t num_thread = pika::get_worker_thread_num();

    if (num_thread != target_thread)
    {
        thread_init_data data(make_thread_function_nullary(pika::util::detail::bind(
                                  &stage_workers, target_thread, local_tasks, stage_worker)),
            "stage_workers", pika::execution::thread_priority::normal,
            pika::execution::thread_schedule_hint(static_cast<std::int16_t>(target_thread)));
        register_work(data);
        return;
    }

    for (std::uint64_t i = 0; i < local_tasks;)
    {
        for (std::uint64_t j = 0; j < suspend_step; ++j)
        {
            stage_worker(target_thread, true);
            ++i;
        }
        for (std::uint64_t j = 0; j < no_suspend_step; ++j)
        {
            stage_worker(target_thread, false);
            ++i;
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
int pika_main(variables_map& vm)
{
    {
        if (vm.count("no-header")) header = false;

        if (vm.count("csv-header")) csv_header = true;

        if (0 == tasks) throw std::invalid_argument("count of 0 tasks specified\n");

        if (suspended_tasks > tasks)
            throw std::invalid_argument("suspended tasks must be smaller than tasks\n");

        std::uint64_t const os_thread_count = get_os_thread_count();

        ///////////////////////////////////////////////////////////////////////
        stage_worker_function stage_worker;

        if ("static-balanced-stackbased" == distribution)
        {
            stage_worker = &stage_worker_static_balanced_stackbased;
        }
        else if ("static-balanced-stackless" == distribution)
        {
            stage_worker = &stage_worker_static_balanced_stackless;
        }
        else if ("static-imbalanced" == distribution)
        {
            stage_worker = &stage_worker_static_imbalanced;
        }
        else if ("round-robin" == distribution) { stage_worker = &stage_worker_round_robin; }
        else
        {
            throw std::invalid_argument(
                "invalid distribution type specified (valid options are \"static-balanced\", "
                "\"static-imbalanced\" or \"round-robin\")");
        }

        ///////////////////////////////////////////////////////////////////////
        std::uint64_t tasks_per_feeder = 0;
        //std::uint64_t total_tasks = 0;
        std::uint64_t suspended_tasks_per_feeder = 0;
        std::uint64_t total_suspended_tasks = 0;

        if ("strong" == scaling)
        {
            if (tasks % os_thread_count)
            {
                throw std::invalid_argument("tasks must be cleanly divisible by OS-thread count\n");
            }
            if (suspended_tasks % os_thread_count)
            {
                throw std::invalid_argument(
                    "suspended tasks must be cleanly divisible by OS-thread count\n");
            }
            tasks_per_feeder = tasks / os_thread_count;
            //total_tasks      = tasks;
            suspended_tasks_per_feeder = suspended_tasks / os_thread_count;
            total_suspended_tasks = suspended_tasks;
        }
        else if ("weak" == scaling)
        {
            tasks_per_feeder = tasks;
            //total_tasks      = tasks * os_thread_count;
            suspended_tasks_per_feeder = suspended_tasks;
            total_suspended_tasks = suspended_tasks * os_thread_count;
        }
        else
        {
            throw std::invalid_argument("invalid scaling type specified (valid options are "
                                        "\"strong\                " or \"weak\")");
        }

        ///////////////////////////////////////////////////////////////////////
        if (suspended_tasks != 0)
        {
            std::uint64_t gcd = std::gcd(tasks_per_feeder, suspended_tasks_per_feeder);

            suspend_step = suspended_tasks_per_feeder / gcd;
            // We check earlier to make sure that there are never more
            // suspended tasks than tasks requested.
            no_suspend_step = (tasks_per_feeder / gcd) - suspend_step;
        }

        ///////////////////////////////////////////////////////////////////////
        std::vector<std::string> counter_shortnames;
        std::vector<std::string> counters;
        if (vm.count("counter"))
        {
            std::vector<std::string> raw_counters = vm["counter"].as<std::vector<std::string>>();

            for (auto& raw_counter : raw_counters)
            {
                std::vector<std::string> entry;
                pika::detail::split(entry, raw_counter, pika::detail::is_any_of(","),
                    pika::detail::token_compress_mode::on);

                PIKA_TEST_EQ(entry.size(), 2);

                counter_shortnames.push_back(entry[0]);
                counters.push_back(entry[1]);
            }
        }

        std::shared_ptr<pika::util::activate_counters> ac;
        if (!counters.empty()) { ac = std::make_shared<pika::util::activate_counters>(counters); }

        ///////////////////////////////////////////////////////////////////////
        // Start the clock.
        high_resolution_timer t;
        if (ac) { ac->reset_counters(); }

        // This needs to stay here; we may have suspended as recently as the
        // performance counter reset (which is called just before the staging
        // function).
        std::uint64_t const num_thread = pika::get_worker_thread_num();

        for (std::uint64_t i = 0; i < os_thread_count; ++i)
        {
            if (num_thread == i) continue;

            thread_init_data data(make_thread_function_nullary(pika::util::detail::bind(
                                      &stage_workers, i, tasks_per_feeder, stage_worker)),
                "stage_workers", pika::execution::thread_priority::normal,
                pika::execution::thread_schedule_hint(static_cast<std::int16_t>(i)));
            register_work(data);
        }

        stage_workers(num_thread, tasks_per_feeder, stage_worker);

        double warmup_estimate = t.elapsed();

        // Schedule a low-priority thread; when it is executed, it checks to
        // make sure all the tasks (which are normal priority) have been
        // executed, and then it
        pika::barrier finished(2);

        thread_init_data data(make_thread_function_nullary(pika::util::detail::bind(
                                  &wait_for_tasks, std::ref(finished), total_suspended_tasks)),
            "wait_for_tasks", pika::execution::thread_priority::low);
        register_work(data);

        finished.wait();

        // Stop the clock
        double time_elapsed = t.elapsed();

        print_results(os_thread_count, time_elapsed, warmup_estimate, counter_shortnames, ac);
    }

    if (suspended_tasks != 0)
    {
        // Force termination of all suspended tasks.
        pika::detail::get_runtime().get_thread_manager().abort_all_suspended_threads();
    }
    pika::finalize();
    return EXIT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options.
    options_description cmdline("usage: " PIKA_APPLICATION_STRING " [options]");

    // clang-format off
    cmdline.add_options()
        ( "scaling"
        , value<std::string>(&scaling)->default_value("weak")
        , "type of scaling to benchmark (valid options are \"strong\" or "
          "\"weak\")")

        ( "distribution"
        , value<std::string>(&distribution)->default_value(
            "static-balanced-stackbased")
        , "type of distribution to perform (valid options are "
          "\"static-balanced-stackbased\", \"static-balanced-stackless\", "
          "\"static-imbalanced\", or \"round-robin\")")

        ( "tasks"
        , value<std::uint64_t>(&tasks)->default_value(500000)
        , "number of tasks to invoke (when strong-scaling, this is the total "
          "number of tasks invoked; when weak-scaling, it is the number of "
          "tasks per core)")

        ( "suspended-tasks"
        , value<std::uint64_t>(&suspended_tasks)->default_value(0)
        , "number of tasks to suspend (when strong-scaling, this is the total "
          "number of tasks suspended; when weak-scaling, it is the number of "
          "suspended per core)")

        ( "delay"
        , value<std::uint64_t>(&delay)->default_value(5)
        , "duration of delay in microseconds")

        ( "counter"
        , value<std::vector<std::string> >()->composing()
        , "activate and report the specified performance counter")

        ( "no-header"
        , "do not print out the header")

        ( "csv-header"
        , "print out csv header")
        ;
    // clang-format on

    // Initialize and run pika.
    pika::init_params init_args;
    init_args.desc_cmdline = cmdline;

    return pika::init(argc, argv, init_args);
}
