//  Copyright (c) 2020 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Test that loops over each core assigned to the program and launches
// tasks bound to that core incrementally.
// Tasks should always report the right core number when they run.

#include <pika/debugging/print.hpp>
#include <pika/local/execution.hpp>
#include <pika/local/future.hpp>
#include <pika/local/init.hpp>
#include <pika/local/runtime.hpp>
#include <pika/local/thread.hpp>
#include <pika/modules/resource_partitioner.hpp>
#include <pika/modules/schedulers.hpp>
#include <pika/modules/testing.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>

namespace pika {
    // use <true>/<false> to enable/disable debug printing
    using sbc_print_on = pika::debug::enable_print<false>;
    static sbc_print_on deb_schbin("SCHBIND");
}    // namespace pika

// counts down on destruction
struct dec_counter
{
    explicit dec_counter(std::atomic<int>& counter)
      : counter_(counter)
    {
    }
    ~dec_counter()
    {
        --counter_;
    }
    //
    std::atomic<int>& counter_;
};

void threadLoop()
{
    unsigned const iterations = 2048;
    std::atomic<int> count_down(iterations);

    auto f = [&count_down](std::size_t iteration, std::size_t thread_expected) {
        dec_counter dec(count_down);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        std::size_t thread_actual = pika::get_worker_thread_num();
        pika::deb_schbin.debug(pika::debug::str<10>("Iteration"),
            pika::debug::dec<4>(iteration),
            pika::debug::str<20>("Running on thread"), thread_actual,
            pika::debug::str<10>("Expected"), thread_expected);
        PIKA_TEST_EQ(thread_actual, thread_expected);
    };

    std::size_t threads = pika::get_num_worker_threads();
    // launch tasks on threads using numbering 0,1,2,3...0,1,2,3
    for (std::size_t i = 0; i < iterations; ++i)
    {
        auto exec = pika::execution::parallel_executor(
            pika::threads::thread_priority::bound,
            pika::threads::thread_stacksize::default_,
            pika::threads::thread_schedule_hint(std::int16_t(i % threads)));
        pika::async(exec, f, i, (i % threads)).get();
    }

    do
    {
        pika::this_thread::yield();
        pika::deb_schbin.debug(
            pika::debug::str<15>("count_down"), pika::debug::dec<4>(count_down));
    } while (count_down > 0);

    pika::deb_schbin.debug(
        pika::debug::str<15>("complete"), pika::debug::dec<4>(count_down));
    PIKA_TEST_EQ(count_down, 0);
}

int pika_main()
{
    auto const current = pika::threads::get_self_id_data()->get_scheduler_base();
    std::cout << "Scheduler is " << current->get_description() << std::endl;
    if (std::string("core-shared_priority_queue_scheduler") !=
        current->get_description())
    {
        std::cout << "The scheduler might not work properly " << std::endl;
    }

    threadLoop();

    pika::local::finalize();
    pika::deb_schbin.debug(pika::debug::str<15>("Finalized"));
    return pika::util::report_errors();
}

int main(int argc, char* argv[])
{
    pika::local::init_params init_args;

    init_args.rp_callback = [](auto& rp,
                                pika::program_options::variables_map const&) {
        // setup the default pool with a numa/binding aware scheduler
        rp.create_thread_pool("default",
            pika::resource::scheduling_policy::shared_priority,
            pika::threads::policies::scheduler_mode(
                pika::threads::policies::default_mode));
    };

    return pika::local::init(pika_main, argc, argv, init_args);
}
