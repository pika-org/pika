//  Copyright (c) 2022 ETH Zurich
//  Copyright (c) 2019 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/execution.hpp>
#include <pika/functional.hpp>
#include <pika/init.hpp>
#include <pika/testing.hpp>

#include <apex_options.hpp>

#include <utility>

namespace ex = pika::execution::experimental;
namespace tt = pika::this_thread::experimental;

auto test_senders_execute_no_parent_annotation()
{
    // <unknown>
    ex::execute(ex::thread_pool_scheduler{}, [] {});

    // 0-execute-no-parent-A
    ex::execute(ex::with_annotation(
                    ex::thread_pool_scheduler{}, "0-execute-no-parent-A"),
        [] {});

    // 0-execute-no-parent-C
    ex::execute(ex::with_annotation(
                    ex::thread_pool_scheduler{}, "0-execute-no-parent-B"),
        pika::annotated_function([] {}, "0-execute-no-parent-C"));
}

auto test_senders_execute_parent_annotation()
{
    // 1-execute-parent
    ex::execute(
        ex::with_annotation(ex::thread_pool_scheduler{}, "1-execute-parent"),
        [] {
            // 1-execute-parent
            ex::execute(ex::thread_pool_scheduler{}, [] {});

            // 1-execute-parent-A
            ex::execute(ex::with_annotation(
                            ex::thread_pool_scheduler{}, "1-execute-parent-A"),
                [] {});

            // 1-execute-no-parent-C
            ex::execute(ex::with_annotation(
                            ex::thread_pool_scheduler{}, "1-execute-parent-B"),
                pika::annotated_function([] {}, "1-execute-parent-C"));
        });
}

auto test_senders_schedule_no_parent_annotation()
{
    return ex::when_all(
        // <unknown>
        ex::schedule(ex::thread_pool_scheduler{}),

        // 2-schedule-no-parent-A
        ex::schedule(ex::with_annotation(
            ex::thread_pool_scheduler{}, "2-schedule-no-parent-A")),

        // <unknown> and 2-schedule-parent-B, plus one yielded <unknown> which
        // does not show up in the call count (but does show up in OTF2 files)
        ex::schedule(ex::thread_pool_scheduler{}) |
            ex::then(pika::annotated_function([] {}, "2-schedule-no-parent-B")),

        // 2-schedule-parent-C and 2-schedule-parent-D
        ex::schedule(ex::with_annotation(
            ex::thread_pool_scheduler{}, "2-schedule-no-parent-C")) |
            ex::then(pika::annotated_function([] {}, "2-schedule-no-parent-D")),

        // 2 x <unknown>
        ex::schedule(ex::thread_pool_scheduler{}) |
            ex::transfer(ex::thread_pool_scheduler{}),

        // 2-schedule-no-parent-E and <unknown>
        ex::schedule(ex::with_annotation(
            ex::thread_pool_scheduler{}, "2-schedule-no-parent-E")) |
            ex::transfer(ex::thread_pool_scheduler{}),

        // 2-schedule-no-parent-F and <unknown>
        ex::schedule(ex::thread_pool_scheduler{}) |
            ex::transfer(ex::with_annotation(
                ex::thread_pool_scheduler{}, "2-schedule-no-parent-F")),

        // 2 x <unknown>
        ex::schedule(ex::thread_pool_scheduler{}) | ex::bulk(2, [](int) {}),

        // 2 x 2-schedule-no-parent-G
        ex::schedule(ex::with_annotation(
            ex::thread_pool_scheduler{}, "2-schedule-no-parent-G")) |
            ex::bulk(2, [](int) {}),

        // 2 x <unknown>, 1 x 3-schedule-no-parent-H, plus one yielded
        // 3-schedule-no-parent-H which does not show up in the call count (but
        // does show up in OTF2 files)
        ex::schedule(ex::thread_pool_scheduler{}) |
            ex::bulk(2,
                pika::annotated_function([](int) {}, "2-schedule-no-parent-H"))

    );
}

auto test_senders_schedule_parent_annotation()
{
    // 3-schedule-parent
    return ex::schedule(ex::with_annotation(
               ex::thread_pool_scheduler{}, "3-schedule-parent")) |
        ex::let_value([]() {
            return ex::when_all(
                // 3-schedule-parent
                ex::schedule(ex::thread_pool_scheduler{}),

                // 3-schedule-parent-A
                ex::schedule(ex::with_annotation(
                    ex::thread_pool_scheduler{}, "3-schedule-parent-A")),

                // 3-schedule-parent and 3-schedule-parent-B, plus one yielded
                // 3-schedule-parent which does not show up in the call count
                // (but does show up in OTF2 files)
                ex::schedule(ex::thread_pool_scheduler{}) |
                    ex::then(
                        pika::annotated_function([] {}, "3-schedule-parent-B")),

                // 3-schedule-parent-C and 3-schedule-parent-D
                ex::schedule(ex::with_annotation(
                    ex::thread_pool_scheduler{}, "3-schedule-parent-C")) |
                    ex::then(
                        pika::annotated_function([] {}, "3-schedule-parent-D")),

                // 2 x 3-schedule-parent
                ex::schedule(ex::thread_pool_scheduler{}) |
                    ex::transfer(ex::thread_pool_scheduler{}),

                // 3-schedule-parent-E and 3-schedule-parent
                ex::schedule(ex::with_annotation(
                    ex::thread_pool_scheduler{}, "3-schedule-parent-E")) |
                    ex::transfer(ex::thread_pool_scheduler{}),

                // 3-schedule-parent-F and 3-schedule-parent
                ex::schedule(ex::thread_pool_scheduler{}) |
                    ex::transfer(ex::with_annotation(
                        ex::thread_pool_scheduler{}, "3-schedule-parent-F")),

                // 2 x 3-schedule-parent
                ex::schedule(ex::thread_pool_scheduler{}) |
                    ex::bulk(2, [](int) {}),

                // 2 x 3-schedule-parent-G
                ex::schedule(ex::with_annotation(
                    ex::thread_pool_scheduler{}, "3-schedule-parent-G")) |
                    ex::bulk(2, [](int) {}),

                // 2 x 3-schedule-parent, 1 x 3-schedule-parent-H, plus one
                // yielded 3-schedule-parent-H which does not show up in the
                // call count (but does show up in OTF2 files)
                ex::schedule(ex::thread_pool_scheduler{}) |
                    ex::bulk(2,
                        pika::annotated_function(
                            [](int) {}, "3-schedule-parent-H"))
                //
            );
        });
}

int main(int argc, char* argv[])
{
    apex::apex_options::use_screen_output(true);
    pika::start(nullptr, argc, argv);

    test_senders_execute_no_parent_annotation();
    test_senders_execute_parent_annotation();
    tt::sync_wait(test_senders_schedule_no_parent_annotation());
    tt::sync_wait(test_senders_schedule_parent_annotation());

    // <unknown>
    ex::execute(ex::thread_pool_scheduler{}, [] { pika::finalize(); });

    PIKA_TEST_EQ(pika::stop(), 0);
    return pika::util::report_errors();
}
