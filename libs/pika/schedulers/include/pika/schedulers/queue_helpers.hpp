////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2017-2018 John Biddiscombe
//  Copyright (c) 2007-2016 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <pika/local/config.hpp>
#include <pika/modules/logging.hpp>
#include <pika/schedulers/deadlock_detection.hpp>
#include <pika/threading_base/thread_data.hpp>
#include <pika/threading_base/thread_queue_init_parameters.hpp>
#include <pika/type_support/unused.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace pika { namespace threads { namespace policies {

    ///////////////////////////////////////////////////////////////////////////////
    namespace detail {
        ///////////////////////////////////////////////////////////////////////////
        // debug helper function, logs all suspended threads
        // this returns true if all threads in the map are currently suspended
        template <typename Map>
        bool dump_suspended_threads(std::size_t num_thread, Map& tm,
            std::int64_t& idle_loop_count, bool running) PIKA_COLD;

        template <typename Map>
        bool dump_suspended_threads(std::size_t num_thread, Map& tm,
            std::int64_t& idle_loop_count, bool running)
        {
#if !defined(PIKA_HAVE_THREAD_MINIMAL_DEADLOCK_DETECTION)
            PIKA_UNUSED(num_thread);
            PIKA_UNUSED(tm);
            PIKA_UNUSED(idle_loop_count);
            PIKA_UNUSED(running);    //-V601
            return false;
#else
            if (!get_minimal_deadlock_detection_enabled())
                return false;

            // attempt to output possibly deadlocked threads occasionally only
            if (PIKA_LIKELY((idle_loop_count++ % PIKA_IDLE_LOOP_COUNT_MAX) != 0))
                return false;

            bool result = false;
            bool collect_suspended = true;

            bool logged_headline = false;
            typename Map::const_iterator end = tm.end();
            for (typename Map::const_iterator it = tm.begin(); it != end; ++it)
            {
                threads::thread_data const* thrd = get_thread_id_data(*it);
                threads::thread_schedule_state state =
                    thrd->get_state().state();
                threads::thread_schedule_state marked_state =
                    thrd->get_marked_state();

                if (state != marked_state)
                {
                    // log each thread only once
                    if (!logged_headline)
                    {
                        if (running)
                        {
                            LTM_(warning).format("Listing suspended threads "
                                                 "while queue ({}) is empty:",
                                num_thread);
                        }
                        else
                        {
                            LPIKA_CONSOLE_(pika::util::logging::level::warning)
                                .format("  [TM] Listing suspended threads "
                                        "while queue ({}) is empty:\n",
                                    num_thread);
                        }
                        logged_headline = true;
                    }

                    if (running)
                    {
                        LTM_(warning)
                            .format("queue({}): {}({:08x}.{:02x}/{:08x})",
                                num_thread, get_thread_state_name(state), *it,
                                thrd->get_thread_phase(),
                                thrd->get_component_id())
#ifdef PIKA_HAVE_THREAD_PARENT_REFERENCE
                            .format(" P{:08x}", thrd->get_parent_thread_id())
#endif
                            .format(": {}: {}", thrd->get_description(),
                                thrd->get_lco_description());
                    }
                    else
                    {
                        LPIKA_CONSOLE_(pika::util::logging::level::warning)
                            .format("queue({}): {}({:08x}.{:02x}/{:08x})",
                                num_thread, get_thread_state_name(state), *it,
                                thrd->get_thread_phase(),
                                thrd->get_component_id())
#ifdef PIKA_HAVE_THREAD_PARENT_REFERENCE
                            .format(" P{:08x}", thrd->get_parent_thread_id())
#endif
                            .format(": {}: {}", thrd->get_description(),
                                thrd->get_lco_description());
                    }
                    thrd->set_marked_state(state);

                    // result should be true if we found only suspended threads
                    if (collect_suspended)
                    {
                        switch (state)
                        {
                        case threads::thread_schedule_state::suspended:
                            result = true;    // at least one is suspended
                            break;

                        case threads::thread_schedule_state::pending:
                        case threads::thread_schedule_state::active:
                            result =
                                false;    // one is active, no deadlock (yet)
                            collect_suspended = false;
                            break;

                        default:
                            // If the thread is terminated we don't care too much
                            // anymore.
                            break;
                        }
                    }
                }
            }
            return result;
#endif
        }
    }    // namespace detail

}}}    // namespace pika::threads::policies
