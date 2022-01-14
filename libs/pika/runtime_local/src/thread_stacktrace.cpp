//  Copyright (c) 2014-2016 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/config.hpp>
#include <pika/modules/format.hpp>
#include <pika/runtime_local/thread_pool_helpers.hpp>
#include <pika/runtime_local/thread_stacktrace.hpp>
#include <pika/threading_base/thread_data.hpp>

#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace pika { namespace util { namespace debug {

    // ------------------------------------------------------------------------
    // return a vector of suspended/other task Ids
    std::vector<pika::threads::thread_id_type> get_task_ids(
        pika::threads::thread_schedule_state state)
    {
        std::vector<pika::threads::thread_id_type> thread_ids_vector;
        //
        pika::threads::enumerate_threads(
            [&thread_ids_vector](pika::threads::thread_id_type id) -> bool {
                thread_ids_vector.push_back(id);
                return true;    // always continue enumeration
            },
            state);
        return thread_ids_vector;
    }

    // ------------------------------------------------------------------------
    // return a vector of thread data structure pointers for suspended tasks
    std::vector<pika::threads::thread_data*> get_task_data(
        pika::threads::thread_schedule_state state)
    {
        std::vector<pika::threads::thread_data*> thread_data_vector;
        //
        pika::threads::enumerate_threads(
            [&thread_data_vector](pika::threads::thread_id_type id) -> bool {
                pika::threads::thread_data* data = get_thread_id_data(id);
                thread_data_vector.push_back(data);
                return true;    // always continue enumeration
            },
            state);
        return thread_data_vector;
    }

    // ------------------------------------------------------------------------
    // return string containing the stack backtrace for suspended tasks
    std::string suspended_task_backtraces()
    {
        std::vector<pika::threads::thread_data*> tlist =
            get_task_data(pika::threads::thread_schedule_state::suspended);
        //
        std::stringstream tmp;
        //
        int count = 0;
        for (const auto& data : tlist)
        {
            pika::util::format_to(tmp, "Stack trace {} : {:#012x} : \n{}\n\n\n",
                count, reinterpret_cast<uintptr_t>(data),
#ifdef PIKA_HAVE_THREAD_BACKTRACE_ON_SUSPENSION
                data->backtrace()
#else
                "Enable PIKA_WITH_THREAD_BACKTRACE_ON_SUSPENSION in CMake"
#endif

            );
            ++count;
        }
        return tmp.str();
    }
}}}    // namespace pika::util::debug
