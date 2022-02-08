//  Copyright (c) 2007-2013 Kevin Huck
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#include <pika/config.hpp>
#ifdef PIKA_HAVE_APEX
#include <pika/assert.hpp>
#include <pika/threading_base/external_timer.hpp>
#include <pika/threading_base/thread_data.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace pika::detail::external_timer {
    std::shared_ptr<task_wrapper> new_task(
        pika::util::thread_description const& description,
        std::uint32_t /* parent_locality_id */,
        threads::thread_id_type parent_task)
    {
        std::shared_ptr<task_wrapper> parent_wrapper = nullptr;
        if (parent_task != nullptr)
        {
            parent_wrapper = get_thread_id_data(parent_task)->get_timer_data();
        }

        if (description.kind() ==
            pika::util::thread_description::data_type_description)
        {
            return external_timer::new_task(
                description.get_description(), UINTMAX_MAX, parent_wrapper);
        }
        else
        {
            PIKA_ASSERT(description.kind() ==
                pika::util::thread_description::data_type_address);
            return external_timer::new_task(
                description.get_address(), UINTMAX_MAX, parent_wrapper);
        }
    }

    std::shared_ptr<task_wrapper> update_task(
        std::shared_ptr<task_wrapper> wrapper,
        pika::util::thread_description const& description)
    {
        if (wrapper == nullptr)
        {
            threads::thread_id_type parent_task;
            // doesn't matter which locality we use, the parent is null
            return external_timer::new_task(description, 0, parent_task);
        }
        else if (description.kind() ==
            pika::util::thread_description::data_type_description)
        {
            // Disambiguate the call by making a temporary string object
            return external_timer::update_task(
                wrapper, std::string(description.get_description()));
        }
        else
        {
            PIKA_ASSERT(description.kind() ==
                pika::util::thread_description::data_type_address);
            return external_timer::update_task(
                wrapper, description.get_address());
        }
    }
}    // namespace pika::detail::external_timer
#endif
