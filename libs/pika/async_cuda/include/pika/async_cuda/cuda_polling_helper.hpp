//  Copyright (c) 2021 ETH Zurich
//  Copyright (c) 2020 John Biddiscombe
//  Copyright (c) 2016 Thomas Heller
//  Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>
#include <pika/async_cuda/detail/cuda_event_callback.hpp>
#include <pika/runtime_local/runtime_local_fwd.hpp>
#include <pika/threading_base/thread_pool_base.hpp>

#include <string>

namespace pika { namespace cuda { namespace experimental {
    // -----------------------------------------------------------------
    // This RAII helper class enables polling for a scoped block
    struct PIKA_NODISCARD enable_user_polling
    {
        enable_user_polling(std::string const& pool_name = "")
          : pool_name_(pool_name)
        {
            // install polling loop on requested thread pool
            if (pool_name_.empty())
            {
                detail::register_polling(pika::resource::get_thread_pool(0));
            }
            else
            {
                detail::register_polling(
                    pika::resource::get_thread_pool(pool_name_));
            }
        }

        ~enable_user_polling()
        {
            if (pool_name_.empty())
            {
                detail::unregister_polling(pika::resource::get_thread_pool(0));
            }
            else
            {
                detail::unregister_polling(
                    pika::resource::get_thread_pool(pool_name_));
            }
        }

    private:
        std::string pool_name_;
    };

}}}    // namespace pika::cuda::experimental
