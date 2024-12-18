//  Copyright (c) 2021 ETH Zurich
//  Copyright (c) 2020 John Biddiscombe
//  Copyright (c) 2016 Thomas Heller
//  Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#include <pika/async_cuda/detail/cuda_event_callback.hpp>
#include <pika/runtime/runtime_fwd.hpp>
#include <pika/threading_base/thread_pool_base.hpp>

#include <string>

namespace pika::cuda::experimental {

    PIKA_EXPORT std::string const& get_pool_name();
    PIKA_EXPORT void set_pool_name(std::string const&);

    /// \brief Enable CUDA polling on the given thread pool.
    ///
    /// RAII helper class to enable and disable polling of CUDA events on the given pool. Enabling
    /// polling is a requirement to signal completion of work submitted to the \ref cuda_scheduler.
    ///
    /// There is no detection of whether polling is already enabled or disabled, or if \ref
    /// enable_user_polling is nested. The constructor and destructor will unconditionally register
    /// and unregister polling, respectively.
    class [[nodiscard]] enable_user_polling
    {
    public:
        /// \brief Start polling for CUDA events on the given thread pool.
        ///
        /// \param pool_name The name of the thread pool to enable polling on. The default is to use
        /// the default thread pool.
        enable_user_polling(std::string const& pool_name = "")
          : pool_name_(pool_name)
        {
            // install polling loop on requested thread pool
            if (pool_name_.empty())
            {
                detail::register_polling(pika::resource::get_thread_pool(0));
                set_pool_name(pika::resource::get_pool_name(0));
            }
            else
            {
                detail::register_polling(pika::resource::get_thread_pool(pool_name_));
                set_pool_name(pool_name_);
            }
        }

        /// \brief Stop polling for CUDA events.
        ///
        /// The destructor will not wait for work submitted to a \ref cuda_scheduler to complete.
        /// The user must ensure that work completes before disabling polling.
        ~enable_user_polling()
        {
            if (pool_name_.empty())
            {
                detail::unregister_polling(pika::resource::get_thread_pool(0));
            }
            else { detail::unregister_polling(pika::resource::get_thread_pool(pool_name_)); }
        }

    private:
        std::string pool_name_;
    };

}    // namespace pika::cuda::experimental
