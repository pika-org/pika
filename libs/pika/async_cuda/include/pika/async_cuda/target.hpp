///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Thomas Heller
//  Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <pika/local/config.hpp>

#include <pika/allocator_support/allocator_deleter.hpp>
#include <pika/assert.hpp>
#include <pika/async_cuda/cuda_future.hpp>
#include <pika/async_cuda/get_targets.hpp>
#include <pika/futures/future.hpp>
#include <pika/futures/traits/future_access.hpp>
#include <pika/synchronization/spinlock.hpp>
#include <pika/type_support/unused.hpp>

#include <pika/async_cuda/custom_gpu_api.hpp>

#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <pika/local/config/warnings_prefix.hpp>

namespace pika { namespace cuda { namespace experimental {

    ///////////////////////////////////////////////////////////////////////////
    struct target
    {
    public:
        struct PIKA_EXPORT native_handle_type
        {
            typedef pika::lcos::local::spinlock mutex_type;

            native_handle_type(int device = 0);

            ~native_handle_type();

            native_handle_type(native_handle_type const& rhs) noexcept;
            native_handle_type(native_handle_type&& rhs) noexcept;

            native_handle_type& operator=(
                native_handle_type const& rhs) noexcept;
            native_handle_type& operator=(native_handle_type&& rhs) noexcept;

            cudaStream_t get_stream() const;

            int get_device() const noexcept
            {
                return device_;
            }

            std::size_t processing_units() const
            {
                return processing_units_;
            }

            std::size_t processor_family() const
            {
                return processor_family_;
            }

            std::string processor_name() const
            {
                return processor_name_;
            }

            void reset() noexcept;

        private:
            void init_processing_units();
            friend struct target;

            mutable mutex_type mtx_;
            int device_;
            std::size_t processing_units_;
            std::size_t processor_family_;
            std::string processor_name_;
            mutable cudaStream_t stream_;
        };

        // Constructs default target
        PIKA_HOST_DEVICE target()
          : handle_()
        {
        }

        // Constructs target from a given device ID
        explicit PIKA_HOST_DEVICE target(int device)
          : handle_(device)
        {
        }

        PIKA_HOST_DEVICE target(target const& rhs) noexcept
          : handle_(rhs.handle_)
        {
        }

        PIKA_HOST_DEVICE target(target&& rhs) noexcept
          : handle_(PIKA_MOVE(rhs.handle_))
        {
        }

        PIKA_HOST_DEVICE target& operator=(target const& rhs) noexcept
        {
            if (&rhs != this)
            {
                handle_ = rhs.handle_;
            }
            return *this;
        }

        PIKA_HOST_DEVICE target& operator=(target&& rhs) noexcept
        {
            if (&rhs != this)
            {
                handle_ = PIKA_MOVE(rhs.handle_);
            }
            return *this;
        }

        PIKA_HOST_DEVICE
        native_handle_type& native_handle() noexcept
        {
            return handle_;
        }
        PIKA_HOST_DEVICE
        native_handle_type const& native_handle() const noexcept
        {
            return handle_;
        }

        void synchronize() const;

        pika::future<void> get_future_with_event() const;
        pika::future<void> get_future_with_callback() const;

        template <typename Allocator>
        pika::future<void> get_future_with_event(Allocator const& alloc) const
        {
            return detail::get_future_with_event(alloc, handle_.get_stream());
        }

        template <typename Allocator>
        pika::future<void> get_future_with_callback(Allocator const& alloc) const
        {
            return detail::get_future_with_callback(
                alloc, handle_.get_stream());
        }

        static std::vector<target> get_local_targets()
        {
            return cuda::experimental::get_local_targets();
        }

        friend bool operator==(target const& lhs, target const& rhs)
        {
            return lhs.handle_.get_device() == rhs.handle_.get_device();
        }

    private:
        native_handle_type handle_;
    };

    using detail::get_future_with_callback;
    PIKA_EXPORT target& get_default_target();
}}}    // namespace pika::cuda::experimental

namespace pika { namespace compute { namespace cuda {
    using target PIKA_DEPRECATED_V(0, 1,
        "pika::compute::cuda::target is deprecated. Please use "
        "pika::cuda::experimental::target instead.") =
        pika::cuda::experimental::target;
}}}    // namespace pika::compute::cuda

#include <pika/local/config/warnings_suffix.hpp>
