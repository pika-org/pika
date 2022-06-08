//  Copyright (c) 2017-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>

#if defined(PIKA_HAVE_THREAD_DESCRIPTION)
#include <pika/threading_base/thread_description.hpp>
#include <pika/threading_base/thread_helpers.hpp>

#if PIKA_HAVE_ITTNOTIFY != 0
#include <pika/modules/itt_notify.hpp>
#elif defined(PIKA_HAVE_APEX)
#include <pika/threading_base/external_timer.hpp>
#elif defined(PIKA_HAVE_TRACY)
#include <Tracy.hpp>
#endif
#endif

#include <string>
#include <type_traits>

namespace pika {
    namespace detail {
        PIKA_EXPORT char const* store_function_annotation(std::string name);
    }    // namespace detail

#if defined(PIKA_HAVE_THREAD_DESCRIPTION)
    ///////////////////////////////////////////////////////////////////////////
#if defined(PIKA_COMPUTE_DEVICE_CODE)
    struct PIKA_NODISCARD scoped_annotation
    {
        PIKA_NON_COPYABLE(scoped_annotation);

        explicit constexpr scoped_annotation(char const*) noexcept {}

        template <typename F>
        explicit PIKA_HOST_DEVICE constexpr scoped_annotation(F&&) noexcept
        {
        }

        // add empty (but non-trivial) destructor to silence warnings
        PIKA_HOST_DEVICE ~scoped_annotation() {}
    };
#elif PIKA_HAVE_ITTNOTIFY != 0
    struct PIKA_NODISCARD scoped_annotation
    {
        PIKA_NON_COPYABLE(scoped_annotation);

        explicit scoped_annotation(char const* name)
          : task_(thread_domain_, pika::util::itt::string_handle(name))
        {
        }
        template <typename F>
        explicit scoped_annotation(F&& f)
          : task_(thread_domain_,
                pika::traits::get_function_annotation_itt<
                    std::decay_t<F>>::call(f))
        {
        }

    private:
        pika::util::itt::thread_domain thread_domain_;
        pika::util::itt::task task_;
    };
#elif defined(PIKA_HAVE_TRACY)
    struct PIKA_NODISCARD scoped_annotation
    {
        PIKA_NON_COPYABLE(scoped_annotation);

        explicit scoped_annotation(char const* annotation)
          : annotation(annotation)
        {
        }

        explicit scoped_annotation(std::string annotation)
          : annotation(detail::store_function_annotation(PIKA_MOVE(annotation)))
        {
        }

        template <typename F,
            typename =
                std::enable_if_t<!std::is_same_v<std::decay_t<F>, std::string>>>
        explicit scoped_annotation(F&& f)
          : annotation(
                pika::traits::get_function_annotation<std::decay_t<F>>::call(f))
        {
        }

    private:
        char const* annotation;

        // We don't use a Zone* macro from Tracy here because they are only
        // meant to be used in function scopes. The Zone* macros make use of
        // e.g.  __FUNCTION__ and other macros that are either unavailable in
        // this scope or are meaningless since they are not evaluated in the
        // scope of the scoped_annotation constructor. We instead manually
        // enable the ScopedZone only if TRACY_ENABLE is set.
#if defined(TRACY_ENABLE)
        tracy::ScopedZone tracy_annotation{
            0, nullptr, 0, nullptr, 0, annotation, strlen(annotation), true};
#endif
    };
#else
    struct PIKA_NODISCARD scoped_annotation
    {
        PIKA_NON_COPYABLE(scoped_annotation);

        explicit scoped_annotation(char const* name)
        {
            auto* self = pika::threads::get_self_ptr();
            if (self != nullptr)
            {
                desc_ = threads::get_thread_id_data(self->get_thread_id())
                            ->set_description(name);
            }

#if defined(PIKA_HAVE_APEX)
            /* update the task wrapper in APEX to use the specified name */
            threads::set_self_timer_data(
                pika::detail::external_timer::update_task(
                    threads::get_self_timer_data(), std::string(name)));
#endif
        }

        explicit scoped_annotation(std::string name)
        {
            auto* self = pika::threads::get_self_ptr();
            if (self != nullptr)
            {
                char const* name_c_str =
#if defined(PIKA_HAVE_APEX)
                    detail::store_function_annotation(name);
#else
                    detail::store_function_annotation(PIKA_MOVE(name));
#endif
                desc_ = threads::get_thread_id_data(self->get_thread_id())
                            ->set_description(name_c_str);
            }

#if defined(PIKA_HAVE_APEX)
            /* update the task wrapper in APEX to use the specified name */
            threads::set_self_timer_data(
                pika::detail::external_timer::update_task(
                    threads::get_self_timer_data(), PIKA_MOVE(name)));
#endif
        }

        template <typename F,
            typename =
                std::enable_if_t<!std::is_same_v<std::decay_t<F>, std::string>>>
        explicit scoped_annotation(F&& f)
        {
            pika::util::thread_description desc(f);

            auto* self = pika::threads::get_self_ptr();
            if (self != nullptr)
            {
                desc_ = threads::get_thread_id_data(self->get_thread_id())
                            ->set_description(desc);
            }

#if defined(PIKA_HAVE_APEX)
            /* update the task wrapper in APEX to use the specified name */
            threads::set_self_timer_data(
                pika::detail::external_timer::update_task(
                    threads::get_self_timer_data(), desc));
#endif
        }

        ~scoped_annotation()
        {
            auto* self = pika::threads::get_self_ptr();
            if (self != nullptr)
            {
                threads::get_thread_id_data(self->get_thread_id())
                    ->set_description(desc_);
            }

#if defined(PIKA_HAVE_APEX)
            threads::set_self_timer_data(
                pika::detail::external_timer::update_task(
                    threads::get_self_timer_data(), desc_));
#endif
        }

        pika::util::thread_description desc_;
    };
#endif

#else
    ///////////////////////////////////////////////////////////////////////////
    struct PIKA_NODISCARD scoped_annotation
    {
        PIKA_NON_COPYABLE(scoped_annotation);

        explicit constexpr scoped_annotation(char const* /*name*/) noexcept {}

        template <typename F>
        explicit PIKA_HOST_DEVICE constexpr scoped_annotation(
            F&& /*f*/) noexcept
        {
        }

        // add empty (but non-trivial) destructor to silence warnings
        PIKA_HOST_DEVICE ~scoped_annotation() {}
    };
#endif
}    // namespace pika
