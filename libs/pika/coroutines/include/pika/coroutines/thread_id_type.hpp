//  Copyright (c) 2018 Thomas Heller
//  Copyright (c) 2019-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file pika/coroutines/thread_id_type.hpp

#pragma once

#include <pika/config.hpp>
#include <pika/assert.hpp>
#include <pika/memory/intrusive_ptr.hpp>
#include <pika/modules/memory.hpp>
#include <pika/thread_support/atomic_count.hpp>

#include <fmt/format.h>
#include <fmt/ostream.h>
#include <fmt/printf.h>

#include <cstddef>
#include <functional>
#include <iosfwd>
#include <string_view>
#include <utility>

namespace pika::threads::detail {
    // same as below, just not holding a reference count
    struct thread_id
    {
    private:
        using thread_id_repr = void*;

    public:
        thread_id() noexcept = default;

        thread_id(thread_id const&) = default;
        thread_id& operator=(thread_id const&) = default;

        constexpr thread_id(thread_id&& rhs) noexcept
          : thrd_(rhs.thrd_)
        {
            rhs.thrd_ = nullptr;
        }
        constexpr thread_id& operator=(thread_id&& rhs) noexcept
        {
            thrd_ = rhs.thrd_;
            rhs.thrd_ = nullptr;
            return *this;
        }

        explicit constexpr thread_id(thread_id_repr const& thrd) noexcept
          : thrd_(thrd)
        {
        }
        constexpr thread_id& operator=(thread_id_repr const& rhs) noexcept
        {
            thrd_ = rhs;
            return *this;
        }

        explicit constexpr operator bool() const noexcept { return nullptr != thrd_; }

        constexpr thread_id_repr get() const noexcept { return thrd_; }

        constexpr void reset() noexcept { thrd_ = nullptr; }

        friend constexpr bool operator==(std::nullptr_t, thread_id const& rhs) noexcept
        {
            return nullptr == rhs.thrd_;
        }

        friend constexpr bool operator!=(std::nullptr_t, thread_id const& rhs) noexcept
        {
            return nullptr != rhs.thrd_;
        }

        friend constexpr bool operator==(thread_id const& lhs, std::nullptr_t) noexcept
        {
            return nullptr == lhs.thrd_;
        }

        friend constexpr bool operator!=(thread_id const& lhs, std::nullptr_t) noexcept
        {
            return nullptr != lhs.thrd_;
        }

        friend constexpr bool operator==(thread_id const& lhs, thread_id const& rhs) noexcept
        {
            return lhs.thrd_ == rhs.thrd_;
        }

        friend constexpr bool operator!=(thread_id const& lhs, thread_id const& rhs) noexcept
        {
            return lhs.thrd_ != rhs.thrd_;
        }

        friend constexpr bool operator<(thread_id const& lhs, thread_id const& rhs) noexcept
        {
            return std::less<thread_id_repr>{}(lhs.thrd_, rhs.thrd_);
        }

        friend constexpr bool operator>(thread_id const& lhs, thread_id const& rhs) noexcept
        {
            return std::less<thread_id_repr>{}(rhs.thrd_, lhs.thrd_);
        }

        friend constexpr bool operator<=(thread_id const& lhs, thread_id const& rhs) noexcept
        {
            return !(rhs > lhs);
        }

        friend constexpr bool operator>=(thread_id const& lhs, thread_id const& rhs) noexcept
        {
            return !(rhs < lhs);
        }

        template <typename Char, typename Traits>
        friend std::basic_ostream<Char, Traits>&
        operator<<(std::basic_ostream<Char, Traits>& os, thread_id const& id)
        {
            os << id.get();
            return os;
        }

    private:
        thread_id_repr thrd_ = nullptr;
    };

    ///////////////////////////////////////////////////////////////////////////
    enum class thread_id_addref
    {
        yes,
        no
    };

    struct thread_data_reference_counting;

    void intrusive_ptr_add_ref(thread_data_reference_counting* p);
    void intrusive_ptr_release(thread_data_reference_counting* p);

    struct thread_data_reference_counting
    {
        // the initial reference count is one by default as each newly
        // created thread will be held alive by the variable returned from
        // the creation function;
        explicit thread_data_reference_counting([[maybe_unused]] thread_id_addref addref = thread_id_addref::yes)
          // : count_(addref == thread_id_addref::yes)
        {
        }

        virtual ~thread_data_reference_counting() = default;
        virtual void destroy_thread() = 0;

        // reference counting
        friend void intrusive_ptr_add_ref(thread_data_reference_counting* p) {
            // ++p->count_;
        }

        friend void intrusive_ptr_release(thread_data_reference_counting* p)
        {
            // PIKA_ASSERT(p->count_ != 0);
            // if (--p->count_ == 0)
            // {
            //     // give this object back to the system
            //     p->destroy_thread();
            // }
        }

        // ::pika::detail::atomic_count count_;
    };

    ///////////////////////////////////////////////////////////////////////////
    struct thread_id_ref
    {
    private:
        using thread_id_repr = pika::intrusive_ptr<detail::thread_data_reference_counting>;

    public:
        thread_id_ref() noexcept = default;

        thread_id_ref(thread_id_ref const&) = default;
        thread_id_ref& operator=(thread_id_ref const&) = default;

        thread_id_ref(thread_id_ref&& rhs) noexcept = default;
        thread_id_ref& operator=(thread_id_ref&& rhs) noexcept = default;

        explicit thread_id_ref(thread_id_repr const& thrd) noexcept
          : thrd_(thrd)
        {
        }
        explicit thread_id_ref(thread_id_repr&& thrd) noexcept
          : thrd_(PIKA_MOVE(thrd))
        {
        }

        thread_id_ref& operator=(thread_id_repr const& rhs) noexcept
        {
            thrd_ = rhs;
            return *this;
        }
        thread_id_ref& operator=(thread_id_repr&& rhs) noexcept
        {
            thrd_ = PIKA_MOVE(rhs);
            return *this;
        }

        using thread_repr = detail::thread_data_reference_counting;

        explicit thread_id_ref(
            thread_repr* thrd, thread_id_addref addref = thread_id_addref::yes) noexcept
          : thrd_(thrd, addref == thread_id_addref::yes)
        {
        }

        thread_id_ref& operator=(thread_repr* rhs) noexcept
        {
            thrd_.reset(rhs);
            return *this;
        }

        thread_id_ref(thread_id const& noref)
          : thrd_(static_cast<thread_repr*>(noref.get()))
        {
        }

        thread_id_ref(thread_id&& noref) noexcept
          : thrd_(static_cast<thread_repr*>(noref.get()))
        {
            noref.reset();
        }

        thread_id_ref& operator=(thread_id const& noref)
        {
            thrd_.reset(static_cast<thread_repr*>(noref.get()));
            return *this;
        }

        thread_id_ref& operator=(thread_id&& noref) noexcept
        {
            thrd_.reset(static_cast<thread_repr*>(noref.get()));
            noref.reset();
            return *this;
        }

        explicit operator bool() const noexcept { return nullptr != thrd_; }

        thread_id noref() const noexcept { return thread_id(thrd_.get()); }

        thread_id_repr& get() & noexcept { return thrd_; }
        thread_id_repr&& get() && noexcept { return PIKA_MOVE(thrd_); }

        thread_id_repr const& get() const& noexcept { return thrd_; }

        void reset() noexcept { thrd_.reset(); }

        void reset(thread_repr* thrd, bool add_ref = true) noexcept { thrd_.reset(thrd, add_ref); }

        constexpr thread_repr* detach() noexcept { return thrd_.detach(); }

        friend bool operator==(std::nullptr_t, thread_id_ref const& rhs) noexcept
        {
            return nullptr == rhs.thrd_;
        }

        friend bool operator!=(std::nullptr_t, thread_id_ref const& rhs) noexcept
        {
            return nullptr != rhs.thrd_;
        }

        friend bool operator==(thread_id_ref const& lhs, std::nullptr_t) noexcept
        {
            return nullptr == lhs.thrd_;
        }

        friend bool operator!=(thread_id_ref const& lhs, std::nullptr_t) noexcept
        {
            return nullptr != lhs.thrd_;
        }

        friend bool operator==(thread_id_ref const& lhs, thread_id_ref const& rhs) noexcept
        {
            return lhs.thrd_ == rhs.thrd_;
        }

        friend bool operator!=(thread_id_ref const& lhs, thread_id_ref const& rhs) noexcept
        {
            return lhs.thrd_ != rhs.thrd_;
        }

        friend bool operator<(thread_id_ref const& lhs, thread_id_ref const& rhs) noexcept
        {
            return std::less<thread_repr const*>{}(lhs.thrd_.get(), rhs.thrd_.get());
        }

        friend bool operator>(thread_id_ref const& lhs, thread_id_ref const& rhs) noexcept
        {
            return std::less<thread_repr const*>{}(rhs.thrd_.get(), lhs.thrd_.get());
        }

        friend bool operator<=(thread_id_ref const& lhs, thread_id_ref const& rhs) noexcept
        {
            return !(rhs > lhs);
        }

        friend bool operator>=(thread_id_ref const& lhs, thread_id_ref const& rhs) noexcept
        {
            return !(rhs < lhs);
        }

        template <typename Char, typename Traits>
        friend std::basic_ostream<Char, Traits>&
        operator<<(std::basic_ostream<Char, Traits>& os, thread_id_ref const& id)
        {
            os << id.get();
            return os;
        }

    private:
        thread_id_repr thrd_;
    };

    inline constexpr thread_id const invalid_thread_id;
}    // namespace pika::threads::detail

namespace std {
    template <>
    struct hash<::pika::threads::detail::thread_id>
    {
        std::size_t PIKA_STATIC_CALL_OPERATOR(::pika::threads::detail::thread_id const& v) noexcept
        {
            std::hash<std::size_t> hasher_;
            return hasher_(reinterpret_cast<std::size_t>(v.get()));
        }
    };

    template <>
    struct hash<::pika::threads::detail::thread_id_ref>
    {
        std::size_t PIKA_STATIC_CALL_OPERATOR(
            ::pika::threads::detail::thread_id_ref const& v) noexcept
        {
            std::hash<std::size_t> hasher_;
            return hasher_(reinterpret_cast<std::size_t>(v.get().get()));
        }
    };
}    // namespace std

template <>
struct fmt::formatter<pika::threads::detail::thread_id> : fmt::formatter<void*>
{
    template <typename FormatContext>
    auto format(pika::threads::detail::thread_id id, FormatContext& ctx) const
    {
        return fmt::formatter<void*>::format(static_cast<void*>(id.get()), ctx);
    }
};

template <>
struct fmt::formatter<pika::threads::detail::thread_id_ref> : fmt::formatter<void*>
{
    template <typename FormatContext>
    auto format(pika::threads::detail::thread_id_ref id, FormatContext& ctx) const
    {
        return fmt::formatter<void*>::format(static_cast<void*>(id.noref().get()), ctx);
    }
};
