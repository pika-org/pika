//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/assert.hpp>
#include <pika/errors/error.hpp>
#include <pika/errors/throw_exception.hpp>
#include <pika/execution_base/operation_state.hpp>
#include <pika/execution_base/receiver.hpp>
#include <pika/execution_base/sender.hpp>
#include <pika/type_support/detail/with_result_of.hpp>
#include <pika/type_support/pack.hpp>

#include <cstddef>
#include <cstring>
#include <exception>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>

#include <pika/config/warnings_prefix.hpp>

// This silences what seems to be a bogus warning emitted by GCC. The type
// erased wrappers below use the small-buffer optimization (SBO). When the
// stored type is bigger than the embedded storage heap allocation happens, but
// there are still code paths that in theory can access the embedded storage and
// so read/write out of bounds. However, since access to the embedded storage is
// guarded by whether the embedded storage can actually be used (i.e. if the
// stored type is small enough) that path should never be taken.
#if defined(__GNUC__)
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Warray-bounds"
#endif

// SBO is currently disabled as it seems to be buggy in certain use cases. It can still be
// explicitly forced to on by defining PIKA_DETAIL_ENABLE_ANY_SENDER_SBO.

namespace pika::detail {
    template <typename T>
    struct empty_vtable_type
    {
        static_assert(sizeof(T) == 0, "No empty vtable type defined for given type T");
    };

    template <typename T>
    using empty_vtable_t = typename empty_vtable_type<T>::type;

    // clang 14 attempts to promote all constexpr variables when compiling for
    // accelerators (CUDA or HIP). In this case it errors out trying to make
    // empty_vtable a device-side variable instead of leaving it as a host-only
    // variable. We don't need empty_vtable on the device, so we simply fall
    // back to the other implementation in those cases.
    //
    // See:
    // - https://github.com/llvm/llvm-project/issues/53780
    // - https://github.com/llvm/llvm-project/commit/73b22935a7a863679021598db6a45fcfb62cd321
    // - https://reviews.llvm.org/D119615
#if defined(PIKA_HAVE_CXX20_TRIVIAL_VIRTUAL_DESTRUCTOR) &&                                         \
    (!defined(PIKA_HAVE_GPU_SUPPORT) ||                                                            \
        defined(PIKA_HAVE_CXX20_TRIVIAL_VIRTUAL_DESTRUCTOR_GPU)) &&                                \
    !(defined(PIKA_COMPUTE_CODE) && defined(PIKA_CLANG_VERSION) &&                                 \
        (PIKA_CLANG_VERSION >= 140000) && (PIKA_CLANG_VERSION < 150000))
    template <typename T>
    inline constexpr empty_vtable_t<T> empty_vtable{};

    template <typename T>
    T const* get_empty_vtable()
    {
        static_assert(std::is_base_of_v<T, empty_vtable_t<T>>,
            "Given empty vtable type should be a base of T");

        return &empty_vtable<T>;
    }
#else
    template <typename T>
    T const* get_empty_vtable()
    {
        static_assert(std::is_base_of_v<T, empty_vtable_t<T>>,
            "Given empty vtable type should be a base of T");

        static empty_vtable_t<T> empty;
        return &empty;
    }
#endif
    template <typename Base, std::size_t EmbeddedStorageSize,
        std::size_t AlignmentSize = sizeof(void*)>
    class copyable_sbo_storage;

    template <typename Base, std::size_t EmbeddedStorageSize,
        std::size_t AlignmentSize = sizeof(void*)>
    class movable_sbo_storage
    {
    protected:
        using base_type = Base;
        static constexpr std::size_t embedded_storage_size = EmbeddedStorageSize;
        static constexpr std::size_t alignment_size = AlignmentSize;

        // The union has two members:
        // - embedded_storage: Embedded storage size array used for types that
        //   are at most embedded_storage_size bytes large, and require at most
        //   alignment_size alignment.
        // - heap_storage: A pointer to base_type that is used when objects
        //   don't fit in the embedded storage.
#if defined(PIKA_DETAIL_ENABLE_ANY_SENDER_SBO)
        union
        {
            alignas(alignment_size) unsigned char embedded_storage[embedded_storage_size];
#endif
            base_type* heap_storage = nullptr;
#if defined(PIKA_DETAIL_ENABLE_ANY_SENDER_SBO)
        };
#endif
        base_type* object = const_cast<base_type*>(get_empty_vtable<base_type>());

        // Returns true when it's safe to use the embedded storage, i.e.
        // when the size and alignment of Impl are small enough.
        template <typename Impl>
        static constexpr bool can_use_embedded_storage()
        {
#if defined(PIKA_DETAIL_ENABLE_ANY_SENDER_SBO)
            constexpr bool fits_storage = sizeof(std::decay_t<Impl>) <= embedded_storage_size;
            constexpr bool sufficiently_aligned = alignof(std::decay_t<Impl>) <= alignment_size;
            return fits_storage && sufficiently_aligned;
#else
            return false;
#endif
        }

        bool using_embedded_storage() const noexcept
        {
#if defined(PIKA_DETAIL_ENABLE_ANY_SENDER_SBO)
            return object == reinterpret_cast<base_type const*>(&embedded_storage);
#else
            return false;
#endif
        }

        void reset_vtable() { object = const_cast<base_type*>(get_empty_vtable<base_type>()); }

        void release() noexcept
        {
            PIKA_ASSERT(!empty());

#if defined(PIKA_DETAIL_ENABLE_ANY_SENDER_SBO)
            if (using_embedded_storage()) { get().~base_type(); }
            else
#endif
            {
                delete heap_storage;
                heap_storage = nullptr;
            }

            reset_vtable();
        }

        void move_assign(movable_sbo_storage&& other) &

        {
            PIKA_ASSERT(&other != this);
            PIKA_ASSERT(empty());

            if (!other.empty())
            {
#if defined(PIKA_DETAIL_ENABLE_ANY_SENDER_SBO)
                if (other.using_embedded_storage())
                {
                    auto p = reinterpret_cast<base_type*>(&embedded_storage);
                    other.get().move_into(p);
                    object = p;
                }
                else
#endif
                {
                    heap_storage = other.heap_storage;
                    other.heap_storage = nullptr;
                    object = heap_storage;
                }

                other.reset_vtable();
            }
        }

        template <typename T>
        void move_assign(copyable_sbo_storage<T, embedded_storage_size, alignment_size>&& other)
        {
            static_assert(std::is_base_of_v<base_type, T>);

            PIKA_ASSERT(static_cast<void*>(&other) != static_cast<void*>(this));
            PIKA_ASSERT(empty());

            if (!other.empty())
            {
#if defined(PIKA_DETAIL_ENABLE_ANY_SENDER_SBO)
                if (other.using_embedded_storage())
                {
                    auto p = reinterpret_cast<base_type*>(&embedded_storage);
                    other.get().move_into(p);
                    object = p;
                }
                else
#endif
                {
                    heap_storage = other.heap_storage;
                    other.heap_storage = nullptr;
                    object = heap_storage;
                }

                other.reset_vtable();
            }
        }

    public:
        movable_sbo_storage() = default;

        ~movable_sbo_storage() noexcept
        {
            if (!empty()) { release(); }
        }

        movable_sbo_storage(movable_sbo_storage&& other) { move_assign(std::move(other)); }

        template <typename T>
        explicit movable_sbo_storage(
            copyable_sbo_storage<T, embedded_storage_size, alignment_size>&& other)
        {
            static_assert(std::is_base_of_v<base_type, T>);

            move_assign(std::move(other));
        }

        movable_sbo_storage& operator=(movable_sbo_storage&& other)
        {
            if (&other != this)
            {
                if (!empty()) { release(); }

                move_assign(std::move(other));
            }
            return *this;
        }

        template <typename T>
        movable_sbo_storage&
        operator=(copyable_sbo_storage<T, embedded_storage_size, alignment_size>&& other)
        {
            static_assert(std::is_base_of_v<base_type, T>);

            if (static_cast<void*>(&other) != static_cast<void*>(this))
            {
                if (!empty()) { release(); }

                move_assign(std::move(other));
            }
            return *this;
        }

        movable_sbo_storage(movable_sbo_storage const&) = delete;
        movable_sbo_storage& operator=(movable_sbo_storage const&) = delete;

        bool empty() const noexcept { return get().empty(); }

        base_type const& get() const noexcept { return *object; }

        base_type& get() noexcept { return *object; }

        template <typename Impl, typename... Ts>
        void store(Ts&&... ts)
        {
            if (!empty()) { release(); }

#if defined(PIKA_DETAIL_ENABLE_ANY_SENDER_SBO)
            if constexpr (can_use_embedded_storage<Impl>())
            {
                Impl* p = reinterpret_cast<Impl*>(&embedded_storage);
                new (p) Impl(std::forward<Ts>(ts)...);
                object = p;
            }
            else
#endif
            {
                heap_storage = new Impl(std::forward<Ts>(ts)...);
                object = heap_storage;
            }
        }

        void reset()
        {
            if (!empty()) { release(); }
        }
    };

    template <typename Base, std::size_t EmbeddedStorageSize, std::size_t AlignmentSize>
    class copyable_sbo_storage
      : public movable_sbo_storage<Base, EmbeddedStorageSize, AlignmentSize>
    {
        template <typename Base2, std::size_t EmbeddedStorageSize2, std::size_t AlignmentSize2>
        friend class movable_sbo_storage;

        using storage_base_type = movable_sbo_storage<Base, EmbeddedStorageSize, AlignmentSize>;

        using typename storage_base_type::base_type;

#if defined(PIKA_DETAIL_ENABLE_ANY_SENDER_SBO)
        using storage_base_type::embedded_storage;
#endif
        using storage_base_type::heap_storage;
        using storage_base_type::object;
        using storage_base_type::release;
        using storage_base_type::using_embedded_storage;

        void copy_assign(copyable_sbo_storage const& other) &
        {
            PIKA_ASSERT(&other != this);
            PIKA_ASSERT(empty());

            if (!other.empty())
            {
#if defined(PIKA_DETAIL_ENABLE_ANY_SENDER_SBO)
                if (other.using_embedded_storage())
                {
                    base_type* p = reinterpret_cast<base_type*>(&embedded_storage);
                    other.get().clone_into(p);
                    object = p;
                }
                else
#endif
                {
                    heap_storage = other.get().clone();
                    object = heap_storage;
                }
            }
        }

    public:
        using storage_base_type::empty;
        using storage_base_type::get;
        using storage_base_type::reset;
        using storage_base_type::store;

        copyable_sbo_storage() = default;
        ~copyable_sbo_storage() noexcept = default;
        copyable_sbo_storage(copyable_sbo_storage&&) = default;
        copyable_sbo_storage& operator=(copyable_sbo_storage&&) = default;

        copyable_sbo_storage(copyable_sbo_storage const& other)
          : storage_base_type()
        {
            copy_assign(other);
        }

        copyable_sbo_storage& operator=(copyable_sbo_storage const& other)
        {
            if (&other != this)
            {
                if (!empty()) { release(); }
                copy_assign(other);
            }
            return *this;
        }
    };
}    // namespace pika::detail

namespace pika::execution::experimental::detail {
    struct PIKA_EXPORT any_operation_state_holder_base
    {
        virtual ~any_operation_state_holder_base() noexcept = default;
        virtual bool empty() const noexcept;
        virtual void start() & noexcept = 0;
    };

    struct PIKA_EXPORT empty_any_operation_state_holder_state final
      : any_operation_state_holder_base
    {
        bool empty() const noexcept override;
        void start() & noexcept override;
    };
}    // namespace pika::execution::experimental::detail

namespace pika::detail {
    template <>
    struct empty_vtable_type<pika::execution::experimental::detail::any_operation_state_holder_base>
    {
        using type = pika::execution::experimental::detail::empty_any_operation_state_holder_state;
    };
}    // namespace pika::detail

namespace pika::execution::experimental::detail {
    template <typename... Ts>
    struct any_receiver_ref_base
    {
        void* receiver = nullptr;

        template <typename Receiver>
        explicit any_receiver_ref_base(Receiver* receiver)
          : receiver(static_cast<void*>(receiver))
        {
        }
        any_receiver_ref_base(any_receiver_ref_base&&) noexcept = default;
        any_receiver_ref_base& operator=(any_receiver_ref_base&&) noexcept = default;
        any_receiver_ref_base(any_receiver_ref_base const&) = delete;
        any_receiver_ref_base& operator=(any_receiver_ref_base const&) = delete;

        virtual void set_value(Ts...) noexcept = 0;
        virtual void set_error(std::exception_ptr) noexcept = 0;
        virtual void set_stopped() noexcept = 0;
    };

    template <typename Receiver, typename... Ts>
    struct any_receiver_ref : any_receiver_ref_base<Ts...>
    {
        using any_receiver_ref_base<Ts...>::receiver;

        template <typename Receiver_>
        explicit any_receiver_ref(Receiver_* receiver)
          : any_receiver_ref_base<Ts...>(receiver)
        {
        }
        any_receiver_ref(any_receiver_ref&&) noexcept = default;
        any_receiver_ref& operator=(any_receiver_ref&&) noexcept = default;
        any_receiver_ref(any_receiver_ref const&) = delete;
        any_receiver_ref& operator=(any_receiver_ref const&) = delete;

        void set_value(Ts... ts) noexcept override
        {
            pika::execution::experimental::set_value(
                std::move(*static_cast<std::decay_t<Receiver>*>(receiver)), std::move(ts)...);
        }

        void set_error(std::exception_ptr ep) noexcept override
        {
            pika::execution::experimental::set_error(
                std::move(*static_cast<std::decay_t<Receiver>*>(receiver)), std::move(ep));
        }

        void set_stopped() noexcept override
        {
            pika::execution::experimental::set_stopped(
                std::move(*static_cast<std::decay_t<Receiver>*>(receiver)));
        }
    };

    template <typename... Ts>
    struct any_receiver
    {
        PIKA_STDEXEC_RECEIVER_CONCEPT

        any_receiver_ref_base<Ts...>* receiver;

        explicit any_receiver(any_receiver_ref_base<Ts...>* receiver)
          : receiver(receiver)
        {
        }
        any_receiver(any_receiver&&) noexcept = default;
        any_receiver& operator=(any_receiver&&) noexcept = default;
        any_receiver(any_receiver const&) = delete;
        any_receiver& operator=(any_receiver const&) = delete;

        template <typename... Ts_>
        auto set_value(
            Ts_&&... ts) && noexcept -> decltype(receiver->set_value(std::forward<Ts_>(ts)...))
        {
            try
            {
                receiver->set_value(std::forward<Ts_>(ts)...);
            }
            catch (...)
            {
                receiver->set_error(std::current_exception());
            }
        }

        friend void tag_invoke(pika::execution::experimental::set_error_t, any_receiver r,
            std::exception_ptr ep) noexcept
        {
            r.receiver->set_error(std::move(ep));
        }

        friend void tag_invoke(
            pika::execution::experimental::set_stopped_t, any_receiver r) noexcept
        {
            r.receiver->set_stopped();
        }

        friend constexpr pika::execution::experimental::empty_env tag_invoke(
            pika::execution::experimental::get_env_t, any_receiver const&) noexcept
        {
            return {};
        }
    };

    template <typename Sender, typename... Ts>
    struct any_operation_state_holder_impl final : any_operation_state_holder_base
    {
        [[no_unique_address]] std::optional<
            std::decay_t<connect_result_t<Sender, any_receiver<Ts...>>>> operation_state;

        template <typename Sender_>
        any_operation_state_holder_impl(Sender_&& sender, any_receiver<Ts...>&& receiver)
          : operation_state(pika::detail::with_result_of([&sender, &receiver]() mutable {
              return pika::execution::experimental::connect(
                  std::forward<Sender_>(sender), std::move(receiver));
          }))
        {
        }
        ~any_operation_state_holder_impl() noexcept override = default;

        void start() & noexcept override
        {
            PIKA_ASSERT(operation_state.has_value());
            // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
            pika::execution::experimental::start(*operation_state);
        }
    };

    class PIKA_EXPORT any_operation_state_holder
    {
        using base_type = detail::any_operation_state_holder_base;
        template <typename Sender, typename... Ts>
        using impl_type = detail::any_operation_state_holder_impl<Sender, Ts...>;
        using storage_type = pika::detail::movable_sbo_storage<base_type, 8 * sizeof(void*)>;

        storage_type storage{};

    public:
        template <typename Sender, typename... Ts>
        any_operation_state_holder(Sender&& sender, any_receiver<Ts...>&& receiver)
        {
            storage.template store<impl_type<Sender, Ts...>>(
                std::forward<Sender>(sender), std::move(receiver));
        }

        ~any_operation_state_holder() noexcept = default;
        any_operation_state_holder(any_operation_state_holder&&) = delete;
        any_operation_state_holder(any_operation_state_holder const&) = delete;
        any_operation_state_holder& operator=(any_operation_state_holder&&) = delete;
        any_operation_state_holder& operator=(any_operation_state_holder const&) = delete;

        void start() & noexcept;
    };

    template <typename Receiver, typename... Ts>
    class any_operation_state
    {
        std::decay_t<std::decay_t<Receiver>> receiver;
        any_receiver_ref<std::decay_t<Receiver>, Ts...> receiver_ref;
        any_operation_state_holder op_state;

    public:
        template <typename Sender, typename Receiver_>
        any_operation_state(Sender&& sender, Receiver_&& receiver)
          : receiver(std::forward<Receiver_>(receiver))
          , receiver_ref{&this->receiver}
          , op_state{std::forward<Sender>(sender).connect(any_receiver<Ts...>(&receiver_ref))}
        {
        }

        ~any_operation_state() noexcept = default;
        any_operation_state(any_operation_state&&) = delete;
        any_operation_state(any_operation_state const&) = delete;
        any_operation_state& operator=(any_operation_state&&) = delete;
        any_operation_state& operator=(any_operation_state const&) = delete;

        friend void tag_invoke(
            pika::execution::experimental::start_t, any_operation_state& os) noexcept
        {
            os.op_state.start();
        }
    };

    [[noreturn]] PIKA_EXPORT void throw_bad_any_call(
        char const* class_name, char const* function_name);
}    // namespace pika::execution::experimental::detail

namespace pika::execution::experimental::detail {
    template <typename... Ts>
    struct unique_any_sender_base
    {
        virtual ~unique_any_sender_base() noexcept = default;
        virtual void move_into(void* p) = 0;
        virtual any_operation_state_holder connect(any_receiver<Ts...>&& receiver) && = 0;
        virtual bool empty() const noexcept { return false; }
    };

    template <typename... Ts>
    struct any_sender_base : public unique_any_sender_base<Ts...>
    {
        virtual any_sender_base* clone() const = 0;
        virtual void clone_into(void* p) const = 0;
        using unique_any_sender_base<Ts...>::connect;
        virtual any_operation_state_holder connect(any_receiver<Ts...>&& receiver) const& = 0;
    };

    template <typename... Ts>
    struct empty_unique_any_sender final : unique_any_sender_base<Ts...>
    {
        void move_into(void*) override { PIKA_UNREACHABLE; }

        bool empty() const noexcept override { return true; }

        [[noreturn]] any_operation_state_holder connect(any_receiver<Ts...>&&) && override
        {
            throw_bad_any_call("unique_any_sender", "connect");
        }
    };

    template <typename... Ts>
    struct empty_any_sender final : any_sender_base<Ts...>
    {
        void move_into(void*) override { PIKA_UNREACHABLE; }

        any_sender_base<Ts...>* clone() const override { PIKA_UNREACHABLE; }

        void clone_into(void*) const override { PIKA_UNREACHABLE; }

        bool empty() const noexcept override { return true; }

        [[noreturn]] any_operation_state_holder connect(any_receiver<Ts...>&&) const& override
        {
            throw_bad_any_call("any_sender", "connect");
        }

        [[noreturn]] any_operation_state_holder connect(any_receiver<Ts...>&&) && override
        {
            throw_bad_any_call("any_sender", "connect");
        }
    };

    template <typename Sender, typename... Ts>
    struct unique_any_sender_impl final : unique_any_sender_base<Ts...>
    {
        std::decay_t<Sender> sender;

        template <typename Sender_,
            typename =
                std::enable_if_t<!std::is_same_v<std::decay_t<Sender_>, unique_any_sender_impl>>>
        explicit unique_any_sender_impl(Sender_&& sender)
          : sender(std::forward<Sender_>(sender))
        {
        }

        ~unique_any_sender_impl() noexcept = default;

        void move_into(void* p) override { new (p) unique_any_sender_impl(std::move(sender)); }

        any_operation_state_holder connect(any_receiver<Ts...>&& receiver) && override
        {
            return any_operation_state_holder{std::move(sender), std::move(receiver)};
        }
    };

    template <typename Sender, typename... Ts>
    struct any_sender_impl final : any_sender_base<Ts...>
    {
        std::decay_t<Sender> sender;

        template <typename Sender_,
            typename = std::enable_if_t<!std::is_same_v<std::decay_t<Sender_>, any_sender_impl>>>
        explicit any_sender_impl(Sender_&& sender)
          : sender(std::forward<Sender_>(sender))
        {
        }

        ~any_sender_impl() noexcept = default;

        void move_into(void* p) override { new (p) any_sender_impl(std::move(sender)); }

        any_sender_base<Ts...>* clone() const override { return new any_sender_impl(sender); }

        void clone_into(void* p) const override { new (p) any_sender_impl(sender); }

        any_operation_state_holder connect(any_receiver<Ts...>&& receiver) const& override
        {
            return any_operation_state_holder{sender, std::move(receiver)};
        }

        any_operation_state_holder connect(any_receiver<Ts...>&& receiver) && override
        {
            return any_operation_state_holder{std::move(sender), std::move(receiver)};
        }
    };
}    // namespace pika::execution::experimental::detail

namespace pika::execution::experimental {
#if !defined(PIKA_HAVE_CXX20_TRIVIAL_VIRTUAL_DESTRUCTOR)
    namespace detail {
        // This helper only exists to make it possible to use (unique_)any_sender in global
        // variables or in general static that may be created before main. When used as a base for
        // (unique_)any_sender, this ensures that the empty vtables for any_operation_state are
        // created as the first thing when creating an (unique_)any_sender. The empty vtables for
        // any_operation_state may otherwise be created much later (when the sender is connected and
        // started), and thus destroyed before the (unique_)any_sender is destroyed. This would be
        // problematic since the (unique_)any_sender can hold previously created
        // any_operation_states indirectly.
        template <typename... Ts>
        struct any_sender_static_empty_vtable_helper
        {
            any_sender_static_empty_vtable_helper()
            {
                pika::detail::get_empty_vtable<any_operation_state_holder_base>();
            }
        };
    }    // namespace detail
#endif

    template <typename... Ts>
    class any_sender;

    template <typename... Ts>
    class unique_any_sender
#if !defined(PIKA_HAVE_CXX20_TRIVIAL_VIRTUAL_DESTRUCTOR)
      : private detail::any_sender_static_empty_vtable_helper<Ts...>
#endif
    {
        static_assert(pika::util::detail::none_of_v<std::is_reference<Ts>...>,
            "unique_any_sender does not handle references as completion signatures");
        using base_type = detail::unique_any_sender_base<Ts...>;
        template <typename Sender>
        using impl_type = detail::unique_any_sender_impl<Sender, Ts...>;
        using storage_type = pika::detail::movable_sbo_storage<base_type, 4 * sizeof(void*)>;

        storage_type storage{};

    public:
        PIKA_STDEXEC_SENDER_CONCEPT
        unique_any_sender() = default;

        template <typename Sender,
            typename = std::enable_if_t<!std::is_same_v<std::decay_t<Sender>, unique_any_sender>>>
        unique_any_sender(Sender&& sender)
        {
            storage.template store<impl_type<Sender>>(std::forward<Sender>(sender));
        }

        template <typename Sender,
            typename = std::enable_if_t<!std::is_same_v<std::decay_t<Sender>, unique_any_sender>>>
        unique_any_sender& operator=(Sender&& sender)
        {
            storage.template store<impl_type<Sender>>(std::forward<Sender>(sender));
            return *this;
        }

        ~unique_any_sender() noexcept = default;
        unique_any_sender(unique_any_sender&&) = default;
        unique_any_sender(unique_any_sender const&) = delete;
        unique_any_sender& operator=(unique_any_sender&&) = default;
        unique_any_sender& operator=(unique_any_sender const&) = delete;

        // cppcheck-suppress noExplicitConstructor
        unique_any_sender(any_sender<Ts...>&& other)
          : storage(std::move(other.storage))
        {
            other.reset();
        }

        unique_any_sender& operator=(any_sender<Ts...>&& other)
        {
            storage = std::move(other.storage);
            other.reset();
            return *this;
        };

        template <template <typename...> class Tuple, template <typename...> class Variant>
        using value_types = Variant<Tuple<Ts...>>;

        template <template <typename...> class Variant>
        using error_types = Variant<std::exception_ptr>;

        static constexpr bool sends_done = false;

        using completion_signatures = pika::execution::experimental::completion_signatures<
            pika::execution::experimental::set_value_t(Ts...),
            pika::execution::experimental::set_error_t(std::exception_ptr),
            pika::execution::experimental::set_stopped_t()>;

        template <typename Receiver>
        friend detail::any_operation_state<Receiver, Ts...> tag_invoke(
            pika::execution::experimental::connect_t, unique_any_sender&& s, Receiver&& receiver)
        {
            // We first move the storage to a temporary variable so that this
            // any_sender is empty after this connect. Doing
            // std::move(storage.get()).connect(...) would leave us with a
            // non-empty any_sender holding a moved-from sender.
            auto moved_storage = std::move(s.storage);
            return {std::move(moved_storage.get()), std::forward<Receiver>(receiver)};
        }

        template <typename Receiver>
        friend detail::any_operation_state<Receiver, Ts...>
        tag_invoke(pika::execution::experimental::connect_t, unique_any_sender const&, Receiver&&)
        {
            static_assert(sizeof(Receiver) == 0,
                "Are you missing a std::move? unique_any_sender is not copyable and thus not "
                "l-value connectable. Make sure you are passing a non-const r-value reference of "
                "the sender.");
            PIKA_UNREACHABLE;
        }

        template <typename Sender>
        void reset(Sender&& sender)
        {
            if constexpr (std::is_same_v<std::decay_t<Sender>, unique_any_sender>)
            {
                *this = std::forward<Sender>(sender);
            }
            else { storage.template store<impl_type<Sender>>(std::forward<Sender>(sender)); }
        }

        void reset() { storage.reset(); }

        bool empty() const noexcept { return storage.empty(); }

        explicit operator bool() const noexcept { return !empty(); }
    };

    template <typename... Ts>
    class any_sender
#if !defined(PIKA_HAVE_CXX20_TRIVIAL_VIRTUAL_DESTRUCTOR)
      : private detail::any_sender_static_empty_vtable_helper<Ts...>
#endif
    {
        static_assert(pika::util::detail::none_of_v<std::is_reference<Ts>...>,
            "any_sender does not handle references as completion signatures");
        using base_type = detail::any_sender_base<Ts...>;
        template <typename Sender>
        using impl_type = detail::any_sender_impl<Sender, Ts...>;
        using storage_type = pika::detail::copyable_sbo_storage<base_type, 4 * sizeof(void*)>;

        storage_type storage{};

        friend unique_any_sender<Ts...>;

    public:
        PIKA_STDEXEC_SENDER_CONCEPT
        any_sender() = default;

        template <typename Sender,
            typename = std::enable_if_t<!std::is_same_v<std::decay_t<Sender>, any_sender>>>
        any_sender(Sender&& sender)
        {
            static_assert(std::is_copy_constructible_v<std::decay_t<Sender>>,
                "any_sender requires the given sender to be copy constructible. Ensure the used "
                "sender type is copy constructible or use unique_any_sender if you do not require "
                "copyability.");
            storage.template store<impl_type<Sender>>(std::forward<Sender>(sender));
        }

        template <typename Sender,
            typename = std::enable_if_t<!std::is_same_v<std::decay_t<Sender>, any_sender>>>
        any_sender& operator=(Sender&& sender)
        {
            static_assert(std::is_copy_constructible_v<std::decay_t<Sender>>,
                "any_sender requires the given sender to be copy constructible. Ensure the used "
                "sender type is copy constructible or use unique_any_sender if you do not require "
                "copyability.");
            storage.template store<impl_type<Sender>>(std::forward<Sender>(sender));
            return *this;
        }

        ~any_sender() noexcept = default;
        any_sender(any_sender&&) = default;
        any_sender(any_sender const&) = default;
        any_sender& operator=(any_sender&&) = default;
        any_sender& operator=(any_sender const&) = default;

        template <template <typename...> class Tuple, template <typename...> class Variant>
        using value_types = Variant<Tuple<Ts...>>;

        template <template <typename...> class Variant>
        using error_types = Variant<std::exception_ptr>;

        static constexpr bool sends_done = false;

        using completion_signatures = pika::execution::experimental::completion_signatures<
            pika::execution::experimental::set_value_t(Ts...),
            pika::execution::experimental::set_error_t(std::exception_ptr),
            pika::execution::experimental::set_stopped_t()>;

        template <typename Receiver>
        friend detail::any_operation_state<Receiver, Ts...> tag_invoke(
            pika::execution::experimental::connect_t, any_sender const& s, Receiver&& receiver)
        {
            return {s.storage.get(), std::forward<Receiver>(receiver)};
        }

        template <typename Receiver>
        friend detail::any_operation_state<Receiver, Ts...>
        tag_invoke(pika::execution::experimental::connect_t, any_sender&& s, Receiver&& receiver)
        {
            // We first move the storage to a temporary variable so that this
            // any_sender is empty after this connect. Doing
            // std::move(storage.get()).connect(...) would leave us with a
            // non-empty any_sender holding a moved-from sender.
            auto moved_storage = std::move(s.storage);
            return {std::move(moved_storage.get()), std::forward<Receiver>(receiver)};
        }

        template <typename Sender>
        void reset(Sender&& sender)
        {
            if constexpr (std::is_same_v<std::decay_t<Sender>, any_sender>)
            {
                *this = std::forward<Sender>(sender);
            }
            else
            {
                static_assert(std::is_copy_constructible_v<std::decay_t<Sender>>,
                    "any_sender requires the given sender to be copy constructible. Ensure the "
                    "used sender type is copy constructible or use unique_any_sender if you do not "
                    "require copyability.");
                storage.template store<impl_type<Sender>>(std::forward<Sender>(sender));
            }
        }

        void reset() { storage.reset(); }

        bool empty() const noexcept { return storage.empty(); }

        explicit operator bool() const noexcept { return !empty(); }
    };

    namespace detail {
        template <template <typename...> class AnySender, typename Sender>
        auto make_any_sender_impl(Sender&& sender)
        {
#if defined(PIKA_HAVE_STDEXEC)
            using value_types_pack = pika::execution::experimental::value_types_of_t<Sender,
                pika::execution::experimental::empty_env, pika::util::detail::pack,
                pika::util::detail::pack>;
#else
            using value_types_pack =
                typename pika::execution::experimental::sender_traits<std::decay_t<Sender>>::
                    template value_types<pika::util::detail::pack, pika::util::detail::pack>;
#endif
            static_assert(value_types_pack::size == 1,
                "any_sender and unique_any_sender require the predecessor sender to send exactly "
                "one variant");
            using single_value_type_variant =
                typename pika::util::detail::at_index_impl<0, value_types_pack>::type;
            using any_sender_type =
                pika::util::detail::change_pack_t<AnySender, single_value_type_variant>;

            return any_sender_type(std::forward<Sender>(sender));
        }
    }    // namespace detail

    template <typename Sender, typename = std::enable_if_t<is_sender_v<Sender>>>
    auto make_unique_any_sender(Sender&& sender)
    {
        return detail::make_any_sender_impl<unique_any_sender>(std::forward<Sender>(sender));
    }

    template <typename Sender, typename = std::enable_if_t<is_sender_v<Sender>>>
    auto make_any_sender(Sender&& sender)
    {
        return detail::make_any_sender_impl<any_sender>(std::forward<Sender>(sender));
    }
}    // namespace pika::execution::experimental

namespace pika::detail {
    template <typename... Ts>
    struct empty_vtable_type<pika::execution::experimental::detail::unique_any_sender_base<Ts...>>
    {
        using type = pika::execution::experimental::detail::empty_unique_any_sender<Ts...>;
    };

    template <typename... Ts>
    struct empty_vtable_type<pika::execution::experimental::detail::any_sender_base<Ts...>>
    {
        using type = pika::execution::experimental::detail::empty_any_sender<Ts...>;
    };
}    // namespace pika::detail

#if defined(__GNUC__)
# pragma GCC diagnostic pop
#endif

#include <pika/config/warnings_suffix.hpp>
