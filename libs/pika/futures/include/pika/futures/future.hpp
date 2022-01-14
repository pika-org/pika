//  Copyright (c) 2007-2021 Hartmut Kaiser
//  Copyright (c) 2013 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>
#include <pika/allocator_support/allocator_deleter.hpp>
#include <pika/allocator_support/internal_allocator.hpp>
#include <pika/assert.hpp>
#include <pika/async_base/launch_policy.hpp>
#include <pika/concepts/concepts.hpp>
#include <pika/errors/try_catch_exception_ptr.hpp>
#include <pika/execution_base/operation_state.hpp>
#include <pika/execution_base/receiver.hpp>
#include <pika/execution_base/sender.hpp>
#include <pika/functional/detail/invoke.hpp>
#include <pika/functional/traits/is_invocable.hpp>
#include <pika/futures/detail/future_data.hpp>
#include <pika/futures/future_fwd.hpp>
#include <pika/futures/traits/acquire_shared_state.hpp>
#include <pika/futures/traits/detail/future_await_traits.hpp>
#include <pika/futures/traits/future_access.hpp>
#include <pika/futures/traits/future_then_result.hpp>
#include <pika/futures/traits/future_traits.hpp>
#include <pika/futures/traits/is_future.hpp>
#include <pika/modules/errors.hpp>
#include <pika/modules/memory.hpp>
#include <pika/serialization/detail/constructor_selector.hpp>
#include <pika/serialization/detail/non_default_constructible.hpp>
#include <pika/serialization/detail/polymorphic_nonintrusive_factory.hpp>
#include <pika/serialization/exception_ptr.hpp>
#include <pika/serialization/serialization_fwd.hpp>
#include <pika/timing/steady_clock.hpp>
#include <pika/type_support/decay.hpp>

#include <exception>
#include <iterator>
#include <memory>
#include <type_traits>
#include <utility>

namespace pika { namespace lcos { namespace detail {
    ///////////////////////////////////////////////////////////////////////////
    enum class future_state
    {
        invalid = 0,
        has_value = 1,
        has_exception = 2
    };

    template <typename Archive, typename Future>
    std::enable_if_t<!std::is_void_v<pika::traits::future_traits_t<Future>>>
    serialize_future_load(Archive& ar, Future& f)
    {
        using value_type = pika::traits::future_traits_t<Future>;
        using shared_state = pika::lcos::detail::future_data<value_type>;
        using init_no_addref = typename shared_state::init_no_addref;

        future_state state = future_state::invalid;
        ar >> state;
        // NOLINTNEXTLINE(bugprone-branch-clone)
        if (state == future_state::has_value)
        {
            if constexpr (std::is_default_constructible_v<value_type>)
            {
                value_type value;
                ar >> value;

                pika::intrusive_ptr<shared_state> p(
                    new shared_state(
                        init_no_addref{}, in_place{}, PIKA_MOVE(value)),
                    false);

                f = pika::traits::future_access<Future>::create(PIKA_MOVE(p));
            }
            else
            {
                value_type&& value =
                    serialization::detail::constructor_selector<
                        value_type>::create(ar);

                pika::intrusive_ptr<shared_state> p(
                    new shared_state(
                        init_no_addref{}, in_place{}, PIKA_MOVE(value)),
                    false);

                f = pika::traits::future_access<Future>::create(PIKA_MOVE(p));
            }
        }
        // NOLINTNEXTLINE(bugprone-branch-clone)
        else if (state == future_state::has_exception)
        {
            std::exception_ptr exception;
            ar >> exception;

            pika::intrusive_ptr<shared_state> p(
                new shared_state(init_no_addref{}, PIKA_MOVE(exception)), false);

            f = pika::traits::future_access<Future>::create(PIKA_MOVE(p));
        }
        // NOLINTNEXTLINE(bugprone-branch-clone)
        else if (state == future_state::invalid)
        {
            f = Future();
        }
        else
        {
            PIKA_THROW_EXCEPTION(invalid_status, "serialize_future_load",
                "attempting to deserialize a future with an unknown state");
        }
    }

    template <typename Archive, typename Future>
    std::enable_if_t<std::is_void_v<pika::traits::future_traits_t<Future>>>
    serialize_future_load(Archive& ar, Future& f)    //-V659
    {
        using shared_state = pika::lcos::detail::future_data<void>;
        using init_no_addref = typename shared_state::init_no_addref;

        future_state state = future_state::invalid;
        ar >> state;
        if (state == future_state::has_value)
        {
            pika::intrusive_ptr<shared_state> p(
                new shared_state(
                    init_no_addref{}, in_place{}, pika::util::unused),
                false);

            f = pika::traits::future_access<Future>::create(PIKA_MOVE(p));
        }
        else if (state == future_state::has_exception)
        {
            std::exception_ptr exception;
            ar >> exception;

            pika::intrusive_ptr<shared_state> p(
                new shared_state(init_no_addref{}, PIKA_MOVE(exception)), false);

            f = pika::traits::future_access<Future>::create(PIKA_MOVE(p));
        }
        else if (state == future_state::invalid)
        {
            f = Future();
        }
        else
        {
            PIKA_THROW_EXCEPTION(invalid_status, "serialize_future_load",
                "attempting to deserialize a future with an unknown state");
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    PIKA_EXPORT void preprocess_future(serialization::output_archive& ar,
        pika::lcos::detail::future_data_refcnt_base& state);

    template <typename Archive, typename Future>
    std::enable_if_t<!std::is_void_v<pika::traits::future_traits_t<Future>>>
    serialize_future_save(Archive& ar, Future const& f)
    {
        using value_type =
            typename pika::traits::future_traits<Future>::result_type;

        future_state state = future_state::invalid;
        if (f.valid() && !f.is_ready())
        {
            if (ar.is_preprocessing())
            {
                pika::traits::detail::shared_state_ptr_for_t<Future> state =
                    pika::traits::future_access<Future>::get_shared_state(f);

                state->execute_deferred();

                preprocess_future(ar, *state);
            }
            else
            {
                PIKA_THROW_EXCEPTION(invalid_status, "serialize_future_save",
                    "future must be ready in order for it to be serialized");
            }
            return;
        }

        if (f.has_value())
        {
            state = future_state::has_value;
            value_type const& value =
                *pika::traits::future_access<Future>::get_shared_state(f)
                     ->get_result();

            if constexpr (!std::is_default_constructible_v<value_type>)
            {
                using serialization::detail::save_construct_data;
                save_construct_data(ar, &value, 0);
            }
            ar << state << value;
        }
        else if (f.has_exception())
        {
            state = future_state::has_exception;
            std::exception_ptr exception = f.get_exception_ptr();
            ar << state << exception;
        }
        else
        {
            state = future_state::invalid;
            ar << state;
        }
    }

    template <typename Archive, typename Future>
    std::enable_if_t<std::is_void_v<pika::traits::future_traits_t<Future>>>
    serialize_future_save(Archive& ar, Future const& f)    //-V659
    {
        future_state state = future_state::invalid;
        if (f.valid() && !f.is_ready())
        {
            if (ar.is_preprocessing())
            {
                pika::traits::detail::shared_state_ptr_for_t<Future> state =
                    pika::traits::future_access<Future>::get_shared_state(f);

                state->execute_deferred();

                preprocess_future(ar, *state);
            }
            else
            {
                PIKA_THROW_EXCEPTION(invalid_status, "serialize_future_save",
                    "future must be ready in order for it to be serialized");
            }
            return;
        }

        if (f.has_value())
        {
            state = future_state::has_value;
            ar << state;
        }
        else if (f.has_exception())
        {
            state = future_state::has_exception;
            std::exception_ptr exception = f.get_exception_ptr();
            ar << state << exception;
        }
        else
        {
            state = future_state::invalid;
            ar << state;
        }
    }

    template <typename Future>
    void serialize_future(serialization::input_archive& ar, Future& f, unsigned)
    {
        serialize_future_load(ar, f);
    }

    template <typename Future>
    void serialize_future(
        serialization::output_archive& ar, Future& f, unsigned)
    {
        serialize_future_save(ar, f);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Future, typename Enable = void>
    struct future_unwrap_result;

    template <template <typename> class Future, typename R>
    struct future_unwrap_result<Future<Future<R>>>
    {
        using type = R;
        using wrapped_type = Future<type>;
    };

    template <typename R>
    struct future_unwrap_result<pika::future<pika::shared_future<R>>>
    {
        using type = R;
        using wrapped_type = pika::future<type>;
    };

    template <typename Future>
    using future_unwrap_result_t = typename future_unwrap_result<Future>::type;

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    struct future_value : future_data_result<T>
    {
        template <typename U>
        PIKA_FORCEINLINE static U get(U&& u)
        {
            return PIKA_FORWARD(U, u);
        }

        static T get_default()
        {
            return T();
        }
    };

    template <typename T>
    struct future_value<T&> : future_data_result<T&>
    {
        PIKA_FORCEINLINE static T& get(T* u)
        {
            return *u;
        }

        static T& get_default()
        {
            static T default_;
            return default_;
        }
    };

    template <>
    struct future_value<void> : future_data_result<void>
    {
        PIKA_FORCEINLINE static void get(pika::util::unused_type) {}

        static void get_default() {}
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename R>
    struct future_get_result
    {
        template <typename SharedState>
        PIKA_FORCEINLINE static R* call(
            SharedState const& state, error_code& ec = throws)
        {
            return state->get_result(ec);
        }
    };

    template <>
    struct future_get_result<util::unused_type>
    {
        template <typename SharedState>
        PIKA_FORCEINLINE static util::unused_type* call(
            SharedState const& state, error_code& ec = throws)
        {
            return state->get_result_void(ec);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Future, typename F, typename ContResult>
    class continuation;

    template <typename ContResult>
    struct continuation_result
    {
        using type = ContResult;
    };

    template <typename ContResult>
    using continuation_result_t =
        typename continuation_result<ContResult>::type;

    template <typename ContResult>
    struct continuation_result<pika::future<ContResult>>
    {
        using type = ContResult;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename ContResult, typename Future, typename Policy, typename F>
    inline traits::detail::shared_state_ptr_t<continuation_result_t<ContResult>>
    make_continuation(Future const& future, Policy&& policy, F&& f);

    // create non-unwrapping continuations
    template <typename ContResult, typename Future, typename Executor,
        typename F>
    inline traits::detail::shared_state_ptr_t<ContResult>
    make_continuation_exec(Future const& future, Executor&& exec, F&& f);

    template <typename ContResult, typename Future, typename Executor,
        typename Policy, typename F>
    inline traits::detail::shared_state_ptr_t<ContResult>
    make_continuation_exec_policy(
        Future const& future, Executor&& exec, Policy&& policy, F&& f);

    template <typename ContResult, typename Allocator, typename Future,
        typename Policy, typename F>
    inline traits::detail::shared_state_ptr_t<continuation_result_t<ContResult>>
    make_continuation_alloc(
        Allocator const& a, Future const& future, Policy&& policy, F&& f);

    template <typename ContResult, typename Allocator, typename Future,
        typename Policy, typename F>
    inline traits::detail::shared_state_ptr_t<ContResult>
    make_continuation_alloc_nounwrap(
        Allocator const& a, Future const& future, Policy&& policy, F&& f);

    ///////////////////////////////////////////////////////////////////////////
    template <typename Future, typename FD, typename Enable = void>
    struct future_then_dispatch
    {
        template <typename F>
        PIKA_FORCEINLINE static decltype(auto) call(
            Future&& /* fut */, F&& /* f */)
        {
            // dummy impl to fail compilation if this function is called
            static_assert(sizeof(Future) == 0, "Cannot use the \
                    dummy implementation of future_then_dispatch, please use \
                    one of the template specialization.");
        }

        template <typename T0, typename F>
        PIKA_FORCEINLINE static decltype(auto) call(
            Future&& /* fut */, T0&& /* t */, F&& /* f */)
        {
            // dummy impl to fail compilation if this function is called
            static_assert(sizeof(Future) == 0, "Cannot use the \
                    dummy implementation of future_then_dispatch, please use \
                    one of the template specialization.");
        }

        template <typename Allocator, typename F>
        PIKA_FORCEINLINE static decltype(auto) call_alloc(
            Allocator const& /* alloc */, Future&& /* fut */, F&& /* f */)
        {
            // dummy impl to fail compilation if this function is called
            static_assert(sizeof(Future) == 0, "Cannot use the \
                    dummy implementation of future_then_dispatch::call_alloc, \
                    please use one of the template specialization.");
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Allocator, typename Future>
    traits::detail::shared_state_ptr_t<future_unwrap_result_t<Future>>
    unwrap_alloc(Allocator const& a, Future&& future, error_code& ec = throws);

    template <typename Future>
    pika::traits::detail::shared_state_ptr_t<future_unwrap_result_t<Future>>
    unwrap(Future&& future, error_code& ec = throws);

    ///////////////////////////////////////////////////////////////////////////
    template <typename Future>
    inline traits::detail::shared_state_ptr_t<void> downcast_to_void(
        Future& future, bool addref)
    {
        using shared_state_type = traits::detail::shared_state_ptr_t<void>;
        using element_type = typename shared_state_type::element_type;

        // same as static_pointer_cast, but with addref option
        return shared_state_type(
            static_cast<element_type*>(
                traits::detail::get_shared_state(future).get()),
            addref);
    }

    ///////////////////////////////////////////////////////////////////////////
    // Operation state for sender compatibility
    template <typename Receiver, typename Future>
    class operation_state
    {
    private:
        using receiver_type = std::decay_t<Receiver>;
        using future_type = std::decay_t<Future>;
        using result_type = typename future_type::result_type;

    public:
        template <typename Receiver_>
        operation_state(Receiver_&& r, future_type f)
          : receiver_(PIKA_FORWARD(Receiver_, r))
          , future_(PIKA_MOVE(f))
        {
        }

        operation_state(operation_state&&) = delete;
        operation_state& operator=(operation_state&&) = delete;
        operation_state(operation_state const&) = delete;
        operation_state& operator=(operation_state const&) = delete;

        friend void tag_invoke(
            pika::execution::experimental::start_t, operation_state& os) noexcept
        {
            os.start_helper();
        }

    private:
        void start_helper() & noexcept
        {
            pika::detail::try_catch_exception_ptr(
                [&]() {
                    auto state = traits::detail::get_shared_state(future_);

                    if (!state)
                    {
                        PIKA_THROW_EXCEPTION(no_state, "operation_state::start",
                            "the future has no valid shared state");
                    }

                    // The operation state has to be kept alive until set_value is
                    // called, which means that we don't need to move receiver and
                    // future into the on_completed callback.
                    state->set_on_completed([this]() mutable {
                        if (future_.has_value())
                        {
                            if constexpr (std::is_void_v<result_type>)
                            {
                                pika::execution::experimental::set_value(
                                    PIKA_MOVE(receiver_));
                            }
                            else
                            {
                                pika::execution::experimental::set_value(
                                    PIKA_MOVE(receiver_), future_.get());
                            }
                        }
                        else if (future_.has_exception())
                        {
                            pika::execution::experimental::set_error(
                                PIKA_MOVE(receiver_),
                                future_.get_exception_ptr());
                        }
                    });
                },
                [&](std::exception_ptr ep) {
                    pika::execution::experimental::set_error(
                        PIKA_MOVE(receiver_), PIKA_MOVE(ep));
                });
        }

        std::decay_t<Receiver> receiver_;
        future_type future_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Derived, typename R>
    class future_base
    {
    public:
        using result_type = R;
        using shared_state_type =
            future_data_base<traits::detail::shared_state_ptr_result_t<R>>;

        // Sender compatibility
        template <template <typename...> class Tuple,
            template <typename...> class Variant>
        using value_types = std::conditional_t<std::is_void_v<result_type>,
            Variant<Tuple<>>, Variant<Tuple<result_type>>>;

        template <template <typename...> class Variant>
        using error_types = Variant<std::exception_ptr>;

        static constexpr bool sends_done = false;

    private:
        template <typename F>
        struct future_then_dispatch
          : lcos::detail::future_then_dispatch<Derived, F>
        {
        };

    public:
        future_base() noexcept = default;

        explicit future_base(pika::intrusive_ptr<shared_state_type> const& p)
          : shared_state_(p)
        {
        }

        explicit future_base(pika::intrusive_ptr<shared_state_type>&& p)
          : shared_state_(PIKA_MOVE(p))
        {
        }

        future_base(future_base const& other) = default;
        future_base(future_base&& other) noexcept = default;

        void swap(future_base& other) noexcept
        {
            shared_state_.swap(other.shared_state_);
        }

        future_base& operator=(future_base const& other) = default;
        future_base& operator=(future_base&& other) noexcept = default;

        // Returns: true only if *this refers to a shared state.
        constexpr bool valid() const noexcept
        {
            return shared_state_ != nullptr;
        }

        // Returns: true if the shared state is ready, false if it isn't.
        bool is_ready() const noexcept
        {
            return shared_state_ != nullptr && shared_state_->is_ready();
        }

        // Returns: true if the shared state is ready and stores a value,
        //          false if it isn't.
        bool has_value() const noexcept
        {
            return shared_state_ != nullptr && shared_state_->has_value();
        }

        // Returns: true if the shared state is ready and stores an exception,
        //          false if it isn't.
        bool has_exception() const noexcept
        {
            return shared_state_ != nullptr && shared_state_->has_exception();
        }

        // Effects:
        //   - Blocks until the future is ready.
        // Returns: The stored exception_ptr if has_exception(), a null
        //          pointer otherwise.
        std::exception_ptr get_exception_ptr() const
        {
            if (!shared_state_)
            {
                PIKA_THROW_EXCEPTION(no_state,
                    "future_base<R>::get_exception_ptr",
                    "this future has no valid shared state");
            }

            using result_type = typename shared_state_type::result_type;

            error_code ec(lightweight);
            lcos::detail::future_get_result<result_type>::call(
                this->shared_state_, ec);
            if (!ec)
            {
                PIKA_ASSERT(!has_exception());
                return std::exception_ptr();
            }
            return pika::detail::access_exception(ec);
        }

        // Notes: The three functions differ only by input parameters.
        //   - The first only takes a callable object which accepts a future
        //     object as a parameter.
        //   - The second function takes an executor as the first parameter
        //     and a callable object as the second parameter.
        //   - The third function takes a launch policy as the first parameter
        //     and a callable object as the second parameter.
        //   In cases where 'decltype(func(*this))' is future<R>, the
        //   resulting type is future<R> instead of future<future<R>>.
        // Effects:
        //   - The continuation is called when the object's shared state is
        //     ready (has a value or exception stored).
        //   - The continuation launches according to the specified launch
        //     policy or executor.
        //   - When the executor or launch policy is not provided the
        //     continuation inherits the parent's launch policy or executor.
        //   - If the parent was created with std::promise or with a
        //     packaged_task (has no associated launch policy), the
        //     continuation behaves the same as the third overload with a
        //     policy argument of launch::async | launch::deferred and the
        //     same argument for func.
        //   - If the parent has a policy of launch::deferred and the
        //     continuation does not have a specified launch policy or
        //     scheduler, then the parent is filled by immediately calling
        //     .wait(), and the policy of the antecedent is launch::deferred
        // Returns: An object of type future<decltype(func(*this))> that
        //          refers to the shared state created by the continuation.
        // Postcondition:
        //   - The future object is moved to the parameter of the continuation
        //     function.
        //   - valid() == false on original future object immediately after it
        //     returns.
        template <typename F>
        static auto then(Derived&& fut, F&& f, error_code& ec = throws)
            -> decltype(future_then_dispatch<std::decay_t<F>>::call(
                PIKA_MOVE(fut), PIKA_FORWARD(F, f)))
        {
            using result_type =
                decltype(future_then_dispatch<std::decay_t<F>>::call(
                    PIKA_MOVE(fut), PIKA_FORWARD(F, f)));

            if (!fut.shared_state_)
            {
                PIKA_THROWS_IF(ec, no_state, "future_base<R>::then",
                    "this future has no valid shared state");
                return result_type();
            }

            return future_then_dispatch<std::decay_t<F>>::call(
                PIKA_MOVE(fut), PIKA_FORWARD(F, f));
        }

        template <typename F, typename T0>
        static auto then(Derived&& fut, T0&& t0, F&& f, error_code& ec = throws)
            -> decltype(future_then_dispatch<std::decay_t<T0>>::call(
                PIKA_MOVE(fut), PIKA_FORWARD(T0, t0), PIKA_FORWARD(F, f)))
        {
            using result_type =
                decltype(future_then_dispatch<std::decay_t<T0>>::call(
                    PIKA_MOVE(fut), PIKA_FORWARD(T0, t0), PIKA_FORWARD(F, f)));

            if (!fut.shared_state_)
            {
                PIKA_THROWS_IF(ec, no_state, "future_base<R>::then",
                    "this future has no valid shared state");
                return result_type();
            }

            return future_then_dispatch<std::decay_t<T0>>::call(
                PIKA_MOVE(fut), PIKA_FORWARD(T0, t0), PIKA_FORWARD(F, f));
        }

        template <typename Allocator, typename F>
        static auto then_alloc(Allocator const& alloc, Derived&& fut, F&& f,
            error_code& ec = throws)
            -> decltype(future_then_dispatch<std::decay_t<F>>::call_alloc(
                alloc, PIKA_MOVE(fut), PIKA_FORWARD(F, f)))
        {
            using result_type =
                decltype(future_then_dispatch<std::decay_t<F>>::call_alloc(
                    alloc, PIKA_MOVE(fut), PIKA_FORWARD(F, f)));

            if (!fut.shared_state_)
            {
                PIKA_THROWS_IF(ec, no_state, "future_base<R>::then_alloc",
                    "this future has no valid shared state");
                return result_type();
            }

            return future_then_dispatch<std::decay_t<F>>::call_alloc(
                alloc, PIKA_MOVE(fut), PIKA_FORWARD(F, f));
        }

        // Effects: blocks until the shared state is ready.
        void wait(error_code& ec = throws) const
        {
            if (!shared_state_)
            {
                PIKA_THROWS_IF(ec, no_state, "future_base<R>::wait",
                    "this future has no valid shared state");
                return;
            }
            shared_state_->wait(ec);
        }

        // Effects: none if the shared state contains a deferred function
        //          (30.6.8), otherwise blocks until the shared state is ready
        //          or until the absolute timeout (30.2.4) specified by
        //          abs_time has expired.
        // Returns:
        //   - future_status::deferred if the shared state contains a deferred
        //     function.
        //   - future_status::ready if the shared state is ready.
        //   - future_status::timeout if the function is returning because the
        //     absolute timeout (30.2.4) specified by abs_time has expired.
        // Throws: timeout-related exceptions (30.2.4).
        pika::future_status wait_until(
            pika::chrono::steady_time_point const& abs_time,
            error_code& ec = throws) const
        {
            if (!shared_state_)
            {
                PIKA_THROWS_IF(ec, no_state, "future_base<R>::wait_until",
                    "this future has no valid shared state");
                return pika::future_status::uninitialized;
            }
            return shared_state_->wait_until(abs_time.value(), ec);
        }

        // Effects: none if the shared state contains a deferred function
        //          (30.6.8), otherwise blocks until the shared state is ready
        //          or until the relative timeout (30.2.4) specified by
        //          rel_time has expired.
        // Returns:
        //   - future_status::deferred if the shared state contains a deferred
        //     function.
        //   - future_status::ready if the shared state is ready.
        //   - future_status::timeout if the function is returning because the
        //     relative timeout (30.2.4) specified by rel_time has expired.
        // Throws: timeout-related exceptions (30.2.4).
        pika::future_status wait_for(
            pika::chrono::steady_duration const& rel_time,
            error_code& ec = throws) const
        {
            return wait_until(rel_time.from_now(), ec);
        }

#if defined(PIKA_HAVE_CXX20_COROUTINES)
        bool await_ready() const noexcept
        {
            return lcos::detail::await_ready(
                *static_cast<Derived const*>(this));
        }

        template <typename Promise>
        void await_suspend(lcos::detail::coroutine_handle<Promise> rh)
        {
            lcos::detail::await_suspend(*static_cast<Derived*>(this), rh);
        }

        decltype(auto) await_resume()
        {
            return lcos::detail::await_resume(*static_cast<Derived*>(this));
        }
#endif

    protected:
        pika::intrusive_ptr<shared_state_type> shared_state_;
    };
}}}    // namespace pika::lcos::detail

namespace pika {

    ///////////////////////////////////////////////////////////////////////////
    template <typename R>
    class future : public lcos::detail::future_base<future<R>, R>
    {
    private:
        using base_type = lcos::detail::future_base<future<R>, R>;

    public:
        using result_type = R;
        using shared_state_type = typename base_type::shared_state_type;

        // Sender compatibility
        using base_type::error_types;
        using base_type::sends_done;
        using base_type::value_types;

        template <typename Receiver>
        friend lcos::detail::operation_state<Receiver, future> tag_invoke(
            pika::execution::experimental::connect_t, future&& f,
            Receiver&& receiver)
        {
            return {PIKA_FORWARD(Receiver, receiver), PIKA_MOVE(f)};
        }

    private:
        struct invalidate
        {
            constexpr explicit invalidate(future& f) noexcept
              : f_(f)
            {
            }

            ~invalidate()
            {
                f_.shared_state_.reset();
            }

            future& f_;
        };

    private:
        template <typename Future>
        friend struct pika::traits::future_access;

        template <typename Future, typename Enable>
        friend struct pika::traits::detail::future_access_customization_point;

        // Effects: constructs a future object from an shared state
        explicit future(pika::intrusive_ptr<shared_state_type> const& state)
          : base_type(state)
        {
        }

        explicit future(pika::intrusive_ptr<shared_state_type>&& state)
          : base_type(PIKA_MOVE(state))
        {
        }

        template <typename SharedState>
        explicit future(pika::intrusive_ptr<SharedState> const& state)
          : base_type(pika::static_pointer_cast<shared_state_type>(state))
        {
        }

    public:
        // Effects: constructs an empty future object that does not refer to
        //          an shared state.
        // Postcondition: valid() == false.
        constexpr future() noexcept = default;

        // Effects: move constructs a future object that refers to the shared
        //          state that was originally referred to by other (if any).
        // Postconditions:
        //   - valid() returns the same value as other.valid() prior to the
        //     constructor invocation.
        //   - other.valid() == false.
        future(future&& other) noexcept = default;

        // Effects: constructs a future object by moving the instance referred
        //          to by rhs and unwrapping the inner future.
        // Postconditions:
        //   - valid() returns the same value as other.valid() prior to the
        //     constructor invocation.
        //   - other.valid() == false.
        future(future<future>&& other) noexcept
          : base_type(
                other.valid() ? lcos::detail::unwrap(PIKA_MOVE(other)) : nullptr)
        {
        }

        // Effects: constructs a future object by moving the instance referred
        //          to by rhs and unwrapping the inner future.
        // Postconditions:
        //   - valid() returns the same value as other.valid() prior to the
        //     constructor invocation.
        //   - other.valid() == false.
        future(future<shared_future<R>>&& other) noexcept
          : base_type(
                other.valid() ? lcos::detail::unwrap(PIKA_MOVE(other)) : nullptr)
        {
        }

        // Effects: constructs a future<void> object that will be ready when
        //          the given future is ready
        // Postconditions:
        //   - valid() returns the same value as other.valid() prior to the
        //     constructor invocation.
        //   - other.valid() == false.
        template <typename T>
        future(future<T>&& other,
            std::enable_if_t<std::is_void_v<R> && !traits::is_future_v<T>, T>* =
                nullptr)
          : base_type(other.valid() ?
                    lcos::detail::downcast_to_void(other, false) :
                    nullptr)
        {
            traits::future_access<future<T>>::detach_shared_state(
                PIKA_MOVE(other));
        }

        // Effects:
        //   - releases any shared state (30.6.4);
        //   - destroys *this.
        ~future() = default;

        // Effects:
        //   - releases any shared state (30.6.4).
        //   - move assigns the contents of other to *this.
        // Postconditions:
        //   - valid() returns the same value as other.valid() prior to the
        //     assignment.
        //   - other.valid() == false.
        future& operator=(future&& other) noexcept = default;

        // Returns: shared_future<R>(PIKA_MOVE(*this)).
        // Postcondition: valid() == false.
        shared_future<R> share() noexcept
        {
            return shared_future<R>(PIKA_MOVE(*this));
        }

        // Effects: wait()s until the shared state is ready, then retrieves
        //          the value stored in the shared state.
        // Returns:
        //   - future::get() returns the value v stored in the object's
        //     shared state as PIKA_MOVE(v).
        //   - future<R&>::get() returns the reference stored as value in the
        //     object's shared state.
        //   - future<void>::get() returns nothing.
        // Throws: the stored exception, if an exception was stored in the
        //         shared state.
        // Postcondition: valid() == false.
        typename pika::traits::future_traits<future>::result_type get()
        {
            if (!this->shared_state_)
            {
                PIKA_THROW_EXCEPTION(no_state, "future<R>::get",
                    "this future has no valid shared state");
            }

            invalidate on_exit(*this);

            using result_type = typename shared_state_type::result_type;
            result_type* result =
                lcos::detail::future_get_result<result_type>::call(
                    this->shared_state_);

            // no error has been reported, return the result
            return lcos::detail::future_value<R>::get(PIKA_MOVE(*result));
        }

        typename pika::traits::future_traits<future>::result_type get(
            error_code& ec)
        {
            if (!this->shared_state_)
            {
                PIKA_THROWS_IF(ec, no_state, "future<R>::get",
                    "this future has no valid shared state");
                return lcos::detail::future_value<R>::get_default();
            }

            invalidate on_exit(*this);

            using result_type = typename shared_state_type::result_type;
            result_type* result =
                lcos::detail::future_get_result<result_type>::call(
                    this->shared_state_, ec);
            if (ec)
            {
                return lcos::detail::future_value<R>::get_default();
            }

            // no error has been reported, return the result
            return lcos::detail::future_value<R>::get(PIKA_MOVE(*result));
        }

        using base_type::get_exception_ptr;

        using base_type::has_exception;
        using base_type::has_value;
        using base_type::is_ready;
        using base_type::valid;

        template <typename F>
        decltype(auto) then(F&& f, error_code& ec = throws)
        {
#if defined(PIKA_COMPUTE_DEVICE_CODE)
            // This and the similar ifdefs below for future::then and
            // shared_future::then only work to satisfy nvcc up to at least
            // CUDA 11. Without this nvcc fails to compile some code with
            // "error: cannot use an entity undefined in device code" without
            // specifying what entity it refers to.
            PIKA_ASSERT(false);
            using future_type = decltype(
                base_type::then(PIKA_MOVE(*this), PIKA_FORWARD(F, f), ec));
            return future_type{};
#else
            invalidate on_exit(*this);
            return base_type::then(PIKA_MOVE(*this), PIKA_FORWARD(F, f), ec);
#endif
        }

        template <typename T0, typename F>
        decltype(auto) then(T0&& t0, F&& f, error_code& ec = throws)
        {
#if defined(PIKA_COMPUTE_DEVICE_CODE)
            PIKA_ASSERT(false);
            using future_type = decltype(base_type::then(
                PIKA_MOVE(*this), PIKA_FORWARD(T0, t0), PIKA_FORWARD(F, f), ec));
            return future_type{};
#else
            invalidate on_exit(*this);
            return base_type::then(
                PIKA_MOVE(*this), PIKA_FORWARD(T0, t0), PIKA_FORWARD(F, f), ec);
#endif
        }

        template <typename Allocator, typename F>
        auto then_alloc(Allocator const& alloc, F&& f, error_code& ec = throws)
            -> decltype(base_type::then_alloc(
#if defined(PIKA_CUDA_VERSION) && (PIKA_CUDA_VERSION < 1104)
                alloc, std::move(*this), std::forward<F>(f), ec))
#else
                alloc, PIKA_MOVE(*this), PIKA_FORWARD(F, f), ec))
#endif
        {
#if defined(PIKA_COMPUTE_DEVICE_CODE)
            PIKA_ASSERT(false);
            using future_type = decltype(base_type::then_alloc(
                alloc, std::move(*this), std::forward<F>(f), ec));
            return future_type{};
#else
            invalidate on_exit(*this);
            return base_type::then_alloc(
                alloc, PIKA_MOVE(*this), PIKA_FORWARD(F, f), ec);
#endif
        }

        using base_type::wait;
        using base_type::wait_for;
        using base_type::wait_until;
    };

    ///////////////////////////////////////////////////////////////////////////
    // Allow to convert any future<U> into any other future<R> based on an
    // existing conversion path U --> R.
    template <typename R, typename U>
    pika::future<R> make_future(pika::future<U>&& f)
    {
        static_assert(std::is_convertible_v<U, R> || std::is_void_v<R>,
            "the argument type must be implicitly convertible to the requested "
            "result type");

        if constexpr (std::is_convertible_v<pika::future<U>, pika::future<R>>)
        {
            return PIKA_MOVE(f);
        }
        else
        {
            return f.then(pika::launch::sync, [](pika::future<U>&& f) -> R {
                return util::void_guard<R>(), f.get();
            });
        }
    }

    // Allow to convert any future<U> into any other future<R> based on a given
    // conversion function: R conv(U).
    template <typename R, typename U, typename Conv>
    pika::future<R> make_future(pika::future<U>&& f, Conv&& conv)
    {
        if constexpr (std::is_convertible_v<pika::future<U>, pika::future<R>>)
        {
            return PIKA_MOVE(f);
        }
        else
        {
            return f.then(pika::launch::sync,
                [conv = PIKA_FORWARD(Conv, conv)](pika::future<U>&& f) -> R {
                    return PIKA_INVOKE(conv, f.get());
                });
        }
    }
}    // namespace pika

namespace pika {

    ///////////////////////////////////////////////////////////////////////////
    template <typename R>
    class shared_future : public lcos::detail::future_base<shared_future<R>, R>
    {
        using base_type = lcos::detail::future_base<shared_future<R>, R>;

    public:
        using result_type = R;
        using shared_state_type = typename base_type::shared_state_type;

        // Sender compatibility
        using base_type::error_types;
        using base_type::sends_done;
        using base_type::value_types;

        template <typename Receiver>
        friend lcos::detail::operation_state<Receiver, shared_future>
        tag_invoke(pika::execution::experimental::connect_t, shared_future&& f,
            Receiver&& receiver)
        {
            return {PIKA_FORWARD(Receiver, receiver), PIKA_MOVE(f)};
        }

        template <typename Receiver>
        friend lcos::detail::operation_state<Receiver, shared_future>
        tag_invoke(pika::execution::experimental::connect_t, shared_future& f,
            Receiver&& receiver)
        {
            return {PIKA_FORWARD(Receiver, receiver), f};
        }

    private:
        template <typename Future>
        friend struct pika::traits::future_access;

        template <typename Future, typename Enable>
        friend struct pika::traits::detail::future_access_customization_point;

        // Effects: constructs a future object from an shared state
        explicit shared_future(
            pika::intrusive_ptr<shared_state_type> const& state)
          : base_type(state)
        {
        }

        explicit shared_future(pika::intrusive_ptr<shared_state_type>&& state)
          : base_type(PIKA_MOVE(state))
        {
        }

        template <typename SharedState>
        explicit shared_future(pika::intrusive_ptr<SharedState> const& state)
          : base_type(pika::static_pointer_cast<shared_state_type>(state))
        {
        }

    public:
        // Effects: constructs an empty future object that does not refer to
        //          an shared state.
        // Postcondition: valid() == false.
        constexpr shared_future() noexcept = default;

        // Effects: constructs a shared_future object that refers to the same
        //          shared state as other (if any).
        // Postcondition: valid() returns the same value as other.valid().
        shared_future(shared_future const& other) = default;

        // Effects: move constructs a future object that refers to the shared
        //          state that was originally referred to by other (if any).
        // Postconditions:
        //   - valid() returns the same value as other.valid() prior to the
        //     constructor invocation.
        //   - other.valid() == false.
        shared_future(shared_future&& other) noexcept = default;

        shared_future(future<R>&& other) noexcept
          : base_type(pika::traits::detail::get_shared_state(other))
        {
            other = future<R>();
        }

        // Effects: constructs a shared_future object by moving the instance
        //          referred to by rhs and unwrapping the inner future.
        // Postconditions:
        //   - valid() returns the same value as other.valid() prior to the
        //     constructor invocation.
        //   - other.valid() == false.
        shared_future(future<shared_future>&& other) noexcept
          : base_type(
                other.valid() ? lcos::detail::unwrap(other.share()) : nullptr)
        {
        }

        // Effects: constructs a future<void> object that will be ready when
        //          the given future is ready
        // Postconditions:
        //   - valid() returns the same value as other.valid() prior to the
        //     constructor invocation.
        template <typename T>
        shared_future(shared_future<T> const& other,
            std::enable_if_t<std::is_void_v<R> && !traits::is_future_v<T>, T>* =
                nullptr)
          : base_type(other.valid() ?
                    lcos::detail::downcast_to_void(other, true) :
                    nullptr)
        {
        }

        // Effects:
        //   - releases any shared state (30.6.4);
        //   - destroys *this.
        ~shared_future() = default;

        // Effects:
        //   - releases any shared state (30.6.4).
        //   - assigns the contents of other to *this. As a result, *this
        //     refers to the same shared state as other (if any).
        // Postconditions:
        //   - valid() == other.valid().
        shared_future& operator=(shared_future const& other) = default;

        // Effects:
        //   - releases any shared state (30.6.4).
        //   - move assigns the contents of other to *this.
        // Postconditions:
        //   - valid() returns the same value as other.valid() prior to the
        //     assignment.
        //   - other.valid() == false.
        shared_future& operator=(shared_future&& other) noexcept = default;

        // Effects: wait()s until the shared state is ready, then retrieves
        //          the value stored in the shared state.
        // Returns:
        //   - shared_future::get() returns a const reference to the value
        //     stored in the object's shared state.
        //   - shared_future<R&>::get() returns the reference stored as value
        //     in the object's shared state.
        //   - shared_future<void>::get() returns nothing.
        // Throws: the stored exception, if an exception was stored in the
        //         shared state.
        // Postcondition: valid() == false.
        typename pika::traits::future_traits<shared_future>::result_type get()
            const    //-V659
        {
            if (!this->shared_state_)
            {
                PIKA_THROW_EXCEPTION(no_state, "shared_future<R>::get",
                    "this future has no valid shared state");
            }

            using result_type = typename shared_state_type::result_type;
            result_type* result =
                lcos::detail::future_get_result<result_type>::call(
                    this->shared_state_);

            // no error has been reported, return the result
            return lcos::detail::future_value<R>::get(*result);
        }

        typename pika::traits::future_traits<shared_future>::result_type get(
            error_code& ec) const    //-V659
        {
            using result_type = typename shared_state_type::result_type;
            if (!this->shared_state_)
            {
                PIKA_THROWS_IF(ec, no_state, "shared_future<R>::get",
                    "this future has no valid shared state");
                static result_type res(
                    lcos::detail::future_value<R>::get_default());
                return res;
            }

            result_type* result =
                lcos::detail::future_get_result<result_type>::call(
                    this->shared_state_, ec);
            if (ec)
            {
                static result_type res(
                    lcos::detail::future_value<R>::get_default());
                return res;
            }

            // no error has been reported, return the result
            return lcos::detail::future_value<R>::get(*result);
        }

        using base_type::get_exception_ptr;

        using base_type::has_exception;
        using base_type::has_value;
        using base_type::is_ready;
        using base_type::valid;

        template <typename F>
        decltype(auto) then(F&& f, error_code& ec = throws) const
        {
#if defined(PIKA_COMPUTE_DEVICE_CODE)
            PIKA_ASSERT(false);
            using future_type = decltype(
                base_type::then(shared_future(*this), PIKA_FORWARD(F, f), ec));
            return future_type{};
#else
            return base_type::then(shared_future(*this), PIKA_FORWARD(F, f), ec);
#endif
        }

        template <typename T0, typename F>
        decltype(auto) then(T0&& t0, F&& f, error_code& ec = throws) const
        {
#if defined(PIKA_COMPUTE_DEVICE_CODE)
            PIKA_ASSERT(false);
            using future_type = decltype(base_type::then(shared_future(*this),
                PIKA_FORWARD(T0, t0), PIKA_FORWARD(F, f), ec));
            return future_type{};
#else
            return base_type::then(shared_future(*this), PIKA_FORWARD(T0, t0),
                PIKA_FORWARD(F, f), ec);
#endif
        }

        template <typename Allocator, typename F>
        auto then_alloc(Allocator const& alloc, F&& f, error_code& ec = throws)
            -> decltype(base_type::then_alloc(
#if defined(PIKA_CUDA_VERSION) && (PIKA_CUDA_VERSION < 1104)
                alloc, std::move(*this), std::forward<F>(f), ec))
#else
                alloc, PIKA_MOVE(*this), PIKA_FORWARD(F, f), ec))
#endif
        {
#if defined(PIKA_COMPUTE_DEVICE_CODE)
            PIKA_ASSERT(false);
            using future_type = decltype(base_type::then_alloc(
                alloc, shared_future(*this), PIKA_FORWARD(F, f), ec));
            return future_type{};
#else
            return base_type::then_alloc(
                alloc, shared_future(*this), PIKA_FORWARD(F, f), ec);
#endif
        }

        using base_type::wait;
        using base_type::wait_for;
        using base_type::wait_until;
    };

    ///////////////////////////////////////////////////////////////////////////
    // Allow to convert any shared_future<U> into any other future<R> based on
    // an existing conversion path U --> R.
    template <typename R, typename U>
    pika::future<R> make_future(pika::shared_future<U> f)
    {
        static_assert(std::is_convertible_v<R, U> || std::is_void_v<R>,
            "the argument type must be implicitly convertible to the requested "
            "result type");

        if constexpr (std::is_convertible_v<pika::shared_future<U>,
                          pika::future<R>>)
        {
            return pika::future<R>(PIKA_MOVE(f));
        }
        else
        {
            return f.then(
                pika::launch::sync, [](pika::shared_future<U>&& f) -> R {
                    return util::void_guard<R>(), f.get();
                });
        }
    }

    // Allow to convert any future<U> into any other future<R> based on a given
    // conversion function: R conv(U).
    template <typename R, typename U, typename Conv>
    pika::future<R> make_future(pika::shared_future<U> f, Conv&& conv)
    {
        static_assert(pika::is_invocable_r_v<R, Conv, U>,
            "the argument type must be convertible to the requested "
            "result type by using the supplied conversion function");

        if constexpr (std::is_convertible_v<pika::shared_future<U>,
                          pika::future<R>>)
        {
            return pika::future<R>(PIKA_MOVE(f));
        }
        else
        {
            return f.then(pika::launch::sync,
                [conv = PIKA_FORWARD(Conv, conv)](pika::shared_future<U> const& f)
                    -> R { return PIKA_INVOKE(conv, f.get()); });
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // Convert any type of future<T> or shared_future<T> into a corresponding
    // shared_future<T>.
    template <typename R>
    pika::shared_future<R> make_shared_future(pika::future<R>&& f) noexcept
    {
        return f.share();
    }

    template <typename R>
    pika::shared_future<R>& make_shared_future(pika::shared_future<R>& f) noexcept
    {
        return f;
    }

    template <typename R>
    pika::shared_future<R>&& make_shared_future(
        pika::shared_future<R>&& f) noexcept
    {
        return PIKA_MOVE(f);
    }

    template <typename R>
    pika::shared_future<R> const& make_shared_future(
        pika::shared_future<R> const& f) noexcept
    {
        return f;
    }
}    // namespace pika

namespace pika {

    ///////////////////////////////////////////////////////////////////////////
    // Extension (see wg21.link/P0319), with allocator
    template <typename T, typename Allocator, typename... Ts>
    std::enable_if_t<std::is_constructible_v<T, Ts&&...> || std::is_void_v<T>,
        future<T>>
    make_ready_future_alloc(Allocator const& a, Ts&&... ts)
    {
        using result_type = T;

        using base_allocator = Allocator;
        using shared_state = traits::shared_state_allocator_t<
            lcos::detail::future_data<result_type>, base_allocator>;

        using other_allocator = typename std::allocator_traits<
            base_allocator>::template rebind_alloc<shared_state>;
        using traits = std::allocator_traits<other_allocator>;

        using init_no_addref = typename shared_state::init_no_addref;

        using unique_ptr = std::unique_ptr<shared_state,
            util::allocator_deleter<other_allocator>>;

        using lcos::detail::in_place;
        other_allocator alloc(a);
        unique_ptr p(traits::allocate(alloc, 1),
            util::allocator_deleter<other_allocator>{alloc});
        traits::construct(alloc, p.get(), init_no_addref{}, in_place{}, alloc,
            PIKA_FORWARD(Ts, ts)...);

        return pika::traits::future_access<future<result_type>>::create(
            p.release(), false);
    }

    // Extension (see wg21.link/P0319)
    template <typename T, typename... Ts>
    PIKA_FORCEINLINE std::enable_if_t<
        std::is_constructible_v<T, Ts&&...> || std::is_void_v<T>, future<T>>
    make_ready_future(Ts&&... ts)
    {
        return make_ready_future_alloc<T>(
            pika::util::internal_allocator<>{}, PIKA_FORWARD(Ts, ts)...);
    }
    ///////////////////////////////////////////////////////////////////////////
    // extension: create a pre-initialized future object, with allocator
    template <int DeductionGuard = 0, typename Allocator, typename T>
    future<pika::util::decay_unwrap_t<T>> make_ready_future_alloc(
        Allocator const& a, T&& init)
    {
        return pika::make_ready_future_alloc<pika::util::decay_unwrap_t<T>>(
            a, PIKA_FORWARD(T, init));
    }

    // extension: create a pre-initialized future object
    template <int DeductionGuard = 0, typename T>
    PIKA_FORCEINLINE future<pika::util::decay_unwrap_t<T>> make_ready_future(
        T&& init)
    {
        return pika::make_ready_future_alloc<pika::util::decay_unwrap_t<T>>(
            pika::util::internal_allocator<>{}, PIKA_FORWARD(T, init));
    }

    ///////////////////////////////////////////////////////////////////////////
    // extension: create a pre-initialized future object which holds the
    // given error
    template <typename T>
    future<T> make_exceptional_future(std::exception_ptr const& e)
    {
        using shared_state = lcos::detail::future_data<T>;
        using init_no_addref = typename shared_state::init_no_addref;

        pika::intrusive_ptr<shared_state> p(
            new shared_state(init_no_addref{}, e), false);

        return pika::traits::future_access<future<T>>::create(PIKA_MOVE(p));
    }

    template <typename T, typename E>
    future<T> make_exceptional_future(E e)
    {
        try
        {
            throw e;
        }
        catch (...)
        {
            return pika::make_exceptional_future<T>(std::current_exception());
        }

        return future<T>();
    }

    ///////////////////////////////////////////////////////////////////////////
    // extension: create a pre-initialized future object which gets ready at
    // a given point in time
    template <int DeductionGuard = 0, typename T>
    future<pika::util::decay_unwrap_t<T>> make_ready_future_at(
        pika::chrono::steady_time_point const& abs_time, T&& init)
    {
        using result_type = pika::util::decay_unwrap_t<T>;
        using shared_state = lcos::detail::timed_future_data<result_type>;

        pika::intrusive_ptr<shared_state> p(
            new shared_state(abs_time.value(), PIKA_FORWARD(T, init)));

        return pika::traits::future_access<future<result_type>>::create(
            PIKA_MOVE(p));
    }

    template <int DeductionGuard = 0, typename T>
    future<pika::util::decay_unwrap_t<T>> make_ready_future_after(
        pika::chrono::steady_duration const& rel_time, T&& init)
    {
        return pika::make_ready_future_at(
            rel_time.from_now(), PIKA_FORWARD(T, init));
    }

    ///////////////////////////////////////////////////////////////////////////
    // extension: create a pre-initialized future object, with allocator
    template <typename Allocator>
    inline future<void> make_ready_future_alloc(Allocator const& a)
    {
        return pika::make_ready_future_alloc<void>(a, util::unused);
    }

    // extension: create a pre-initialized future object
    PIKA_FORCEINLINE future<void> make_ready_future()
    {
        return make_ready_future_alloc<void>(
            pika::util::internal_allocator<>{}, util::unused);
    }

    // Extension (see wg21.link/P0319)
    template <typename T>
    PIKA_FORCEINLINE std::enable_if_t<std::is_void_v<T>, future<void>>
    make_ready_future()
    {
        return pika::make_ready_future();
    }

    // extension: create a pre-initialized future object which gets ready at
    // a given point in time
    inline future<void> make_ready_future_at(
        pika::chrono::steady_time_point const& abs_time)
    {
        using shared_state = lcos::detail::timed_future_data<void>;

        return pika::traits::future_access<future<void>>::create(
            new shared_state(abs_time.value(), pika::util::unused));
    }

    template <typename T>
    std::enable_if_t<std::is_void_v<T>, future<void>> make_ready_future_at(
        pika::chrono::steady_time_point const& abs_time)
    {
        return pika::make_ready_future_at(abs_time);
    }

    ///////////////////////////////////////////////////////////////////////////
    inline future<void> make_ready_future_after(
        pika::chrono::steady_duration const& rel_time)
    {
        return pika::make_ready_future_at(rel_time.from_now());
    }

    template <typename T>
    std::enable_if_t<std::is_void_v<T>, future<void>> make_ready_future_after(
        pika::chrono::steady_duration const& rel_time)
    {
        return pika::make_ready_future_at(rel_time.from_now());
    }
}    // namespace pika

namespace pika { namespace serialization {

    template <typename Archive, typename T>
    PIKA_FORCEINLINE void serialize(
        Archive& ar, ::pika::future<T>& f, unsigned version)
    {
        pika::lcos::detail::serialize_future(ar, f, version);
    }

    template <typename Archive, typename T>
    PIKA_FORCEINLINE void serialize(
        Archive& ar, ::pika::shared_future<T>& f, unsigned version)
    {
        pika::lcos::detail::serialize_future(ar, f, version);
    }
}}    // namespace pika::serialization

///////////////////////////////////////////////////////////////////////////////
// hoist deprecated names into old namespace
namespace pika { namespace lcos {

    template <typename R, typename U>
    PIKA_DEPRECATED_V(0, 1,
        "pika::lcos::make_future is deprecated. Use pika::make_future instead.")
    pika::future<R> make_future(pika::future<U>&& f)
    {
        return pika::make_future<R>(PIKA_MOVE(f));
    }

    template <typename R, typename U, typename Conv>
    PIKA_DEPRECATED_V(0, 1,
        "pika::lcos::make_future is deprecated. Use pika::make_future instead.")
    pika::future<R> make_future(pika::future<U>&& f, Conv&& conv)
    {
        return pika::make_future<R>(PIKA_MOVE(f), PIKA_FORWARD(Conv, conv));
    }

    template <typename R, typename U>
    PIKA_DEPRECATED_V(0, 1,
        "pika::lcos::make_future is deprecated. Use pika::make_future instead.")
    pika::future<R> make_future(pika::shared_future<U> f)
    {
        return pika::make_future<R>(PIKA_MOVE(f));
    }

    template <typename R, typename U, typename Conv>
    PIKA_DEPRECATED_V(0, 1,
        "pika::lcos::make_future is deprecated. Use pika::make_future instead.")
    pika::future<R> make_future(pika::shared_future<U> f, Conv&& conv)
    {
        return pika::make_future<R>(PIKA_MOVE(f), PIKA_FORWARD(Conv, conv));
    }

    template <typename T, typename Allocator, typename... Ts>
    PIKA_DEPRECATED_V(0, 1,
        "pika::lcos::make_ready_future_alloc is deprecated. Use "
        "pika::make_ready_future_alloc instead.")
    std::enable_if_t<std::is_constructible_v<T, Ts&&...> || std::is_void_v<T>,
        pika::future<T>> make_ready_future_alloc(Allocator const& a, Ts&&... ts)
    {
        return pika::make_ready_future_alloc<T>(a, PIKA_FORWARD(Ts, ts)...);
    }

    template <typename T, typename... Ts>
    PIKA_DEPRECATED_V(0, 1,
        "pika::lcos::make_ready_future is deprecated. Use "
        "pika::make_ready_future instead.")
    std::enable_if_t<std::is_constructible_v<T, Ts&&...> || std::is_void_v<T>,
        pika::future<T>> make_ready_future(Ts&&... ts)
    {
        return pika::make_ready_future_alloc<T>(
            pika::util::internal_allocator<>{}, PIKA_FORWARD(Ts, ts)...);
    }

    template <int DeductionGuard = 0, typename Allocator, typename T>
    PIKA_DEPRECATED_V(0, 1,
        "pika::lcos::make_ready_future_alloc is deprecated. Use "
        "pika::make_ready_future_alloc instead.")
    pika::future<pika::util::decay_unwrap_t<T>> make_ready_future_alloc(
        Allocator const& a, T&& init)
    {
        return pika::make_ready_future_alloc<pika::util::decay_unwrap_t<T>>(
            a, PIKA_FORWARD(T, init));
    }

    template <int DeductionGuard = 0, typename T>
    PIKA_DEPRECATED_V(0, 1,
        "pika::lcos::make_ready_future is deprecated. Use "
        "pika::make_ready_future instead.")
    pika::future<pika::util::decay_unwrap_t<T>> make_ready_future(T&& init)
    {
        return pika::make_ready_future_alloc<pika::util::decay_unwrap_t<T>>(
            pika::util::internal_allocator<>{}, PIKA_FORWARD(T, init));
    }

    template <typename T>
    PIKA_DEPRECATED_V(0, 1,
        "pika::lcos::make_exceptional_future is deprecated. Use "
        "pika::make_exceptional_future instead.")
    pika::future<T> make_exceptional_future(std::exception_ptr const& e)
    {
        return pika::make_exceptional_future<T>(e);
    }

    template <typename T, typename E>
    PIKA_DEPRECATED_V(0, 1,
        "pika::lcos::make_exceptional_future is deprecated. Use "
        "pika::make_exceptional_future instead.")
    pika::future<T> make_exceptional_future(E e)
    {
        return pika::make_exceptional_future<T>(PIKA_MOVE(e));
    }

    template <int DeductionGuard = 0, typename T>
    PIKA_DEPRECATED_V(0, 1,
        "pika::lcos::make_ready_future_at is deprecated. Use "
        "pika::make_ready_future_at instead.")
    pika::future<pika::util::decay_unwrap_t<T>> make_ready_future_at(
        pika::chrono::steady_time_point const& abs_time, T&& init)
    {
        return pika::make_ready_future_at(abs_time, PIKA_FORWARD(T, init));
    }

    template <int DeductionGuard = 0, typename T>
    PIKA_DEPRECATED_V(0, 1,
        "pika::lcos::make_ready_future_after is deprecated. Use "
        "pika::make_ready_future_after instead.")
    pika::future<pika::util::decay_unwrap_t<T>> make_ready_future_after(
        pika::chrono::steady_duration const& rel_time, T&& init)
    {
        return pika::make_ready_future_at(
            rel_time.from_now(), PIKA_FORWARD(T, init));
    }

    template <typename Allocator>
    PIKA_DEPRECATED_V(0, 1,
        "pika::lcos::make_ready_future_alloc is deprecated. Use "
        "pika::make_ready_future_alloc instead.")
    pika::future<void> make_ready_future_alloc(Allocator const& a)
    {
        return pika::make_ready_future_alloc<void>(a, util::unused);
    }

    PIKA_DEPRECATED_V(0, 1,
        "pika::lcos::make_ready_future is deprecated. Use "
        "pika::make_ready_future instead.")
    inline pika::future<void> make_ready_future()
    {
        return pika::make_ready_future_alloc<void>(
            pika::util::internal_allocator<>{}, util::unused);
    }

    template <typename T>
    PIKA_DEPRECATED_V(0, 1,
        "pika::lcos::make_ready_future is deprecated. Use "
        "pika::make_ready_future instead.")
    std::enable_if_t<std::is_void_v<T>, pika::future<void>> make_ready_future()
    {
        return pika::make_ready_future();
    }

    PIKA_DEPRECATED_V(0, 1,
        "pika::lcos::make_ready_future_at is deprecated. Use "
        "pika::make_ready_future_at instead.")
    inline pika::future<void> make_ready_future_at(
        pika::chrono::steady_time_point const& abs_time)
    {
        return pika::make_ready_future_at(abs_time);
    }

    template <typename T>
    PIKA_DEPRECATED_V(0, 1,
        "pika::lcos::make_ready_future_at is deprecated. Use "
        "pika::make_ready_future_at instead.")
    std::enable_if_t<std::is_void_v<T>, pika::future<void>> make_ready_future_at(
        pika::chrono::steady_time_point const& abs_time)
    {
        return pika::make_ready_future_at(abs_time);
    }

    PIKA_DEPRECATED_V(0, 1,
        "pika::lcos::make_ready_future_after is deprecated. Use "
        "pika::make_ready_future_after instead.")
    inline pika::future<void> make_ready_future_after(
        pika::chrono::steady_duration const& rel_time)
    {
        return pika::make_ready_future_at(rel_time.from_now());
    }

    template <typename T>
    PIKA_DEPRECATED_V(0, 1,
        "pika::lcos::make_ready_future_after is deprecated. Use "
        "pika::make_ready_future_after instead.")
    std::enable_if_t<std::is_void_v<T>,
        pika::future<void>> make_ready_future_after(pika::chrono::
            steady_duration const& rel_time)
    {
        return pika::make_ready_future_at(rel_time.from_now());
    }

    template <typename R>
    PIKA_DEPRECATED_V(0, 1,
        "pika::lcos::make_shared_future is deprecated. Use "
        "pika::make_shared_future instead.")
    pika::shared_future<R> make_shared_future(pika::future<R>&& f) noexcept
    {
        return f.share();
    }

    template <typename R>
    PIKA_DEPRECATED_V(0, 1,
        "pika::lcos::make_shared_future is deprecated. Use "
        "pika::make_shared_future instead.")
    pika::shared_future<R>& make_shared_future(pika::shared_future<R>& f) noexcept
    {
        return f;
    }

    template <typename R>
    PIKA_DEPRECATED_V(0, 1,
        "pika::lcos::make_shared_future is deprecated. Use "
        "pika::make_shared_future instead.")
    pika::shared_future<R>&& make_shared_future(
        pika::shared_future<R>&& f) noexcept
    {
        return PIKA_MOVE(f);
    }

    template <typename R>
    PIKA_DEPRECATED_V(0, 1,
        "pika::lcos::make_shared_future is deprecated. Use "
        "pika::make_shared_future instead.")
    pika::shared_future<R> const& make_shared_future(
        pika::shared_future<R> const& f) noexcept
    {
        return f;
    }
}}    // namespace pika::lcos

#include <pika/futures/packaged_continuation.hpp>

#define PIKA_MAKE_EXCEPTIONAL_FUTURE(T, errorcode, f, msg)                      \
    pika::make_exceptional_future<T>(PIKA_GET_EXCEPTION(errorcode, f, msg)) /**/
