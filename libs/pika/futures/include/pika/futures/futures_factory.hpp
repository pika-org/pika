//  Copyright (c) 2007-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#include <pika/allocator_support/allocator_deleter.hpp>
#include <pika/allocator_support/internal_allocator.hpp>
#include <pika/async_base/launch_policy.hpp>
#include <pika/coroutines/thread_enums.hpp>
#include <pika/errors/try_catch_exception_ptr.hpp>
#include <pika/execution_base/execution.hpp>
#include <pika/functional/deferred_call.hpp>
#include <pika/futures/detail/future_data.hpp>
#include <pika/futures/future.hpp>
#include <pika/futures/traits/future_access.hpp>
#include <pika/memory/intrusive_ptr.hpp>
#include <pika/modules/errors.hpp>
#include <pika/threading_base/thread_description.hpp>
#include <pika/threading_base/thread_helpers.hpp>
#include <pika/threading_base/thread_num_tss.hpp>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <type_traits>
#include <utility>

namespace pika::lcos::local::detail {

    template <typename Result, typename F, typename Executor,
        typename Base = lcos::detail::task_base<Result>>
    struct task_object;

    template <typename Result, typename F, typename Base>
    struct task_object<Result, F, void, Base> : Base
    {
        using base_type = Base;
        using result_type = typename Base::result_type;
        using init_no_addref = typename Base::init_no_addref;

        F f_;

        explicit task_object(F const& f)
          : f_(f)
        {
        }

        explicit task_object(F&& f) noexcept
          : f_(PIKA_MOVE(f))
        {
        }

        task_object(init_no_addref no_addref, F const& f)
          : base_type(no_addref)
          , f_(f)
        {
        }

        task_object(init_no_addref no_addref, F&& f) noexcept
          : base_type(no_addref)
          , f_(PIKA_MOVE(f))
        {
        }

        void do_run() noexcept override
        {
            pika::intrusive_ptr<base_type> this_(this);
            pika::detail::try_catch_exception_ptr(
                [&]() {
                    if constexpr (std::is_void_v<Result>)
                    {
                        f_();
                        this->set_value(result_type());
                    }
                    else
                    {
                        this->set_value(f_());
                    }
                },
                [&](std::exception_ptr ep) { this->set_exception(PIKA_MOVE(ep)); });
        }

    protected:
        // run in a separate thread
        threads::detail::thread_id_ref_type apply(threads::detail::thread_pool_base* pool,
            const char* annotation, launch policy, error_code& ec) override
        {
            this->check_started();

            pika::intrusive_ptr<base_type> this_(this);
            if (policy == launch::fork)
            {
                threads::detail::thread_init_data data(
                    threads::detail::make_thread_function_nullary(
                        util::detail::deferred_call(&base_type::run_impl, PIKA_MOVE(this_))),
                    ::pika::detail::thread_description(f_, annotation), policy.priority(),
                    execution::thread_schedule_hint(
                        static_cast<std::uint16_t>(get_worker_thread_num())),
                    policy.stacksize(),
                    threads::detail::thread_schedule_state::pending_do_not_schedule, true);

                return threads::detail::register_thread(data, pool, ec);
            }

            threads::detail::thread_init_data data(
                threads::detail::make_thread_function_nullary(
                    util::detail::deferred_call(&base_type::run_impl, PIKA_MOVE(this_))),
                ::pika::detail::thread_description(f_, annotation), policy.priority(),
                policy.hint(), policy.stacksize(), threads::detail::thread_schedule_state::pending);

            return threads::detail::register_work(data, pool, ec);
        }
    };

    template <typename Allocator, typename Result, typename F, typename Base>
    struct task_object_allocator : task_object<Result, F, void, Base>
    {
        using base_type = task_object<Result, F, void, Base>;
        using result_type = typename base_type::result_type;
        using init_no_addref = typename base_type::init_no_addref;

        using other_allocator =
            typename std::allocator_traits<Allocator>::template rebind_alloc<task_object_allocator>;

        task_object_allocator(other_allocator const& alloc, F const& f)
          : base_type(f)
          , alloc_(alloc)
        {
        }

        task_object_allocator(other_allocator const& alloc, F&& f) noexcept
          : base_type(PIKA_MOVE(f))
          , alloc_(alloc)
        {
        }

        task_object_allocator(init_no_addref no_addref, other_allocator const& alloc, F const& f)
          : base_type(no_addref, f)
          , alloc_(alloc)
        {
        }

        task_object_allocator(
            init_no_addref no_addref, other_allocator const& alloc, F&& f) noexcept
          : base_type(no_addref, PIKA_MOVE(f))
          , alloc_(alloc)
        {
        }

    private:
        void destroy() noexcept override
        {
            using traits = std::allocator_traits<other_allocator>;

            other_allocator alloc(alloc_);
            traits::destroy(alloc, this);
            traits::deallocate(alloc, this, 1);
        }

        other_allocator alloc_;
    };

    template <typename Result, typename F, typename Executor, typename Base>
    struct task_object : task_object<Result, F, void, Base>
    {
        using base_type = task_object<Result, F, void, Base>;
        using result_type = typename base_type::result_type;
        using init_no_addref = typename base_type::init_no_addref;

        Executor* exec_ = nullptr;

        explicit task_object(F const& f)
          : base_type(f)
        {
        }

        explicit task_object(F&& f) noexcept
          : base_type(PIKA_MOVE(f))
        {
        }

        task_object(Executor& exec, F const& f)
          : base_type(f)
          , exec_(&exec)
        {
        }

        task_object(Executor& exec, F&& f) noexcept
          : base_type(PIKA_MOVE(f))
          , exec_(&exec)
        {
        }

        task_object(init_no_addref no_addref, F const& f)
          : base_type(no_addref, f)
        {
        }

        task_object(init_no_addref no_addref, F&& f) noexcept
          : base_type(no_addref, PIKA_MOVE(f))
        {
        }

        task_object(Executor& exec, init_no_addref no_addref, F const& f)
          : base_type(no_addref, f)
          , exec_(&exec)
        {
        }

        task_object(Executor& exec, init_no_addref no_addref, F&& f) noexcept
          : base_type(no_addref, PIKA_MOVE(f))
          , exec_(&exec)
        {
        }

    protected:
        // run in a separate thread
        threads::detail::thread_id_ref_type apply(threads::detail::thread_pool_base* pool,
            const char* annotation, launch policy, error_code& ec) override
        {
            if (exec_)
            {
                this->check_started();

                pika::intrusive_ptr<base_type> this_(this);
                parallel::execution::post(*exec_,
                    util::detail::deferred_call(&base_type::run_impl, PIKA_MOVE(this_)),
                    exec_->get_schedulehint(), annotation);
                return threads::detail::invalid_thread_id;
            }

            return this->base_type::apply(pool, annotation, policy, ec);
        }
    };

    ///////////////////////////////////////////////////////////////////////
    template <typename Result, typename F, typename Executor>
    struct cancelable_task_object;

    template <typename Result, typename F>
    struct cancelable_task_object<Result, F, void>
      : task_object<Result, F, void, lcos::detail::cancelable_task_base<Result>>
    {
        using base_type = task_object<Result, F, void, lcos::detail::cancelable_task_base<Result>>;
        using result_type = typename base_type::result_type;
        using init_no_addref = typename base_type::init_no_addref;

        explicit cancelable_task_object(F const& f)
          : base_type(f)
        {
        }

        explicit cancelable_task_object(F&& f) noexcept
          : base_type(PIKA_MOVE(f))
        {
        }

        cancelable_task_object(init_no_addref no_addref, F const& f)
          : base_type(no_addref, f)
        {
        }

        cancelable_task_object(init_no_addref no_addref, F&& f) noexcept
          : base_type(no_addref, PIKA_MOVE(f))
        {
        }
    };

    template <typename Allocator, typename Result, typename F>
    struct cancelable_task_object_allocator : cancelable_task_object<Result, F, void>
    {
        using base_type = cancelable_task_object<Result, F, void>;
        using result_type = typename base_type::result_type;
        using init_no_addref = typename base_type::init_no_addref;

        using other_allocator = typename std::allocator_traits<Allocator>::template rebind_alloc<
            cancelable_task_object_allocator>;

        cancelable_task_object_allocator(other_allocator const& alloc, F const& f)
          : base_type(f)
          , alloc_(alloc)
        {
        }

        cancelable_task_object_allocator(other_allocator const& alloc, F&& f) noexcept
          : base_type(PIKA_MOVE(f))
          , alloc_(alloc)
        {
        }

        cancelable_task_object_allocator(
            init_no_addref no_addref, other_allocator const& alloc, F const& f)
          : base_type(no_addref, f)
          , alloc_(alloc)
        {
        }

        cancelable_task_object_allocator(
            init_no_addref no_addref, other_allocator const& alloc, F&& f) noexcept
          : base_type(no_addref, PIKA_MOVE(f))
          , alloc_(alloc)
        {
        }

    private:
        void destroy() noexcept override
        {
            using traits = std::allocator_traits<other_allocator>;

            other_allocator alloc(alloc_);
            traits::destroy(alloc, this);
            traits::deallocate(alloc, this, 1);
        }

        other_allocator alloc_;
    };

    template <typename Result, typename F, typename Executor>
    struct cancelable_task_object
      : task_object<Result, F, Executor, lcos::detail::cancelable_task_base<Result>>
    {
        using base_type =
            task_object<Result, F, Executor, lcos::detail::cancelable_task_base<Result>>;
        using result_type = typename base_type::result_type;
        using init_no_addref = typename base_type::init_no_addref;

        explicit cancelable_task_object(F const& f)
          : base_type(f)
        {
        }

        explicit cancelable_task_object(F&& f) noexcept
          : base_type(PIKA_MOVE(f))
        {
        }

        cancelable_task_object(Executor& exec, F const& f)
          : base_type(exec, f)
        {
        }

        cancelable_task_object(Executor& exec, F&& f) noexcept
          : base_type(exec, PIKA_MOVE(f))
        {
        }

        cancelable_task_object(init_no_addref no_addref, F const& f)
          : base_type(no_addref, f)
        {
        }

        cancelable_task_object(init_no_addref no_addref, F&& f) noexcept
          : base_type(no_addref, PIKA_MOVE(f))
        {
        }

        cancelable_task_object(Executor& exec, init_no_addref no_addref, F const& f)
          : base_type(exec, no_addref, f)
        {
        }

        cancelable_task_object(Executor& exec, init_no_addref no_addref, F&& f) noexcept
          : base_type(exec, no_addref, PIKA_MOVE(f))
        {
        }
    };
}    // namespace pika::lcos::local::detail

namespace pika::traits::detail {

    template <typename Result, typename F, typename Base, typename Allocator>
    struct shared_state_allocator<lcos::local::detail::task_object<Result, F, void, Base>,
        Allocator>
    {
        using type = lcos::local::detail::task_object_allocator<Allocator, Result, F, Base>;
    };

    template <typename Result, typename F, typename Allocator>
    struct shared_state_allocator<lcos::local::detail::cancelable_task_object<Result, F, void>,
        Allocator>
    {
        using type = lcos::local::detail::cancelable_task_object_allocator<Allocator, Result, F>;
    };
}    // namespace pika::traits::detail

namespace pika::lcos::local {

    ///////////////////////////////////////////////////////////////////////////
    // The futures_factory is very similar to a packaged_task except that it
    // allows for the owner to go out of scope before the future becomes ready.
    // We provide this class to avoid semantic differences to the C++11
    // std::packaged_task, while otoh it is a very convenient way for us to
    // implement pika::async.
    template <typename Func, bool Cancelable = false>
    class futures_factory;

    namespace detail {
        ///////////////////////////////////////////////////////////////////////
        template <typename Result, bool Cancelable, typename Executor = void>
        struct create_task_object;

        template <typename Result>
        struct create_task_object<Result, false, void>
        {
            using return_type = pika::intrusive_ptr<lcos::detail::task_base<Result>>;
            using init_no_addref = typename lcos::detail::future_data_refcnt_base::init_no_addref;

            template <typename F>
            static return_type call(F&& f)
            {
                return return_type(
                    new task_object<Result, F, void>(init_no_addref{}, PIKA_FORWARD(F, f)), false);
            }

            template <typename R>
            static return_type call(R (*f)())
            {
                return return_type(
                    new task_object<Result, Result (*)(), void>(init_no_addref{}, f), false);
            }

            template <typename Allocator, typename F>
            static return_type call(Allocator const& a, F&& f)
            {
                using base_allocator = Allocator;
                using shared_state =
                    typename traits::detail::shared_state_allocator<task_object<Result, F, void>,
                        base_allocator>::type;

                using other_allocator = typename std::allocator_traits<
                    base_allocator>::template rebind_alloc<shared_state>;
                using traits = std::allocator_traits<other_allocator>;

                using init_no_addref = typename shared_state::init_no_addref;

                using unique_ptr =
                    std::unique_ptr<shared_state, pika::detail::allocator_deleter<other_allocator>>;

                other_allocator alloc(a);
                unique_ptr p(traits::allocate(alloc, 1),
                    pika::detail::allocator_deleter<other_allocator>{alloc});
                traits::construct(alloc, p.get(), init_no_addref{}, alloc, PIKA_FORWARD(F, f));

                return return_type(p.release(), false);
            }

            template <typename Allocator, typename R>
            static return_type call(Allocator const& a, R (*f)())
            {
                using base_allocator = Allocator;
                using shared_state = typename traits::detail::shared_state_allocator<
                    task_object<Result, Result (*)(), void>, base_allocator>::type;

                using other_allocator = typename std::allocator_traits<
                    base_allocator>::template rebind_alloc<shared_state>;
                using traits = std::allocator_traits<other_allocator>;

                using init_no_addref = typename shared_state::init_no_addref;

                using unique_ptr =
                    std::unique_ptr<shared_state, pika::detail::allocator_deleter<other_allocator>>;

                other_allocator alloc(a);
                unique_ptr p(traits::allocate(alloc, 1),
                    pika::detail::allocator_deleter<other_allocator>{alloc});
                traits::construct(alloc, p.get(), init_no_addref{}, alloc, f);

                return return_type(p.release(), false);
            }
        };

        template <typename Result, typename Executor>
        struct create_task_object<Result, false, Executor> : create_task_object<Result, false, void>
        {
            using return_type = pika::intrusive_ptr<lcos::detail::task_base<Result>>;
            using init_no_addref = typename lcos::detail::future_data_refcnt_base::init_no_addref;

            template <typename F>
            static return_type call(Executor& exec, F&& f)
            {
                return return_type(new task_object<Result, F, Executor>(
                                       exec, init_no_addref{}, PIKA_FORWARD(F, f)),
                    false);
            }

            template <typename R>
            static return_type call(Executor& exec, R (*f)())
            {
                return return_type(
                    new task_object<Result, Result (*)(), Executor>(exec, init_no_addref{}, f),
                    false);
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Result>
        struct create_task_object<Result, true, void>
        {
            using return_type = pika::intrusive_ptr<lcos::detail::task_base<Result>>;
            using init_no_addref = typename lcos::detail::future_data_refcnt_base::init_no_addref;

            template <typename F>
            static return_type call(F&& f)
            {
                return return_type(new cancelable_task_object<Result, F, void>(
                                       init_no_addref{}, PIKA_FORWARD(F, f)),
                    false);
            }

            template <typename R>
            static return_type call(R (*f)())
            {
                return return_type(
                    new cancelable_task_object<Result, Result (*)(), void>(init_no_addref{}, f),
                    false);
            }

            template <typename Allocator, typename F>
            static return_type call(Allocator const& a, F&& f)
            {
                using base_allocator = Allocator;
                using shared_state = typename traits::detail::shared_state_allocator<
                    cancelable_task_object<Result, F, void>, base_allocator>::type;

                using other_allocator = typename std::allocator_traits<
                    base_allocator>::template rebind_alloc<shared_state>;
                using traits = std::allocator_traits<other_allocator>;

                using init_no_addref = typename shared_state::init_no_addref;

                using unique_ptr =
                    std::unique_ptr<shared_state, pika::detail::allocator_deleter<other_allocator>>;

                other_allocator alloc(a);
                unique_ptr p(traits::allocate(alloc, 1),
                    pika::detail::allocator_deleter<other_allocator>{alloc});
                traits::construct(alloc, p.get(), init_no_addref{}, alloc, PIKA_FORWARD(F, f));

                return return_type(p.release(), false);
            }

            template <typename Allocator, typename R>
            static return_type call(Allocator const& a, R (*f)())
            {
                using base_allocator = Allocator;
                using shared_state = typename traits::detail::shared_state_allocator<
                    cancelable_task_object<Result, Result (*)(), void>, base_allocator>::type;

                using other_allocator = typename std::allocator_traits<
                    base_allocator>::template rebind_alloc<shared_state>;
                using traits = std::allocator_traits<other_allocator>;

                using init_no_addref = typename shared_state::init_no_addref;

                using unique_ptr =
                    std::unique_ptr<shared_state, pika::detail::allocator_deleter<other_allocator>>;

                other_allocator alloc(a);
                unique_ptr p(traits::allocate(alloc, 1),
                    pika::detail::allocator_deleter<other_allocator>{alloc});
                traits::construct(alloc, p.get(), init_no_addref{}, alloc, f);

                return return_type(p.release(), false);
            }
        };

        template <typename Result, typename Executor>
        struct create_task_object<Result, true, Executor> : create_task_object<Result, true, void>
        {
            using return_type = pika::intrusive_ptr<lcos::detail::task_base<Result>>;
            using init_no_addref = typename lcos::detail::future_data_refcnt_base::init_no_addref;

            template <typename F>
            static return_type call(Executor& exec, F&& f)
            {
                return return_type(new cancelable_task_object<Result, F, Executor>(
                                       exec, init_no_addref{}, PIKA_FORWARD(F, f)),
                    false);
            }

            template <typename R>
            static return_type call(Executor& exec, R (*f)())
            {
                return return_type(new cancelable_task_object<Result, Result (*)(), Executor>(
                                       exec, init_no_addref{}, f),
                    false);
            }
        };
    }    // namespace detail

    template <typename Result, bool Cancelable>
    class futures_factory<Result(), Cancelable>
    {
    protected:
        using task_impl_type = lcos::detail::task_base<Result>;

    public:
        // construction and destruction
        futures_factory() = default;

        template <typename Executor, typename F>
        explicit futures_factory(Executor& exec, F&& f)
          : task_(detail::create_task_object<Result, Cancelable, Executor>::call(
                exec, PIKA_FORWARD(F, f)))
        {
        }

        template <typename Executor>
        explicit futures_factory(Executor& exec, Result (*f)())
          : task_(detail::create_task_object<Result, Cancelable, Executor>::call(exec, f))
        {
        }

        template <typename F,
            typename Enable = std::enable_if_t<!std::is_same_v<std::decay_t<F>, futures_factory>>>
        explicit futures_factory(F&& f)
          : task_(detail::create_task_object<Result, Cancelable>::call(
                pika::detail::internal_allocator<>{}, PIKA_FORWARD(F, f)))
        {
        }

        explicit futures_factory(Result (*f)())
          : task_(detail::create_task_object<Result, Cancelable>::call(
                pika::detail::internal_allocator<>{}, f))
        {
        }

        ~futures_factory() = default;

        futures_factory(futures_factory const& rhs) = delete;
        futures_factory& operator=(futures_factory const& rhs) = delete;

        futures_factory(futures_factory&& rhs) noexcept
          : task_(PIKA_MOVE(rhs.task_))
          , future_obtained_(rhs.future_obtained_)
        {
            rhs.task_.reset();
            rhs.future_obtained_ = false;
        }

        futures_factory& operator=(futures_factory&& rhs) noexcept
        {
            if (this != &rhs)
            {
                task_ = PIKA_MOVE(rhs.task_);
                future_obtained_ = rhs.future_obtained_;

                rhs.task_.reset();
                rhs.future_obtained_ = false;
            }
            return *this;
        }

        // synchronous execution
        void operator()() const
        {
            if (!task_)
            {
                PIKA_THROW_EXCEPTION(pika::error::task_moved,
                    "futures_factory<Result()>::operator()",
                    "futures_factory invalid (has it been moved?)");
                return;
            }
            task_->run();
        }

        // asynchronous execution
        threads::detail::thread_id_ref_type apply(const char* annotation = "futures_factory::apply",
            launch policy = launch::async, error_code& ec = throws) const
        {
            return apply(threads::detail::get_self_or_default_pool(), annotation, policy, ec);
        }

        threads::detail::thread_id_ref_type apply(threads::detail::thread_pool_base* pool,
            const char* annotation = "futures_factory::apply", launch policy = launch::async,
            error_code& ec = throws) const
        {
            if (!task_)
            {
                PIKA_THROW_EXCEPTION(pika::error::task_moved, "futures_factory<Result()>::apply()",
                    "futures_factory invalid (has it been moved?)");
                return threads::detail::invalid_thread_id;
            }
            return task_->apply(pool, annotation, policy, ec);
        }

        // This is the same as get_future, except that it moves the
        // shared state into the returned future.
        pika::future<Result> get_future(error_code& ec = throws)
        {
            if (!task_)
            {
                PIKA_THROWS_IF(ec, pika::error::task_moved, "futures_factory<Result()>::get_future",
                    "futures_factory invalid (has it been moved?)");
                return pika::future<Result>();
            }
            if (future_obtained_)
            {
                PIKA_THROWS_IF(ec, pika::error::future_already_retrieved,
                    "futures_factory<Result()>::get_future",
                    "future already has been retrieved from this factory");
                return pika::future<Result>();
            }

            future_obtained_ = true;

            using traits::future_access;
            return future_access<pika::future<Result>>::create(PIKA_MOVE(task_));
        }

        constexpr bool valid() const noexcept
        {
            return !!task_;
        }

        void set_exception(std::exception_ptr const& e)
        {
            if (!task_)
            {
                PIKA_THROW_EXCEPTION(pika::error::task_moved,
                    "futures_factory<Result()>::set_exception",
                    "futures_factory invalid (has it been moved?)");
                return;
            }
            task_->set_exception(e);
        }

    protected:
        pika::intrusive_ptr<task_impl_type> task_;
        bool future_obtained_ = false;
    };
}    // namespace pika::lcos::local
