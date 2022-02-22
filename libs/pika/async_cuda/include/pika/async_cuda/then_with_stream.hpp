//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#include <pika/assert.hpp>
#include <pika/async_cuda/cublas_handle.hpp>
#include <pika/async_cuda/cuda_exception.hpp>
#include <pika/async_cuda/cuda_scheduler.hpp>
#include <pika/async_cuda/cusolver_handle.hpp>
#include <pika/async_cuda/custom_blas_api.hpp>
#include <pika/async_cuda/custom_gpu_api.hpp>
#include <pika/async_cuda/custom_lapack_api.hpp>
#include <pika/async_cuda/detail/cuda_event_callback.hpp>
#include <pika/datastructures/tuple.hpp>
#include <pika/execution/algorithms/detail/partial_algorithm.hpp>
#include <pika/execution/algorithms/then.hpp>
#include <pika/execution_base/receiver.hpp>
#include <pika/execution_base/sender.hpp>
#include <pika/functional/detail/tag_fallback_invoke.hpp>
#include <pika/type_support/pack.hpp>

#include <exception>
#include <functional>
#include <string>
#include <type_traits>
#include <utility>

namespace pika::cuda::experimental::detail {
    PIKA_EXPORT pika::cuda::experimental::cublas_handle const&
    get_thread_local_cublas_handle(
        cuda_stream const& stream, cublasPointerMode_t pointer_mode);

    template <typename F, typename... Ts>
    auto invoke_with_thread_local_cublas_handle(cuda_stream const& stream,
        cublasPointerMode_t pointer_mode, F&& f, Ts&&... ts)
        -> decltype(PIKA_INVOKE(PIKA_FORWARD(F, f),
            std::declval<cublasHandle_t>(), PIKA_FORWARD(Ts, ts)...))
    {
        return PIKA_INVOKE(PIKA_FORWARD(F, f),
            get_thread_local_cublas_handle(stream, pointer_mode).get(),
            PIKA_FORWARD(Ts, ts)...);
    }

#if defined(PIKA_HAVE_CUDA)
    PIKA_EXPORT pika::cuda::experimental::cusolver_handle const&
    get_thread_local_cusolver_handle(cuda_stream const& stream);

    template <typename F, typename... Ts>
    auto invoke_with_thread_local_cusolver_handle(cuda_stream const& stream,
        F&& f, Ts&&... ts) -> decltype(PIKA_INVOKE(PIKA_FORWARD(F, f),
        std::declval<cusolverDnHandle_t>(), PIKA_FORWARD(Ts, ts)...))
    {
        return PIKA_INVOKE(PIKA_FORWARD(F, f),
            get_thread_local_cusolver_handle(stream).get(),
            PIKA_FORWARD(Ts, ts)...);
    }
#endif

    template <typename R, typename... Ts>
    void set_value_event_callback_helper(cudaError_t status, R&& r, Ts&&... ts)
    {
        static_assert(sizeof...(Ts) <= 1, "Expecting at most one value");

        PIKA_ASSERT(status != cudaErrorNotReady);

        if (status == cudaSuccess)
        {
            pika::execution::experimental::set_value(
                PIKA_FORWARD(R, r), PIKA_FORWARD(Ts, ts)...);
        }
        else
        {
            pika::execution::experimental::set_error(PIKA_FORWARD(R, r),
                std::make_exception_ptr(
                    cuda_exception(std::string("Getting event after "
                                               "CUDA stream transform "
                                               "failed with status ") +
                            cudaGetErrorString(status),
                        status)));
        }
    }

    template <typename... Ts>
    void extend_argument_lifetimes(
        cuda_scheduler&& sched, cuda_stream const& stream, Ts&&... ts)
    {
        if constexpr (sizeof...(Ts) > 0)
        {
            detail::add_event_callback(
                [keep_alive = pika::make_tuple(PIKA_MOVE(sched),
                     PIKA_FORWARD(Ts, ts)...)](cudaError_t status) mutable {
                    PIKA_ASSERT(status != cudaErrorNotReady);
                    PIKA_UNUSED(status);
                },
                stream.get());
        }
    }

    template <typename R, typename... Ts>
    void set_value_immediate_void(
        cuda_scheduler&& sched, cuda_stream const& stream, R&& r, Ts&&... ts)
    {
        pika::execution::experimental::set_value(PIKA_FORWARD(R, r));

        // Even though we call set_value immediately, we still extend the
        // life time of the arguments by capturing them in a callback that
        // is triggered when the event is ready.
        extend_argument_lifetimes(
            PIKA_MOVE(sched), stream, PIKA_FORWARD(Ts, ts)...);
    }

    template <typename R, typename... Ts>
    void set_value_event_callback_void(
        cuda_scheduler&& sched, cuda_stream const& stream, R&& r, Ts&&... ts)
    {
        detail::add_event_callback(
            [r = PIKA_FORWARD(R, r),
                keep_alive = pika::make_tuple(PIKA_MOVE(sched),
                    PIKA_FORWARD(Ts, ts)...)](cudaError_t status) mutable {
                set_value_event_callback_helper(status, PIKA_MOVE(r));
            },
            stream.get());
    }

    template <typename R, typename T, typename... Ts>
    void set_value_immediate_non_void(cuda_scheduler&& sched,
        cuda_stream const& stream, R&& r, T&& t, Ts&&... ts)
    {
        pika::execution::experimental::set_value(
            PIKA_FORWARD(R, r), PIKA_FORWARD(T, t));

        // Even though we call set_value immediately, we still extend the
        // life time of the arguments by capturing them in a callback that
        // is triggered when the event is ready.
        extend_argument_lifetimes(
            PIKA_MOVE(sched), stream, PIKA_FORWARD(Ts, ts)...);
    }

    template <typename R, typename T, typename... Ts>
    void set_value_event_callback_non_void(cuda_scheduler&& sched,
        cuda_stream const& stream, R&& r, T&& t, Ts&&... ts)
    {
        detail::add_event_callback(
            [t = PIKA_FORWARD(T, t), r = PIKA_FORWARD(R, r),
                keep_alive = pika::make_tuple(PIKA_MOVE(sched),
                    PIKA_FORWARD(Ts, ts)...)](cudaError_t status) mutable {
                set_value_event_callback_helper(
                    status, PIKA_MOVE(r), PIKA_MOVE(t));
            },
            stream.get());
    }

    template <typename R, typename F>
    struct then_with_cuda_receiver;

    template <typename R>
    struct is_then_with_cuda_receiver : std::false_type
    {
    };

    template <typename R, typename F>
    struct is_then_with_cuda_receiver<then_with_cuda_receiver<R, F>>
      : std::true_type
    {
    };

    template <typename R, typename F>
    struct then_with_cuda_receiver
    {
        PIKA_NO_UNIQUE_ADDRESS std::decay_t<R> r;
        PIKA_NO_UNIQUE_ADDRESS std::decay_t<F> f;
        cuda_scheduler sched;
        pika::optional<std::reference_wrapper<const cuda_stream>> stream;

        template <typename R_, typename F_>
        then_with_cuda_receiver(R_&& r, F_&& f, cuda_scheduler sched)
          : r(PIKA_FORWARD(R_, r))
          , f(PIKA_FORWARD(F_, f))
          , sched(PIKA_MOVE(sched))
        {
        }

        then_with_cuda_receiver(then_with_cuda_receiver&&) = default;
        then_with_cuda_receiver& operator=(then_with_cuda_receiver&&) = default;
        then_with_cuda_receiver(then_with_cuda_receiver const&) = delete;
        then_with_cuda_receiver& operator=(
            then_with_cuda_receiver const&) = delete;

        template <typename E>
        friend void tag_invoke(pika::execution::experimental::set_error_t,
            then_with_cuda_receiver&& r, E&& e) noexcept
        {
            pika::execution::experimental::set_error(
                PIKA_MOVE(r.r), PIKA_FORWARD(E, e));
        }

        friend void tag_invoke(pika::execution::experimental::set_done_t,
            then_with_cuda_receiver&& r) noexcept
        {
            pika::execution::experimental::set_done(PIKA_MOVE(r.r));
        }

        template <typename... Ts>
        void set_value(Ts&&... ts) noexcept
        {
            pika::detail::try_catch_exception_ptr(
                [&]() mutable {
                    if (!stream)
                    {
                        stream.emplace(sched.get_next_stream());
                    }

                    // If the next receiver is also a then_with_cuda_receiver
                    // and it uses the same scheduler/pool we set its stream to
                    // the same as for this task.
                    bool successor_uses_same_stream = false;
                    if constexpr (is_then_with_cuda_receiver<
                                      std::decay_t<R>>::value)
                    {
                        if (sched == r.sched)
                        {
                            PIKA_ASSERT(stream);
                            PIKA_ASSERT(!r.stream);
                            r.stream = stream;

                            successor_uses_same_stream = true;
                        }
                    }

                    if constexpr (std::is_void_v<
                                      typename pika::util::invoke_result<F,
                                          cuda_stream const&, Ts...>::type>)
                    {
                        // When the return type is void, there is no value to
                        // forward to the receiver
                        PIKA_INVOKE(PIKA_MOVE(f), stream.value(), ts...);

                        if constexpr (is_then_with_cuda_receiver<
                                          std::decay_t<R>>::value)
                        {
                            if (successor_uses_same_stream)
                            {
                                // When the next receiver uses the same stream
                                // we can immediately call set_value, with the
                                // knowledge that a later receiver will
                                // synchronize the stream when a
                                // non-then_with_cuda receiver is connected.
                                set_value_immediate_void(PIKA_MOVE(sched),
                                    stream.value(), PIKA_MOVE(r),
                                    PIKA_FORWARD(Ts, ts)...);
                            }
                            else
                            {
                                // When the streams are different, we add a
                                // callback which will call set_value on the
                                // receiver.
                                set_value_event_callback_void(PIKA_MOVE(sched),
                                    stream.value(), PIKA_MOVE(r),
                                    PIKA_FORWARD(Ts, ts)...);
                            }
                        }
                        else
                        {
                            // When the next receiver is not a
                            // then_with_cuda_receiver, we add a callback
                            // which will call set_value on the receiver.
                            set_value_event_callback_void(PIKA_MOVE(sched),
                                stream.value(), PIKA_MOVE(r),
                                PIKA_FORWARD(Ts, ts)...);
                        }
                    }
                    else
                    {
                        // When the return type is non-void, we have to forward
                        // the value to the receiver
                        auto t = PIKA_INVOKE(PIKA_MOVE(f), stream.value(),
                            PIKA_FORWARD(Ts, ts)...);

                        if constexpr (is_then_with_cuda_receiver<
                                          std::decay_t<R>>::value)
                        {
                            if (successor_uses_same_stream)
                            {
                                // When the next receiver uses the same stream
                                // we can immediately call set_value, with the
                                // knowledge that a later receiver will
                                // synchronize the stream when a
                                // non-then_with_cuda receiver is connected.
                                set_value_immediate_non_void(PIKA_MOVE(sched),
                                    stream.value(), PIKA_MOVE(r), PIKA_MOVE(t),
                                    PIKA_FORWARD(Ts, ts)...);
                            }
                            else
                            {
                                // When the streams are different, we add a
                                // callback which will call set_value on the
                                // receiver.
                                set_value_event_callback_non_void(
                                    PIKA_MOVE(sched), stream.value(),
                                    PIKA_MOVE(r), PIKA_MOVE(t),
                                    PIKA_FORWARD(Ts, ts)...);
                            }
                        }
                        else
                        {
                            // When the next receiver is not a
                            // then_with_cuda_receiver, we add a callback
                            // which will call set_value on the receiver.
                            set_value_event_callback_non_void(PIKA_MOVE(sched),
                                stream.value(), PIKA_MOVE(r), PIKA_MOVE(t),
                                PIKA_FORWARD(Ts, ts)...);
                        }
                    }
                },
                [&](std::exception_ptr ep) mutable {
                    pika::execution::experimental::set_error(
                        PIKA_MOVE(r), PIKA_MOVE(ep));
                });
        }
    };

    template <typename S, typename F, typename Enable>
    struct then_with_cuda_sender;

    template <typename S>
    struct is_then_with_cuda_sender : std::false_type
    {
    };

    template <typename S, typename F, typename Enable>
    struct is_then_with_cuda_sender<then_with_cuda_sender<S, F, Enable>>
      : std::true_type
    {
    };

    template <typename S, typename F>
    struct is_cuda_stream_invocable_with_sender
    {
        template <typename Tuple>
        struct is_invocable_helper;

        template <template <typename...> class Tuple, typename... Ts>
        struct is_invocable_helper<Tuple<Ts...>>
        {
            using type = pika::is_invocable<F, cuda_stream const&, Ts...>;
        };

        static constexpr bool value = pika::util::detail::change_pack_t<
            pika::util::all_of,
            pika::util::detail::transform_t<
                typename pika::execution::experimental::sender_traits<S>::
                    template value_types<pika::util::pack, pika::util::pack>,
                is_invocable_helper>>::value;
    };

    template <typename S, typename F>
    inline constexpr bool is_cuda_stream_invocable_with_sender_v =
        is_cuda_stream_invocable_with_sender<S, F>::value;

    template <typename S, typename F,
        typename Enable =
            std::enable_if_t<is_cuda_stream_invocable_with_sender_v<S, F>>>
    struct then_with_cuda_sender
    {
        PIKA_NO_UNIQUE_ADDRESS std::decay_t<S> s;
        PIKA_NO_UNIQUE_ADDRESS std::decay_t<F> f;
        cuda_scheduler sched;

        template <typename S_, typename F_>
        then_with_cuda_sender(S_&& s, F_&& f, cuda_scheduler sched)
          : s(PIKA_FORWARD(S_, s))
          , f(PIKA_FORWARD(F_, f))
          , sched(PIKA_MOVE(sched))
        {
        }

        then_with_cuda_sender(then_with_cuda_sender&&) = default;
        then_with_cuda_sender& operator=(then_with_cuda_sender&&) = default;
        then_with_cuda_sender(then_with_cuda_sender const&) = default;
        then_with_cuda_sender& operator=(
            then_with_cuda_sender const&) = default;

        template <typename Tuple>
        struct invoke_result_helper;

        template <template <typename...> class Tuple, typename... Ts>
        struct invoke_result_helper<Tuple<Ts...>>
        {
            using result_type =
                pika::util::invoke_result_t<F, cuda_stream const&, Ts...>;
            using type = std::conditional_t<std::is_void_v<result_type>,
                Tuple<>, Tuple<result_type>>;
        };

        template <template <typename...> class Tuple,
            template <typename...> class Variant>
        using value_types =
            pika::util::detail::unique_t<pika::util::detail::transform_t<
                typename pika::execution::experimental::sender_traits<
                    S>::template value_types<Tuple, Variant>,
                invoke_result_helper>>;

        template <template <typename...> class Variant>
        using error_types =
            pika::util::detail::unique_t<pika::util::detail::prepend_t<
                typename pika::execution::experimental::sender_traits<
                    S>::template error_types<Variant>,
                std::exception_ptr>>;

        static constexpr bool sends_done = false;

        template <typename R>
        friend auto tag_invoke(pika::execution::experimental::connect_t,
            then_with_cuda_sender&& s, R&& r)
        {
            return pika::execution::experimental::connect(PIKA_MOVE(s.s),
                then_with_cuda_receiver<R, F>{
                    PIKA_FORWARD(R, r), PIKA_MOVE(s.f), PIKA_MOVE(s.sched)});
        }

        template <typename R>
        friend auto tag_invoke(pika::execution::experimental::connect_t,
            then_with_cuda_sender& s, R&& r)
        {
            return pika::execution::experimental::connect(s.s,
                then_with_cuda_receiver<R, F>{
                    PIKA_FORWARD(R, r), s.f, s.sched});
        }

        friend cuda_scheduler tag_invoke(
            pika::execution::experimental::get_completion_scheduler_t<
                pika::execution::experimental::set_value_t>,
            then_with_cuda_sender const& s)
        {
            return s.sched;
        }
    };

    // This should be a hidden friend in then_with_cuda_receiver. However,
    // nvcc does not know how to compile it with some argument types
    // ("error: no instance of overloaded function std::forward matches the
    // argument list").
    template <typename R, typename F, typename... Ts>
    void tag_invoke(pika::execution::experimental::set_value_t,
        then_with_cuda_receiver<R, F>&& r, Ts&&... ts)
    {
        r.set_value(PIKA_FORWARD(Ts, ts)...);
    }

    template <typename S, typename F>
    auto then_with_cuda_pool(S&& s, F&& f)
        -> decltype(detail::then_with_cuda_sender<S, F>{PIKA_FORWARD(S, s),
            PIKA_FORWARD(F, f),
            pika::execution::experimental::get_completion_scheduler<
                pika::execution::experimental::set_value_t>(s)})
    {
        auto completion_sched =
            pika::execution::experimental::get_completion_scheduler<
                pika::execution::experimental::set_value_t>(s);
        static_assert(std::is_same_v<std::decay_t<decltype(completion_sched)>,
                          cuda_scheduler>,
            "then_with_cuda_pool can only be used with senders whose "
            "completion scheduler is cuda_scheduler");

        return detail::then_with_cuda_sender<S, F>{PIKA_FORWARD(S, s),
            PIKA_FORWARD(F, f), std::move(completion_sched)};
    }

    // This is a wrapper for functions that expect a cudaStream_t in the last
    // position (as is convention for CUDA functions taking streams).
    template <typename F>
    struct cuda_stream_callable
    {
        std::decay_t<F> f;

        template <typename... Ts>
        auto operator()(cuda_stream const& stream, Ts&&... ts)
        // nvcc does not compile this correctly with noexcept(...)
#if defined(PIKA_CLANG_VERSION)
            noexcept(
                noexcept(PIKA_INVOKE(f, PIKA_FORWARD(Ts, ts)..., stream.get())))
#endif
                -> decltype(
                    PIKA_INVOKE(f, PIKA_FORWARD(Ts, ts)..., stream.get()))
        {
            return PIKA_INVOKE(f, PIKA_FORWARD(Ts, ts)..., stream.get());
        }
    };

    // This is a wrapper for functions that expect a cublasHandle_t in the first
    // position (as is convention for cuBLAS functions taking handles).
    template <typename F>
    struct cublas_handle_callable
    {
        std::decay_t<F> f;
        cublasPointerMode_t pointer_mode;

        template <typename... Ts>
        auto operator()(cuda_stream const& stream, Ts&&... ts)
        // nvcc does not compile this correctly with noexcept(...)
#if defined(PIKA_CLANG_VERSION)
            noexcept(noexcept(invoke_with_thread_local_cublas_handle(
                stream, pointer_mode, f, PIKA_FORWARD(Ts, ts)...)))
#endif
                -> decltype(invoke_with_thread_local_cublas_handle(
                    stream, pointer_mode, f, PIKA_FORWARD(Ts, ts)...))
        {
            return invoke_with_thread_local_cublas_handle(
                stream, pointer_mode, f, PIKA_FORWARD(Ts, ts)...);
        }
    };

#if defined(PIKA_HAVE_CUDA)
    // This is a wrapper for functions that expect a cusolverHandle_t in the
    // first position (as is convention for cuBLAS functions taking handles).
    template <typename F>
    struct cusolver_handle_callable
    {
        std::decay_t<F> f;

        template <typename... Ts>
        auto operator()(cuda_stream const& stream, Ts&&... ts)
        // nvcc does not compile this correctly with noexcept(...)
#if defined(PIKA_CLANG_VERSION)
            noexcept(noexcept(invoke_with_thread_local_cusolver_handle(
                stream, f, PIKA_FORWARD(Ts, ts)...)))
#endif
                -> decltype(invoke_with_thread_local_cusolver_handle(
                    stream, f, PIKA_FORWARD(Ts, ts)...))
        {
            return invoke_with_thread_local_cusolver_handle(
                stream, f, PIKA_FORWARD(Ts, ts)...);
        }
    };
#endif
}    // namespace pika::cuda::experimental::detail

namespace pika::cuda::experimental {
    // NOTE: None of the below are customizations of then. They have different
    // semantics:
    // - a stream/handle is inserted as an additional argument into the call to
    //   f
    // - values from the predecessor sender are not forwarded, only passed by
    //   reference, to the call to f to keep them alive until the event is ready
    // - this operation can only be used when the predecessor sender has
    //   cuda_scheduler as its completion scheduler

    /// Attach a continuation to run f with an additional CUDA stream.
    ///
    /// Attaches a continuation to the given sender which will call f with the
    /// arguments sent by the given sender with an additional cudaStream_t
    /// argument as the last argument. This can only be called on a sender with
    /// a completion scheduler that is cuda_scheduler. f does not have exclusive
    /// access to the given stream and other calls may reuse the same stream
    /// concurrently.
    inline constexpr struct then_with_stream_t final
    {
        template <typename S, typename F>
        constexpr PIKA_FORCEINLINE auto operator()(S&& s, F&& f) const
            -> decltype(detail::then_with_cuda_pool(PIKA_FORWARD(S, s),
                detail::cuda_stream_callable<F>{PIKA_FORWARD(F, f)}))
        {
            return detail::then_with_cuda_pool(PIKA_FORWARD(S, s),
                detail::cuda_stream_callable<F>{PIKA_FORWARD(F, f)});
        }

        template <typename F>
        constexpr PIKA_FORCEINLINE auto operator()(F&& f) const -> decltype(
            pika::execution::experimental::detail::partial_algorithm<
                then_with_stream_t, F>{PIKA_FORWARD(F, f)})
        {
            return pika::execution::experimental::detail::partial_algorithm<
                then_with_stream_t, F>{PIKA_FORWARD(F, f)};
        }
    } then_with_stream{};

    /// Attach a continuation to run f with an additional cuBLAS handle.
    ///
    /// Attaches a continuation to the given sender which will call f with the
    /// arguments sent by the given sender with an additional cublasHandle_t
    /// argument as the first argument. This can only be called on a sender with
    /// a completion scheduler that is cuda_scheduler. The handle is
    /// thread-local and f may not yield a pika thread until after the handle
    /// has been used the last time by f.
    inline constexpr struct then_with_cublas_t final
    {
        template <typename S, typename F>
        constexpr PIKA_FORCEINLINE auto operator()(
            S&& s, F&& f, cublasPointerMode_t pointer_mode) const
            -> decltype(detail::then_with_cuda_pool(PIKA_FORWARD(S, s),
                detail::cublas_handle_callable<F>{
                    PIKA_FORWARD(F, f), pointer_mode}))
        {
            return detail::then_with_cuda_pool(PIKA_FORWARD(S, s),
                detail::cublas_handle_callable<F>{
                    PIKA_FORWARD(F, f), pointer_mode});
        }

        template <typename F>
        constexpr PIKA_FORCEINLINE auto operator()(
            F&& f, cublasPointerMode_t pointer_mode) const
            -> decltype(
                pika::execution::experimental::detail::partial_algorithm<
                    then_with_cublas_t, F, cublasPointerMode_t>{
                    PIKA_FORWARD(F, f), pointer_mode})
        {
            return pika::execution::experimental::detail::partial_algorithm<
                then_with_cublas_t, F, cublasPointerMode_t>{
                PIKA_FORWARD(F, f), pointer_mode};
        }
    } then_with_cublas{};

#if defined(PIKA_HAVE_CUDA)
    /// Attach a continuation to run f with an additional cuSOLVER handle.
    ///
    /// Attaches a continuation to the given sender which will call f with the
    /// arguments sent by the given sender with an additional cusolverDnHandle_t
    /// argument as the first argument. This can only be called on a sender with
    /// a completion scheduler that is cuda_scheduler. The handle is
    /// thread-local and f may not yield a pika thread until after the handle
    /// has been used the last time by f.
    inline constexpr struct then_with_cusolver_t final
    {
        template <typename S, typename F>
        constexpr PIKA_FORCEINLINE auto operator()(S&& s, F&& f) const
            -> decltype(detail::then_with_cuda_pool(PIKA_FORWARD(S, s),
                detail::cusolver_handle_callable<F>{PIKA_FORWARD(F, f)}))
        {
            return detail::then_with_cuda_pool(PIKA_FORWARD(S, s),
                detail::cusolver_handle_callable<F>{PIKA_FORWARD(F, f)});
        }

        template <typename F>
        constexpr PIKA_FORCEINLINE auto operator()(F&& f) const -> decltype(
            pika::execution::experimental::detail::partial_algorithm<
                then_with_cusolver_t, F>{PIKA_FORWARD(F, f)})
        {
            return pika::execution::experimental::detail::partial_algorithm<
                then_with_cusolver_t, F>{PIKA_FORWARD(F, f)};
        }
    } then_with_cusolver{};
#endif

    /// Attach a continuation to run f with a CUDA stream, cuBLAS handle, or
    /// cuSOLVER handle.
    ///
    /// This is a generic version of then_with_stream, then_with_cublas, and
    /// then_with_cusolver which will use one of the three depending on what f
    /// is callable with. If f is callable with more than one of the mentioned
    /// adaptors they will be prioritized in the given order.
    inline constexpr struct then_with_any_cuda_t final
    {
        template <typename S, typename F,
            PIKA_CONCEPT_REQUIRES_(
                pika::execution::experimental::is_sender_v<S>)>
        constexpr PIKA_FORCEINLINE auto operator()(S&& s, F&& f,
            cublasPointerMode_t pointer_mode = CUBLAS_POINTER_MODE_HOST) const
        {
            if constexpr (std::is_invocable_v<then_with_stream_t, S, F>)
            {
                return then_with_stream(PIKA_FORWARD(S, s), PIKA_FORWARD(F, f));
            }
            else if constexpr (std::is_invocable_v<then_with_cublas_t, S, F,
                                   cublasPointerMode_t>)
            {
                return then_with_cublas(
                    PIKA_FORWARD(S, s), PIKA_FORWARD(F, f), pointer_mode);
            }
#if defined(PIKA_HAVE_CUDA)
            else if constexpr (std::is_invocable_v<then_with_cusolver_t, S, F>)
            {
                return then_with_cusolver(
                    PIKA_FORWARD(S, s), PIKA_FORWARD(F, f));
            }
#endif
            else
            {
                static_assert(sizeof(S) == 0,
                    "Attempting to use then_with_any_cuda, but f is not "
                    "invocable with a CUDA stream as the last argument or "
                    "cuBLAS/cuSOLVER handle as the first argument.");
            }
        }

        template <typename F>
        constexpr PIKA_FORCEINLINE auto operator()(F&& f,
            cublasPointerMode_t pointer_mode = CUBLAS_POINTER_MODE_HOST) const
        {
            return pika::execution::experimental::detail::partial_algorithm<
                then_with_any_cuda_t, F, cublasPointerMode_t>{
                PIKA_FORWARD(F, f), pointer_mode};
        }
    } then_with_any_cuda{};
}    // namespace pika::cuda::experimental
