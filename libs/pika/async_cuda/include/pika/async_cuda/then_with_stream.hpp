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
#include <pika/datastructures/optional.hpp>
#include <pika/datastructures/tuple.hpp>
#include <pika/datastructures/variant.hpp>
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

namespace pika::cuda::experimental::then_with_stream_detail {
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

    template <typename OperationState>
    void set_value_immediate_void(OperationState& op_state)
    {
        PIKA_ASSERT(pika::holds_alternative<pika::monostate>(op_state.result));
        pika::execution::experimental::set_value(PIKA_MOVE(op_state.receiver));
    }

    template <typename Result, typename OperationState>
    void set_value_immediate_non_void(OperationState& op_state)
    {
        PIKA_ASSERT(pika::holds_alternative<Result>(op_state.result));
        pika::execution::experimental::set_value(PIKA_MOVE(op_state.receiver),
            PIKA_MOVE(pika::get<Result>(op_state.result)));
    }

    template <typename OperationState>
    void set_value_event_callback_void(OperationState& op_state)
    {
        detail::add_event_callback(
            [&op_state](cudaError_t status) mutable {
                PIKA_ASSERT(
                    pika::holds_alternative<pika::monostate>(op_state.result));
                set_value_event_callback_helper(
                    status, PIKA_MOVE(op_state.receiver));
            },
            op_state.stream.value().get().get());
    }

    template <typename Result, typename OperationState>
    void set_value_event_callback_non_void(OperationState& op_state)
    {
        detail::add_event_callback(
            [&op_state](cudaError_t status) mutable {
                PIKA_ASSERT(pika::holds_alternative<Result>(op_state.result));
                set_value_event_callback_helper(status,
                    PIKA_MOVE(op_state.receiver),
                    PIKA_MOVE(pika::get<Result>(op_state.result)));
            },
            op_state.stream.value().get().get());
    }

    template <typename Sender, typename F, typename Enable>
    struct then_with_cuda_stream_sender;

    template <typename Sender>
    struct is_then_with_cuda_stream_sender : std::false_type
    {
    };

    template <typename Sender, typename F, typename Enable>
    struct is_then_with_cuda_stream_sender<
        then_with_cuda_stream_sender<Sender, F, Enable>> : std::true_type
    {
    };

    template <typename Sender, typename F>
    struct is_cuda_stream_invocable_with_sender
    {
        template <typename Tuple>
        struct is_invocable_helper;

        template <template <typename...> class Tuple, typename... Ts>
        struct is_invocable_helper<Tuple<Ts...>>
        {
            using type = pika::is_invocable<F, cuda_stream const&,
                std::add_lvalue_reference_t<std::decay_t<Ts>>...>;
        };

        static constexpr bool value = pika::util::detail::change_pack_t<
            pika::util::all_of,
            pika::util::detail::transform_t<
                typename pika::execution::experimental::sender_traits<Sender>::
                    template value_types<pika::util::pack, pika::util::pack>,
                is_invocable_helper>>::value;
    };

    template <typename Sender, typename F>
    inline constexpr bool is_cuda_stream_invocable_with_sender_v =
        is_cuda_stream_invocable_with_sender<Sender, F>::value;

    template <typename Sender, typename F,
        typename Enable =
            std::enable_if_t<is_cuda_stream_invocable_with_sender_v<Sender, F>>>
    struct then_with_cuda_stream_sender
    {
        PIKA_NO_UNIQUE_ADDRESS std::decay_t<Sender> sender;
        PIKA_NO_UNIQUE_ADDRESS std::decay_t<F> f;
        cuda_scheduler sched;

        template <typename Sender_, typename F_>
        then_with_cuda_stream_sender(
            Sender_&& sender, F_&& f, cuda_scheduler sched)
          : sender(PIKA_FORWARD(Sender_, sender))
          , f(PIKA_FORWARD(F_, f))
          , sched(PIKA_MOVE(sched))
        {
        }

        then_with_cuda_stream_sender(then_with_cuda_stream_sender&&) = default;
        then_with_cuda_stream_sender& operator=(
            then_with_cuda_stream_sender&&) = default;
        then_with_cuda_stream_sender(
            then_with_cuda_stream_sender const&) = default;
        then_with_cuda_stream_sender& operator=(
            then_with_cuda_stream_sender const&) = default;

        template <typename Tuple>
        struct invoke_result_helper;

        template <template <typename...> class Tuple, typename... Ts>
        struct invoke_result_helper<Tuple<Ts...>>
        {
            using result_type =
                pika::util::invoke_result_t<F, cuda_stream const&,
                    std::add_lvalue_reference_t<std::decay_t<Ts>>...>;
            using type = std::conditional_t<std::is_void_v<result_type>,
                Tuple<>, Tuple<result_type>>;
        };

        template <template <typename...> class Tuple,
            template <typename...> class Variant>
        using value_types =
            pika::util::detail::unique_t<pika::util::detail::transform_t<
                typename pika::execution::experimental::sender_traits<
                    Sender>::template value_types<Tuple, Variant>,
                invoke_result_helper>>;

        template <template <typename...> class Variant>
        using error_types =
            pika::util::detail::unique_t<pika::util::detail::prepend_t<
                typename pika::execution::experimental::sender_traits<
                    Sender>::template error_types<Variant>,
                std::exception_ptr>>;

        static constexpr bool sends_done = false;

        template <typename Receiver>
        struct operation_state
        {
            PIKA_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;
            PIKA_NO_UNIQUE_ADDRESS std::decay_t<F> f;
            cuda_scheduler sched;
            pika::optional<std::reference_wrapper<const cuda_stream>> stream;

            struct then_with_cuda_stream_receiver_tag
            {
            };

            template <typename R, typename = void>
            struct is_then_with_cuda_stream_receiver : std::false_type
            {
            };

            template <typename R>
            struct is_then_with_cuda_stream_receiver<R,
                std::void_t<typename std::decay_t<
                    R>::then_with_cuda_stream_receiver_tag>> : std::true_type
            {
            };

            struct then_with_cuda_stream_receiver
            {
                using then_with_cuda_stream_receiver_tag = void;

                operation_state& op_state;

                then_with_cuda_stream_receiver(
                    then_with_cuda_stream_receiver&&) = default;
                then_with_cuda_stream_receiver& operator=(
                    then_with_cuda_stream_receiver&&) = default;
                then_with_cuda_stream_receiver(
                    then_with_cuda_stream_receiver const&) = delete;
                then_with_cuda_stream_receiver& operator=(
                    then_with_cuda_stream_receiver const&) = delete;

                template <typename Error>
                friend void tag_invoke(
                    pika::execution::experimental::set_error_t,
                    then_with_cuda_stream_receiver&& r, Error&& error) noexcept
                {
                    pika::execution::experimental::set_error(
                        PIKA_MOVE(r.op_state.receiver),
                        PIKA_FORWARD(Error, error));
                }

                friend void tag_invoke(
                    pika::execution::experimental::set_done_t,
                    then_with_cuda_stream_receiver&& r) noexcept
                {
                    pika::execution::experimental::set_done(
                        PIKA_MOVE(r.op_state.receiver));
                }

                template <typename...>
                struct check_type;

                template <typename... Ts>
                auto set_value(Ts&&... ts) noexcept -> decltype(
                    PIKA_INVOKE(PIKA_MOVE(f), stream.value(), ts...), void())
                {
                    pika::detail::try_catch_exception_ptr(
                        [&]() mutable {
                            if (!op_state.stream)
                            {
                                op_state.stream.emplace(
                                    op_state.sched.get_next_stream());
                            }

                            // If the next receiver is also a
                            // then_with_cuda_stream_receiver and it uses the
                            // same scheduler/pool we set its stream to the same
                            // as for this task.
                            [[maybe_unused]] bool successor_uses_same_stream =
                                false;
                            if constexpr (is_then_with_cuda_stream_receiver<
                                              std::decay_t<Receiver>>::value)
                            {
                                if (op_state.sched ==
                                    op_state.receiver.op_state.sched)
                                {
                                    PIKA_ASSERT(op_state.stream);
                                    PIKA_ASSERT(
                                        !op_state.receiver.op_state.stream);
                                    op_state.receiver.op_state.stream =
                                        op_state.stream;

                                    successor_uses_same_stream = true;
                                }
                            }

                            using invoke_result_type =
                                std::decay_t<pika::util::invoke_result_t<F,
                                    cuda_stream const&,
                                    std::add_lvalue_reference_t<
                                        std::decay_t<Ts>>...>>;
                            constexpr bool is_void_result =
                                std::is_void_v<invoke_result_type>;
                            if constexpr (is_void_result)
                            {
                                // When the return type is void, there is no
                                // value to forward to the receiver
                                PIKA_INVOKE(PIKA_MOVE(op_state.f),
                                    op_state.stream.value(), ts...);

                                if constexpr (is_then_with_cuda_stream_receiver<
                                                  std::decay_t<Receiver>>::
                                                  value)
                                {
                                    if (successor_uses_same_stream)
                                    {
                                        // When the next receiver uses the same
                                        // stream we can immediately call
                                        // set_value, with the knowledge that a
                                        // later receiver will synchronize the
                                        // stream when a
                                        // non-then_with_cuda_stream receiver is
                                        // connected.
                                        set_value_immediate_void(op_state);
                                    }
                                    else
                                    {
                                        // When the streams are different, we
                                        // add a callback which will call
                                        // set_value on the receiver.
                                        set_value_event_callback_void(op_state);
                                    }
                                }
                                else
                                {
                                    // When the next receiver is not a
                                    // then_with_cuda_stream_receiver, we add a
                                    // callback which will call set_value on the
                                    // receiver.
                                    set_value_event_callback_void(op_state);
                                }
                            }
                            else
                            {
                                // When the return type is non-void, we have to
                                // forward the value to the receiver
                                op_state.result
                                    .template emplace<invoke_result_type>(
                                        PIKA_INVOKE(PIKA_MOVE(op_state.f),
                                            op_state.stream.value(), ts...));

                                if constexpr (is_then_with_cuda_stream_receiver<
                                                  std::decay_t<Receiver>>::
                                                  value)
                                {
                                    if (successor_uses_same_stream)
                                    {
                                        // When the next receiver uses the same
                                        // stream we can immediately call
                                        // set_value, with the knowledge that a
                                        // later receiver will synchronize the
                                        // stream when a
                                        // non-then_with_cuda_stream receiver is
                                        // connected.
                                        set_value_immediate_non_void<
                                            invoke_result_type>(op_state);
                                    }
                                    else
                                    {
                                        // When the streams are different, we
                                        // add a callback which will call
                                        // set_value on the receiver.
                                        set_value_event_callback_non_void<
                                            invoke_result_type>(op_state);
                                    }
                                }
                                else
                                {
                                    // When the next receiver is not a
                                    // then_with_cuda_stream_receiver, we add a
                                    // callback which will call set_value on the
                                    // receiver.
                                    set_value_event_callback_non_void<
                                        invoke_result_type>(op_state);
                                }
                            }
                        },
                        [&](std::exception_ptr ep) mutable {
                            pika::execution::experimental::set_error(
                                PIKA_MOVE(op_state.receiver), PIKA_MOVE(ep));
                        });
                }
            };

            // This should be a hidden friend in then_with_cuda_stream_receiver.
            // However, nvcc does not know how to compile it with some argument
            // types ("error: no instance of overloaded function std::forward
            // matches the argument list").
            template <typename... Ts>
            friend auto tag_invoke(pika::execution::experimental::set_value_t,
                then_with_cuda_stream_receiver&& r, Ts&&... ts) noexcept
                -> decltype(r.set_value(PIKA_FORWARD(Ts, ts)...))
            {
                r.set_value(PIKA_FORWARD(Ts, ts)...);
            }

            using operation_state_type =
                pika::execution::experimental::connect_result_t<
                    std::decay_t<Sender>, then_with_cuda_stream_receiver>;
            operation_state_type op_state;

            using ts_type = pika::util::detail::prepend_t<
                typename pika::execution::experimental::sender_traits<
                    std::decay_t<Sender>>::template value_types<pika::tuple,
                    pika::variant>,
                pika::monostate>;
            ts_type ts;

            // We store the return value of f in a variant. We know that
            // value_types of the transform_mpi_sender contains packs of at most
            // one element (the return value of f), so we only specialize
            // result_types_helper for zero or one value. For empty packs we use
            // pika::monostate since we don't need to store anything in that
            // case.
            //
            // All in all, we:
            // - transform one-element packs to the single element, and empty
            //   packs to pika::monostate
            // - add pika::monostate to the pack in case it wasn't there already
            // - remove duplicates in case pika::monostate has been added twice
            // - change the outer pack to a pika::variant
            template <typename Tuple>
            struct result_types_helper;

            template <template <typename...> class Tuple, typename T>
            struct result_types_helper<Tuple<T>>
            {
                using type = std::decay_t<T>;
            };

            template <template <typename...> class Tuple>
            struct result_types_helper<Tuple<>>
            {
                using type = pika::monostate;
            };
            using result_type = pika::util::detail::change_pack_t<pika::variant,
                pika::util::detail::unique_t<pika::util::detail::prepend_t<
                    pika::util::detail::transform_t<
                        then_with_cuda_stream_sender::value_types<
                            pika::util::pack, pika::util::pack>,
                        result_types_helper>,
                    pika::monostate>>>;
            result_type result;

            template <typename Receiver_, typename F_, typename Sender_>
            operation_state(Receiver_&& receiver, F_&& f, cuda_scheduler sched,
                Sender_&& sender)
              : receiver(PIKA_FORWARD(Receiver_, receiver))
              , f(PIKA_FORWARD(F_, f))
              , sched(PIKA_MOVE(sched))
              , op_state(pika::execution::experimental::connect(
                    PIKA_FORWARD(Sender_, sender),
                    then_with_cuda_stream_receiver{*this}))
            {
            }

            friend constexpr void tag_invoke(
                pika::execution::experimental::start_t,
                operation_state& os) noexcept
            {
                pika::execution::experimental::start(os.op_state);
            }
        };

        template <typename Receiver>
        friend auto tag_invoke(pika::execution::experimental::connect_t,
            then_with_cuda_stream_sender&& s, Receiver&& receiver)
        {
            return operation_state<Receiver>(PIKA_FORWARD(Receiver, receiver),
                PIKA_MOVE(s.f), PIKA_MOVE(s.sched), PIKA_MOVE(s.sender));
        }

        template <typename Receiver>
        friend auto tag_invoke(pika::execution::experimental::connect_t,
            then_with_cuda_stream_sender& s, Receiver&& receiver)
        {
            return operation_state<Receiver>(
                PIKA_FORWARD(Receiver, receiver), s.f, s.sched, s.sender);
        }

        friend cuda_scheduler tag_invoke(
            pika::execution::experimental::get_completion_scheduler_t<
                pika::execution::experimental::set_value_t>,
            then_with_cuda_stream_sender const& s)
        {
            return s.sched;
        }
    };

    /// This is a helper that calls f with the values sent by sender and a
    /// cuda_stream as the last argument.
    template <typename Sender, typename F>
    auto then_with_cuda_stream(Sender&& sender, F&& f) -> decltype(
        then_with_stream_detail::then_with_cuda_stream_sender<Sender, F>{
            PIKA_FORWARD(Sender, sender), PIKA_FORWARD(F, f),
            pika::execution::experimental::get_completion_scheduler<
                pika::execution::experimental::set_value_t>(sender)})
    {
        auto completion_sched =
            pika::execution::experimental::get_completion_scheduler<
                pika::execution::experimental::set_value_t>(sender);
        static_assert(std::is_same_v<std::decay_t<decltype(completion_sched)>,
                          cuda_scheduler>,
            "then_with_cuda_stream can only be used with senders whose "
            "completion scheduler is cuda_scheduler");

        return then_with_stream_detail::then_with_cuda_stream_sender<Sender, F>{
            PIKA_FORWARD(Sender, sender), PIKA_FORWARD(F, f),
            std::move(completion_sched)};
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
}    // namespace pika::cuda::experimental::then_with_stream_detail

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
        template <typename Sender, typename F>
        constexpr PIKA_FORCEINLINE auto operator()(Sender&& sender, F&& f) const
            -> decltype(then_with_stream_detail::then_with_cuda_stream(
                PIKA_FORWARD(Sender, sender),
                then_with_stream_detail::cuda_stream_callable<F>{
                    PIKA_FORWARD(F, f)}))
        {
            return then_with_stream_detail::then_with_cuda_stream(
                PIKA_FORWARD(Sender, sender),
                then_with_stream_detail::cuda_stream_callable<F>{
                    PIKA_FORWARD(F, f)});
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
        template <typename Sender, typename F>
        constexpr PIKA_FORCEINLINE auto operator()(
            Sender&& sender, F&& f, cublasPointerMode_t pointer_mode) const
            -> decltype(then_with_stream_detail::then_with_cuda_stream(
                PIKA_FORWARD(Sender, sender),
                then_with_stream_detail::cublas_handle_callable<F>{
                    PIKA_FORWARD(F, f), pointer_mode}))
        {
            return then_with_stream_detail::then_with_cuda_stream(
                PIKA_FORWARD(Sender, sender),
                then_with_stream_detail::cublas_handle_callable<F>{
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
        template <typename Sender, typename F>
        constexpr PIKA_FORCEINLINE auto operator()(Sender&& sender, F&& f) const
            -> decltype(then_with_stream_detail::then_with_cuda_stream(
                PIKA_FORWARD(Sender, sender),
                then_with_stream_detail::cusolver_handle_callable<F>{
                    PIKA_FORWARD(F, f)}))
        {
            return then_with_stream_detail::then_with_cuda_stream(
                PIKA_FORWARD(Sender, sender),
                then_with_stream_detail::cusolver_handle_callable<F>{
                    PIKA_FORWARD(F, f)});
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
        template <typename Sender, typename F,
            PIKA_CONCEPT_REQUIRES_(
                pika::execution::experimental::is_sender_v<Sender>)>
        constexpr PIKA_FORCEINLINE auto operator()(Sender&& sender, F&& f,
            cublasPointerMode_t pointer_mode = CUBLAS_POINTER_MODE_HOST) const
        {
            if constexpr (std::is_invocable_v<then_with_stream_t, Sender, F>)
            {
                return then_with_stream(
                    PIKA_FORWARD(Sender, sender), PIKA_FORWARD(F, f));
            }
            else if constexpr (std::is_invocable_v<then_with_cublas_t, Sender,
                                   F, cublasPointerMode_t>)
            {
                return then_with_cublas(PIKA_FORWARD(Sender, sender),
                    PIKA_FORWARD(F, f), pointer_mode);
            }
#if defined(PIKA_HAVE_CUDA)
            else if constexpr (std::is_invocable_v<then_with_cusolver_t, Sender,
                                   F>)
            {
                return then_with_cusolver(
                    PIKA_FORWARD(Sender, sender), PIKA_FORWARD(F, f));
            }
#endif
            else
            {
                static_assert(sizeof(Sender) == 0,
                    "Attempting to use then_with_any_cuda, but f is not "
                    "invocable with a CUDA stream as the last argument or "
                    "cuBLAS/cuSOLVER handle as the first argument.");
            }
            // This silences a bogus warning from nvcc about no return from a
            // non-void function.
#if defined(__NVCC__)
            __builtin_unreachable();
#endif
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
