//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#include <pika/assert.hpp>
#include <pika/async_cuda/cublas_handle.hpp>
#include <pika/async_cuda/cuda_scheduler.hpp>
#include <pika/async_cuda/cusolver_handle.hpp>
#include <pika/async_cuda/custom_blas_api.hpp>
#include <pika/async_cuda/custom_lapack_api.hpp>
#include <pika/async_cuda/detail/cuda_event_callback.hpp>
#include <pika/datastructures/variant.hpp>
#include <pika/execution/algorithms/detail/helpers.hpp>
#include <pika/execution/algorithms/detail/partial_algorithm.hpp>
#include <pika/execution/algorithms/then.hpp>
#include <pika/execution_base/receiver.hpp>
#include <pika/execution_base/sender.hpp>
#include <pika/functional/detail/tag_fallback_invoke.hpp>
#include <pika/functional/invoke_fused.hpp>
#include <pika/type_support/pack.hpp>

#include <fmt/format.h>
#include <whip.hpp>

#include <exception>
#include <functional>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

namespace pika::cuda::experimental::then_with_stream_detail {
    PIKA_EXPORT pika::cuda::experimental::cublas_handle const& get_thread_local_cublas_handle(
        cuda_stream const& stream, cublasPointerMode_t pointer_mode);

    template <typename F, typename... Ts>
    auto invoke_with_thread_local_cublas_handle(
        cuda_stream const& stream, cublasPointerMode_t pointer_mode, F&& f, Ts&&... ts)
        -> decltype(PIKA_INVOKE(
            PIKA_FORWARD(F, f), std::declval<cublasHandle_t>(), PIKA_FORWARD(Ts, ts)...))
    {
        return PIKA_INVOKE(PIKA_FORWARD(F, f),
            get_thread_local_cublas_handle(stream, pointer_mode).get(), PIKA_FORWARD(Ts, ts)...);
    }

    PIKA_EXPORT pika::cuda::experimental::cusolver_handle const& get_thread_local_cusolver_handle(
        cuda_stream const& stream);

    template <typename F, typename... Ts>
    auto invoke_with_thread_local_cusolver_handle(cuda_stream const& stream, F&& f, Ts&&... ts)
        -> decltype(PIKA_INVOKE(
            PIKA_FORWARD(F, f), std::declval<cusolverDnHandle_t>(), PIKA_FORWARD(Ts, ts)...))
    {
        return PIKA_INVOKE(PIKA_FORWARD(F, f), get_thread_local_cusolver_handle(stream).get(),
            PIKA_FORWARD(Ts, ts)...);
    }

    template <typename R, typename... Ts>
    void set_value_event_callback_helper(whip::error_t status, R&& r, Ts&&... ts)
    {
        static_assert(sizeof...(Ts) <= 1, "Expecting at most one value");

        PIKA_ASSERT(status != whip::error_not_ready);

        if (status == whip::success)
        {
            pika::execution::experimental::set_value(PIKA_FORWARD(R, r), PIKA_FORWARD(Ts, ts)...);
        }
        else
        {
            pika::execution::experimental::set_error(PIKA_FORWARD(R, r),
                std::make_exception_ptr(pika::exception(pika::error::unknown_error,
                    fmt::format(
                        "Getting event after CUDA stream transform failed with status {} ({})",
                        status, whip::get_error_string(status)))));
        }
    }

    template <typename OperationState>
    void set_value_immediate_void(OperationState& op_state)
    {
        PIKA_ASSERT(pika::detail::holds_alternative<pika::detail::monostate>(op_state.result));
        pika::execution::experimental::set_value(PIKA_MOVE(op_state.receiver));
    }

    template <typename Result, typename OperationState>
    void set_value_immediate_non_void(OperationState& op_state)
    {
        PIKA_ASSERT(pika::detail::holds_alternative<Result>(op_state.result));
        pika::execution::experimental::set_value(
            PIKA_MOVE(op_state.receiver), PIKA_MOVE(pika::detail::get<Result>(op_state.result)));
    }

    template <typename OperationState>
    void set_value_event_callback_void(OperationState& op_state)
    {
        detail::add_event_callback(
            [&op_state](whip::error_t status) mutable {
                PIKA_ASSERT(
                    pika::detail::holds_alternative<pika::detail::monostate>(op_state.result));
                op_state.ts = {};
                set_value_event_callback_helper(status, PIKA_MOVE(op_state.receiver));
            },
            op_state.stream.value().get());
    }

    template <typename Result, typename OperationState>
    void set_value_event_callback_non_void(OperationState& op_state)
    {
        detail::add_event_callback(
            [&op_state](whip::error_t status) mutable {
                PIKA_ASSERT(pika::detail::holds_alternative<Result>(op_state.result));
                op_state.ts = {};
                set_value_event_callback_helper(status, PIKA_MOVE(op_state.receiver),
                    PIKA_MOVE(pika::detail::get<Result>(op_state.result)));
            },
            op_state.stream.value().get());
    }

    template <typename Sender, typename F>
    struct then_with_cuda_stream_sender_impl
    {
        struct then_with_cuda_stream_sender_type;
    };

    template <typename Sender, typename F>
    using then_with_cuda_stream_sender =
        typename then_with_cuda_stream_sender_impl<Sender, F>::then_with_cuda_stream_sender_type;

    template <typename Sender, typename F>
    struct then_with_cuda_stream_sender_impl<Sender, F>::then_with_cuda_stream_sender_type
    {
        using is_sender = void;

        // nvcc 12.0 is not able to compile this with no_unique_address
#if defined(PIKA_CUDA_VERSION) && PIKA_CUDA_VERSION >= 1200
        std::decay_t<Sender> sender;
        std::decay_t<F> f;
#else
        PIKA_NO_UNIQUE_ADDRESS std::decay_t<Sender> sender;
        PIKA_NO_UNIQUE_ADDRESS std::decay_t<F> f;
#endif
        cuda_scheduler sched;

        template <typename Sender_, typename F_>
        then_with_cuda_stream_sender_type(Sender_&& sender, F_&& f, cuda_scheduler sched)
          : sender(PIKA_FORWARD(Sender_, sender))
          , f(PIKA_FORWARD(F_, f))
          , sched(PIKA_MOVE(sched))
        {
        }

        then_with_cuda_stream_sender_type(then_with_cuda_stream_sender_type&&) = default;
        then_with_cuda_stream_sender_type& operator=(then_with_cuda_stream_sender_type&&) = default;
        then_with_cuda_stream_sender_type(then_with_cuda_stream_sender_type const&) = default;
        then_with_cuda_stream_sender_type& operator=(
            then_with_cuda_stream_sender_type const&) = default;

#if defined(PIKA_HAVE_P2300_REFERENCE_IMPLEMENTATION)
        template <typename... Ts>
        requires std::is_invocable_v<F, cuda_stream const&,
            std::add_lvalue_reference_t<std::decay_t<Ts>>...>
        using invoke_result_helper =
            pika::execution::experimental::completion_signatures<pika::execution::experimental::
                    detail::result_type_signature_helper_t<std::invoke_result_t<F,
                        cuda_stream const&, std::add_lvalue_reference_t<std::decay_t<Ts>>...>>>;

        using completion_signatures =
            pika::execution::experimental::make_completion_signatures<std::decay_t<Sender>,
                pika::execution::experimental::empty_env,
                pika::execution::experimental::completion_signatures<
                    pika::execution::experimental::set_error_t(std::exception_ptr)>,
                invoke_result_helper>;
#else
        template <typename Tuple>
        struct invoke_result_helper;

        template <template <typename...> class Tuple, typename... Ts>
        struct invoke_result_helper<Tuple<Ts...>>
        {
            using result_type = std::invoke_result_t<F, cuda_stream const&,
                std::add_lvalue_reference_t<std::decay_t<Ts>>...>;
            using type =
                std::conditional_t<std::is_void_v<result_type>, Tuple<>, Tuple<result_type>>;
        };

        template <template <typename...> class Tuple, template <typename...> class Variant>
        using value_types = pika::util::detail::unique_t<
            pika::util::detail::transform_t<typename pika::execution::experimental::sender_traits<
                                                Sender>::template value_types<Tuple, Variant>,
                invoke_result_helper>>;

        template <template <typename...> class Variant>
        using error_types = pika::util::detail::unique_t<
            pika::util::detail::prepend_t<typename pika::execution::experimental::sender_traits<
                                              Sender>::template error_types<Variant>,
                std::exception_ptr>>;

        static constexpr bool sends_done = false;
#endif

        template <typename Receiver>
        struct operation_state
        {
            // nvcc 12.0 is not able to compile this with no_unique_address
#if defined(PIKA_CUDA_VERSION) && PIKA_CUDA_VERSION >= 1200
            std::decay_t<Receiver> receiver;
            std::decay_t<F> f;
#else
            PIKA_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;
            PIKA_NO_UNIQUE_ADDRESS std::decay_t<F> f;
#endif
            cuda_scheduler sched;
            std::optional<std::reference_wrapper<const cuda_stream>> stream;

            struct then_with_cuda_stream_receiver_tag
            {
            };

            template <typename R, typename = void>
            struct is_then_with_cuda_stream_receiver : std::false_type
            {
            };

            template <typename R>
            struct is_then_with_cuda_stream_receiver<R,
                std::void_t<typename std::decay_t<R>::then_with_cuda_stream_receiver_tag>>
              : std::true_type
            {
            };

            struct then_with_cuda_stream_receiver
            {
                using is_receiver = void;
                using then_with_cuda_stream_receiver_tag = void;

                operation_state& op_state;

                explicit then_with_cuda_stream_receiver(operation_state& op_state)
                  : op_state(op_state)
                {
                }
                then_with_cuda_stream_receiver(then_with_cuda_stream_receiver&&) = default;
                then_with_cuda_stream_receiver& operator=(
                    then_with_cuda_stream_receiver&&) = default;
                then_with_cuda_stream_receiver(then_with_cuda_stream_receiver const&) = delete;
                then_with_cuda_stream_receiver& operator=(
                    then_with_cuda_stream_receiver const&) = delete;

                template <typename Error>
                friend void tag_invoke(pika::execution::experimental::set_error_t,
                    then_with_cuda_stream_receiver&& r, Error&& error) noexcept
                {
                    pika::execution::experimental::set_error(
                        PIKA_MOVE(r.op_state.receiver), PIKA_FORWARD(Error, error));
                }

                friend void tag_invoke(pika::execution::experimental::set_stopped_t,
                    then_with_cuda_stream_receiver&& r) noexcept
                {
                    pika::execution::experimental::set_stopped(PIKA_MOVE(r.op_state.receiver));
                }

                template <typename... Ts>
                auto set_value(Ts&&... ts) noexcept
                    -> decltype(PIKA_INVOKE(PIKA_MOVE(f), stream.value(), ts...), void())
                {
                    pika::detail::try_catch_exception_ptr(
                        [&]() mutable {
                            using ts_element_type = std::tuple<std::decay_t<Ts>...>;
                            op_state.ts.template emplace<ts_element_type>(PIKA_FORWARD(Ts, ts)...);
                            [[maybe_unused]] auto& t = std::get<ts_element_type>(op_state.ts);

                            if (!op_state.stream)
                            {
                                op_state.stream.emplace(op_state.sched.get_next_stream());
                            }

                            // If the next receiver is also a
                            // then_with_cuda_stream_receiver and it uses the
                            // same scheduler/pool we set its stream to the same
                            // as for this task.
                            [[maybe_unused]] bool successor_uses_same_stream = false;
                            if constexpr (is_then_with_cuda_stream_receiver<
                                              std::decay_t<Receiver>>::value)
                            {
                                if (op_state.sched == op_state.receiver.op_state.sched)
                                {
                                    PIKA_ASSERT(op_state.stream);
                                    PIKA_ASSERT(!op_state.receiver.op_state.stream);
                                    op_state.receiver.op_state.stream = op_state.stream;

                                    successor_uses_same_stream = true;
                                }
                            }

                            using invoke_result_type =
                                std::decay_t<std::invoke_result_t<F, cuda_stream const&,
                                    std::add_lvalue_reference_t<std::decay_t<Ts>>...>>;
                            constexpr bool is_void_result = std::is_void_v<invoke_result_type>;
                            if constexpr (is_void_result)
                            {
                            // nvcc fails to compile the invoke_fused call
                            // until version 11.3.  Since this is never meant
                            // to be called on the device, we can disable it
                            // from being compiled on the device. However, if
                            // we completely disable it device side functions
                            // will not be correctly instantiated, so we
                            // compile a dummy form invoke_fused instead that
                            // should also never be called.
#if defined(__NVCC__) && defined(PIKA_COMPUTE_DEVICE_CODE) && defined(PIKA_CUDA_VERSION) &&        \
    (PIKA_CUDA_VERSION < 1103)
                                PIKA_ASSERT(false);
                                PIKA_INVOKE(PIKA_MOVE(op_state.f), op_state.stream.value(), ts...);
#else
                                // When the return type is void, there is no
                                // value to forward to the receiver
                                pika::util::detail::invoke_fused(
                                    [&](auto&... ts) mutable {
                                        PIKA_INVOKE(
                                            PIKA_MOVE(op_state.f), op_state.stream.value(), ts...);
                                    },
                                    t);
#endif

                                if constexpr (is_then_with_cuda_stream_receiver<
                                                  std::decay_t<Receiver>>::value)
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
                            // nvcc fails to compile the invoke_fused call
                            // until version 11.3.  Since this is never meant
                            // to be called on the device, we can disable it
                            // from being compiled on the device. However, if
                            // we completely disable it device side functions
                            // will not be correctly instantiated, so we
                            // compile a dummy form invoke_fused instead that
                            // should also never be called.
#if defined(__NVCC__) && defined(PIKA_COMPUTE_DEVICE_CODE) && defined(PIKA_CUDA_VERSION) &&        \
    (PIKA_CUDA_VERSION < 1103)
                                PIKA_ASSERT(false);
                                op_state.result.template emplace<invoke_result_type>(PIKA_INVOKE(
                                    PIKA_MOVE(op_state.f), op_state.stream.value(), ts...));
#else
                                // When the return type is non-void, we have to
                                // forward the value to the receiver
                                pika::util::detail::invoke_fused(
                                    [&](auto&... ts) mutable {
                                        op_state.result.template emplace<invoke_result_type>(
                                            PIKA_INVOKE(PIKA_MOVE(op_state.f),
                                                op_state.stream.value(), ts...));
                                    },
                                    t);
#endif

                                if constexpr (is_then_with_cuda_stream_receiver<
                                                  std::decay_t<Receiver>>::value)
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
                                        set_value_immediate_non_void<invoke_result_type>(op_state);
                                    }
                                    else
                                    {
                                        // When the streams are different, we
                                        // add a callback which will call
                                        // set_value on the receiver.
                                        set_value_event_callback_non_void<invoke_result_type>(
                                            op_state);
                                    }
                                }
                                else
                                {
                                    // When the next receiver is not a
                                    // then_with_cuda_stream_receiver, we add a
                                    // callback which will call set_value on the
                                    // receiver.
                                    set_value_event_callback_non_void<invoke_result_type>(op_state);
                                }
                            }
                        },
                        [&](std::exception_ptr ep) mutable {
                            pika::execution::experimental::set_error(
                                PIKA_MOVE(op_state.receiver), PIKA_MOVE(ep));
                        });
                }

                friend constexpr pika::execution::experimental::empty_env tag_invoke(
                    pika::execution::experimental::get_env_t,
                    then_with_cuda_stream_receiver const&) noexcept
                {
                    return {};
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
                pika::execution::experimental::connect_result_t<std::decay_t<Sender>,
                    then_with_cuda_stream_receiver>;
            operation_state_type op_state;

            template <typename Tuple>
            struct value_types_helper
            {
                using type = pika::util::detail::transform_t<Tuple, std::decay>;
            };

#if defined(PIKA_HAVE_P2300_REFERENCE_IMPLEMENTATION)
            using ts_type = pika::util::detail::prepend_t<
                pika::util::detail::transform_t<
                    pika::execution::experimental::value_types_of_t<std::decay_t<Sender>,
                        pika::execution::experimental::empty_env, std::tuple,
                        pika::detail::variant>,
                    value_types_helper>,
                pika::detail::monostate>;
#else
            using ts_type = pika::util::detail::prepend_t<
                pika::util::detail::transform_t<
                    typename pika::execution::experimental::sender_traits<std::decay_t<Sender>>::
                        template value_types<std::tuple, pika::detail::variant>,
                    value_types_helper>,
                pika::detail::monostate>;
#endif
            ts_type ts;

            // We store the return value of f in a variant. We know that
            // value_types of the then_with_cuda_sender contains packs of at
            // most one element (the return value of f), so we only specialize
            // result_types_helper for zero or one value. For empty packs we use
            // pika::detail::monostate since we don't need to store anything in
            // that case.
            //
            // All in all, we:
            // - transform one-element packs to the single element, and empty
            //   packs to pika::detail::monostate
            // - add pika::detail::monostate to the pack in case it wasn't there already
            // - remove duplicates in case pika::detail::monostate has been added twice
            // - change the outer pack to a pika::detail::variant
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
                using type = pika::detail::monostate;
            };
#if defined(PIKA_HAVE_P2300_REFERENCE_IMPLEMENTATION)
            using result_type = pika::util::detail::change_pack_t<pika::detail::variant,
                pika::util::detail::unique_t<pika::util::detail::prepend_t<
                    pika::util::detail::transform_t<
                        pika::execution::experimental::value_types_of_t<
                            then_with_cuda_stream_sender_type,
                            pika::execution::experimental::empty_env, pika::util::detail::pack,
                            pika::util::detail::pack>,
                        result_types_helper>,
                    pika::detail::monostate>>>;
#else
            using result_type = pika::util::detail::change_pack_t<pika::detail::variant,
                pika::util::detail::unique_t<pika::util::detail::prepend_t<
                    pika::util::detail::transform_t<
                        then_with_cuda_stream_sender_type::value_types<pika::util::detail::pack,
                            pika::util::detail::pack>,
                        result_types_helper>,
                    pika::detail::monostate>>>;
#endif
            result_type result;

            template <typename Receiver_, typename F_, typename Sender_>
            operation_state(Receiver_&& receiver, F_&& f, cuda_scheduler sched, Sender_&& sender)
              : receiver(PIKA_FORWARD(Receiver_, receiver))
              , f(PIKA_FORWARD(F_, f))
              , sched(PIKA_MOVE(sched))
              , op_state(pika::execution::experimental::connect(
                    PIKA_FORWARD(Sender_, sender), then_with_cuda_stream_receiver{*this}))
            {
            }

            friend constexpr void tag_invoke(
                pika::execution::experimental::start_t, operation_state& os) noexcept
            {
                pika::execution::experimental::start(os.op_state);
            }
        };

        template <typename Receiver>
        friend auto tag_invoke(pika::execution::experimental::connect_t,
            then_with_cuda_stream_sender_type&& s, Receiver&& receiver)
        {
            return operation_state<Receiver>(PIKA_FORWARD(Receiver, receiver), PIKA_MOVE(s.f),
                PIKA_MOVE(s.sched), PIKA_MOVE(s.sender));
        }

        template <typename Receiver>
        friend auto tag_invoke(pika::execution::experimental::connect_t,
            then_with_cuda_stream_sender_type const& s, Receiver&& receiver)
        {
            return operation_state<Receiver>(
                PIKA_FORWARD(Receiver, receiver), s.f, s.sched, s.sender);
        }

        friend auto tag_invoke(
            pika::execution::experimental::get_env_t, then_with_cuda_stream_sender_type const& s)
        {
            return pika::execution::experimental::get_env(s.sender);
        }
    };

    /// This is a helper that calls f with the values sent by sender and a
    /// cuda_stream as the last argument.
    template <typename Sender, typename F>
    auto then_with_cuda_stream(Sender&& sender, F&& f)
    {
        auto completion_sched = pika::execution::experimental::get_completion_scheduler<
            pika::execution::experimental::set_value_t>(
            pika::execution::experimental::get_env(sender));
        static_assert(std::is_same_v<std::decay_t<decltype(completion_sched)>, cuda_scheduler>,
            "then_with_cuda_stream can only be used with senders whose completion scheduler is "
            "cuda_scheduler");

        return then_with_stream_detail::then_with_cuda_stream_sender<Sender, F>{
            PIKA_FORWARD(Sender, sender), PIKA_FORWARD(F, f), std::move(completion_sched)};
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
            noexcept(noexcept(PIKA_INVOKE(f, PIKA_FORWARD(Ts, ts)..., stream.get())))
#endif
                -> decltype(PIKA_INVOKE(f, PIKA_FORWARD(Ts, ts)..., stream.get()))
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
            noexcept(noexcept(
                invoke_with_thread_local_cusolver_handle(stream, f, PIKA_FORWARD(Ts, ts)...)))
#endif
                -> decltype(invoke_with_thread_local_cusolver_handle(
                    stream, f, PIKA_FORWARD(Ts, ts)...))
        {
            return invoke_with_thread_local_cusolver_handle(stream, f, PIKA_FORWARD(Ts, ts)...);
        }
    };
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
        {
            return then_with_stream_detail::then_with_cuda_stream(PIKA_FORWARD(Sender, sender),
                then_with_stream_detail::cuda_stream_callable<F>{PIKA_FORWARD(F, f)});
        }

        template <typename F>
        constexpr PIKA_FORCEINLINE auto operator()(F&& f) const
        {
            return pika::execution::experimental::detail::partial_algorithm<then_with_stream_t, F>{
                PIKA_FORWARD(F, f)};
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
        constexpr PIKA_FORCEINLINE auto
        operator()(Sender&& sender, F&& f, cublasPointerMode_t pointer_mode) const
        {
            return then_with_stream_detail::then_with_cuda_stream(PIKA_FORWARD(Sender, sender),
                then_with_stream_detail::cublas_handle_callable<F>{
                    PIKA_FORWARD(F, f), pointer_mode});
        }

        template <typename F>
        constexpr PIKA_FORCEINLINE auto operator()(F&& f, cublasPointerMode_t pointer_mode) const
        {
            return pika::execution::experimental::detail::partial_algorithm<then_with_cublas_t, F,
                cublasPointerMode_t>{PIKA_FORWARD(F, f), pointer_mode};
        }
    } then_with_cublas{};

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
        {
            return then_with_stream_detail::then_with_cuda_stream(PIKA_FORWARD(Sender, sender),
                then_with_stream_detail::cusolver_handle_callable<F>{PIKA_FORWARD(F, f)});
        }

        template <typename F>
        constexpr PIKA_FORCEINLINE auto operator()(F&& f) const
        {
            return pika::execution::experimental::detail::partial_algorithm<then_with_cusolver_t,
                F>{PIKA_FORWARD(F, f)};
        }
    } then_with_cusolver{};
}    // namespace pika::cuda::experimental
