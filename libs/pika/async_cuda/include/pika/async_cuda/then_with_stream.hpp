//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#include <pika/assert.hpp>
#include <pika/async_cuda/cuda_scheduler.hpp>
#include <pika/async_cuda/detail/cuda_event_callback.hpp>
#include <pika/async_cuda_base/cublas_handle.hpp>
#include <pika/async_cuda_base/cusolver_handle.hpp>
#include <pika/async_cuda_base/custom_blas_api.hpp>
#include <pika/async_cuda_base/custom_lapack_api.hpp>
#include <pika/datastructures/variant.hpp>
#include <pika/execution/algorithms/detail/helpers.hpp>
#include <pika/execution/algorithms/detail/partial_algorithm.hpp>
#include <pika/execution/algorithms/then.hpp>
#include <pika/execution_base/receiver.hpp>
#include <pika/execution_base/sender.hpp>
#include <pika/functional/detail/tag_fallback_invoke.hpp>
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
    template <typename F, typename... Ts>
    auto invoke_with_thread_local_cublas_handle(cuda_scheduler& sched, cuda_stream const& stream,
        cublasPointerMode_t pointer_mode, F&& f,
        Ts&&... ts) -> decltype(PIKA_INVOKE(std::forward<F>(f), std::declval<cublasHandle_t>(),
                        std::forward<Ts>(ts)...))
    {
        auto locked_handle = sched.get_cublas_handle(stream, pointer_mode);
        return PIKA_INVOKE(std::forward<F>(f), locked_handle.get().get(), std::forward<Ts>(ts)...);
    }

    template <typename F, typename... Ts>
    auto invoke_with_thread_local_cusolver_handle(cuda_scheduler& sched, cuda_stream const& stream,
        F&& f, Ts&&... ts) -> decltype(PIKA_INVOKE(std::forward<F>(f),
                               std::declval<cusolverDnHandle_t>(), std::forward<Ts>(ts)...))
    {
        auto locked_handle = sched.get_cusolver_handle(stream);
        return PIKA_INVOKE(std::forward<F>(f), locked_handle.get().get(), std::forward<Ts>(ts)...);
    }

    template <typename R, typename... Ts>
    void set_value_event_callback_helper(whip::error_t status, R&& r, Ts&&... ts)
    {
        static_assert(sizeof...(Ts) <= 1, "Expecting at most one value");

        PIKA_ASSERT(status != whip::error_not_ready);

        if (status == whip::success)
        {
            pika::execution::experimental::set_value(std::forward<R>(r), std::forward<Ts>(ts)...);
        }
        else
        {
            pika::execution::experimental::set_error(std::forward<R>(r),
                std::make_exception_ptr(pika::exception(pika::error::unknown_error,
                    fmt::format(
                        "Getting event after CUDA stream transform failed with status {} ({})",
                        static_cast<int>(status), whip::get_error_string(status)))));
        }
    }

    template <typename OperationState>
    void set_value_immediate_void(OperationState& op_state)
    {
        PIKA_ASSERT(pika::detail::holds_alternative<pika::detail::monostate>(op_state.result));
        pika::execution::experimental::set_value(std::move(op_state.receiver));
    }

    template <typename Result, typename OperationState>
    void set_value_immediate_non_void(OperationState& op_state)
    {
        PIKA_ASSERT(pika::detail::holds_alternative<Result>(op_state.result));
        pika::execution::experimental::set_value(
            std::move(op_state.receiver), std::move(pika::detail::get<Result>(op_state.result)));
    }

    template <typename OperationState>
    void set_value_event_callback_void(OperationState& op_state)
    {
        detail::add_event_callback(
            [&op_state](whip::error_t status) mutable {
                PIKA_ASSERT(
                    pika::detail::holds_alternative<pika::detail::monostate>(op_state.result));
                op_state.ts = {};
                set_value_event_callback_helper(status, std::move(op_state.receiver));
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
                set_value_event_callback_helper(status, std::move(op_state.receiver),
                    std::move(pika::detail::get<Result>(op_state.result)));
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
        PIKA_STDEXEC_SENDER_CONCEPT

        PIKA_NO_UNIQUE_ADDRESS std::decay_t<Sender> sender;
        PIKA_NO_UNIQUE_ADDRESS std::decay_t<F> f;
        cuda_scheduler sched;

        template <typename Sender_, typename F_>
        then_with_cuda_stream_sender_type(Sender_&& sender, F_&& f, cuda_scheduler sched)
          : sender(std::forward<Sender_>(sender))
          , f(std::forward<F_>(f))
          , sched(std::move(sched))
        {
        }

        then_with_cuda_stream_sender_type(then_with_cuda_stream_sender_type&&) = default;
        then_with_cuda_stream_sender_type& operator=(then_with_cuda_stream_sender_type&&) = default;
        then_with_cuda_stream_sender_type(then_with_cuda_stream_sender_type const&) = default;
        then_with_cuda_stream_sender_type& operator=(
            then_with_cuda_stream_sender_type const&) = default;

#if defined(PIKA_HAVE_STDEXEC)
        template <typename... Ts>
            requires std::is_invocable_v<F, cuda_scheduler&, cuda_stream const&,
                         std::add_lvalue_reference_t<std::decay_t<Ts>>...>
        using invoke_result_helper =
            pika::execution::experimental::completion_signatures<pika::execution::experimental::
                    detail::result_type_signature_helper_t<std::invoke_result_t<F, cuda_scheduler&,
                        cuda_stream const&, std::add_lvalue_reference_t<std::decay_t<Ts>>...>>>;

        using completion_signatures =
            pika::execution::experimental::transform_completion_signatures_of<std::decay_t<Sender>,
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
            using result_type = std::invoke_result_t<F, cuda_scheduler&, cuda_stream const&,
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
            PIKA_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;
            PIKA_NO_UNIQUE_ADDRESS std::decay_t<F> f;
            cuda_scheduler sched;
            std::optional<std::reference_wrapper<cuda_stream const>> stream;

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
                PIKA_STDEXEC_RECEIVER_CONCEPT
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
                        std::move(r.op_state.receiver), std::forward<Error>(error));
                }

                friend void tag_invoke(pika::execution::experimental::set_stopped_t,
                    then_with_cuda_stream_receiver&& r) noexcept
                {
                    pika::execution::experimental::set_stopped(std::move(r.op_state.receiver));
                }

                template <typename... Ts>
                auto set_value(Ts&&... ts) && noexcept
                    -> decltype(PIKA_INVOKE(std::move(f), op_state.sched, stream.value(), ts...),
                        void())
                {
                    auto r = std::move(*this);
                    pika::detail::try_catch_exception_ptr(
                        [&]() mutable {
                            using ts_element_type = std::tuple<std::decay_t<Ts>...>;
                            r.op_state.ts.template emplace<ts_element_type>(
                                std::forward<Ts>(ts)...);
                            [[maybe_unused]] auto& t = std::get<ts_element_type>(r.op_state.ts);

                            if (!r.op_state.stream)
                            {
                                r.op_state.stream.emplace(r.op_state.sched.get_next_stream());
                            }

                            // If the next receiver is also a
                            // then_with_cuda_stream_receiver and it uses the
                            // same scheduler/pool we set its stream to the same
                            // as for this task.
                            [[maybe_unused]] bool successor_uses_same_stream = false;
                            if constexpr (is_then_with_cuda_stream_receiver<
                                              std::decay_t<Receiver>>::value)
                            {
                                if (r.op_state.sched == r.op_state.receiver.op_state.sched)
                                {
                                    PIKA_ASSERT(r.op_state.stream);
                                    PIKA_ASSERT(!r.op_state.receiver.op_state.stream);
                                    r.op_state.receiver.op_state.stream = r.op_state.stream;

                                    successor_uses_same_stream = true;
                                }
                            }

                            using invoke_result_type = std::decay_t<
                                std::invoke_result_t<F, cuda_scheduler&, cuda_stream const&,
                                    std::add_lvalue_reference_t<std::decay_t<Ts>>...>>;
                            constexpr bool is_void_result = std::is_void_v<invoke_result_type>;
                            if constexpr (is_void_result)
                            {
                                std::apply(
                                    [&](auto&... ts) mutable {
                                        PIKA_INVOKE(std::move(op_state.f), op_state.sched,
                                            r.op_state.stream.value(), ts...);
                                    },
                                    t);

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
                                        set_value_immediate_void(r.op_state);
                                    }
                                    else
                                    {
                                        // When the streams are different, we
                                        // add a callback which will call
                                        // set_value on the receiver.
                                        set_value_event_callback_void(r.op_state);
                                    }
                                }
                                else
                                {
                                    // When the next receiver is not a
                                    // then_with_cuda_stream_receiver, we add a
                                    // callback which will call set_value on the
                                    // receiver.
                                    set_value_event_callback_void(r.op_state);
                                }
                            }
                            else
                            {
                                std::apply(
                                    [&](auto&... ts) mutable {
                                        r.op_state.result.template emplace<invoke_result_type>(
                                            PIKA_INVOKE(std::move(r.op_state.f), r.op_state.sched,
                                                r.op_state.stream.value(), ts...));
                                    },
                                    t);

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
                                        set_value_immediate_non_void<invoke_result_type>(
                                            r.op_state);
                                    }
                                    else
                                    {
                                        // When the streams are different, we
                                        // add a callback which will call
                                        // set_value on the receiver.
                                        set_value_event_callback_non_void<invoke_result_type>(
                                            r.op_state);
                                    }
                                }
                                else
                                {
                                    // When the next receiver is not a
                                    // then_with_cuda_stream_receiver, we add a
                                    // callback which will call set_value on the
                                    // receiver.
                                    set_value_event_callback_non_void<invoke_result_type>(
                                        r.op_state);
                                }
                            }
                        },
                        [&](std::exception_ptr ep) mutable {
                            pika::execution::experimental::set_error(
                                std::move(r.op_state.receiver), std::move(ep));
                        });
                }

                friend constexpr pika::execution::experimental::empty_env tag_invoke(
                    pika::execution::experimental::get_env_t,
                    then_with_cuda_stream_receiver const&) noexcept
                {
                    return {};
                }
            };

            using operation_state_type =
                pika::execution::experimental::connect_result_t<std::decay_t<Sender>,
                    then_with_cuda_stream_receiver>;
            operation_state_type op_state;

            template <typename Tuple>
            struct value_types_helper
            {
                using type = pika::util::detail::transform_t<Tuple, std::decay>;
            };

#if defined(PIKA_HAVE_STDEXEC)
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
#if defined(PIKA_HAVE_STDEXEC)
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
              : receiver(std::forward<Receiver_>(receiver))
              , f(std::forward<F_>(f))
              , sched(std::move(sched))
              , op_state(pika::execution::experimental::connect(
                    std::forward<Sender_>(sender), then_with_cuda_stream_receiver{*this}))
            {
            }

            void start() & noexcept { pika::execution::experimental::start(op_state); }
        };

        template <typename Receiver>
        friend auto tag_invoke(pika::execution::experimental::connect_t,
            then_with_cuda_stream_sender_type&& s, Receiver&& receiver)
        {
            return operation_state<Receiver>(std::forward<Receiver>(receiver), std::move(s.f),
                std::move(s.sched), std::move(s.sender));
        }

        template <typename Receiver>
        friend auto tag_invoke(pika::execution::experimental::connect_t,
            then_with_cuda_stream_sender_type const& s, Receiver&& receiver)
        {
            return operation_state<Receiver>(
                std::forward<Receiver>(receiver), s.f, s.sched, s.sender);
        }

        friend auto tag_invoke(pika::execution::experimental::get_env_t,
            then_with_cuda_stream_sender_type const& s) noexcept
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
            std::forward<Sender>(sender), std::forward<F>(f), std::move(completion_sched)};
    }

    // This is a wrapper for functions that expect a cudaStream_t in the last
    // position (as is convention for CUDA functions taking streams).
    template <typename F>
    struct cuda_stream_callable
    {
        std::decay_t<F> f;

        template <typename... Ts>
        auto operator()(cuda_scheduler&, cuda_stream const& stream, Ts&&... ts)
        // nvcc does not compile this correctly with noexcept(...)
#if defined(PIKA_CLANG_VERSION)
            noexcept(noexcept(PIKA_INVOKE(f, std::forward<Ts>(ts)..., stream.get())))
#endif
                -> decltype(PIKA_INVOKE(f, std::forward<Ts>(ts)..., stream.get()))
        {
            return PIKA_INVOKE(f, std::forward<Ts>(ts)..., stream.get());
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
        auto operator()(cuda_scheduler& sched, cuda_stream const& stream, Ts&&... ts)
        // nvcc does not compile this correctly with noexcept(...)
#if defined(PIKA_CLANG_VERSION)
            noexcept(noexcept(invoke_with_thread_local_cublas_handle(
                sched, stream, pointer_mode, f, std::forward<Ts>(ts)...)))
#endif
                -> decltype(invoke_with_thread_local_cublas_handle(
                    sched, stream, pointer_mode, f, std::forward<Ts>(ts)...))
        {
            return invoke_with_thread_local_cublas_handle(
                sched, stream, pointer_mode, f, std::forward<Ts>(ts)...);
        }
    };

    // This is a wrapper for functions that expect a cusolverHandle_t in the
    // first position (as is convention for cuBLAS functions taking handles).
    template <typename F>
    struct cusolver_handle_callable
    {
        std::decay_t<F> f;

        template <typename... Ts>
        auto operator()(cuda_scheduler& sched, cuda_stream const& stream, Ts&&... ts)
        // nvcc does not compile this correctly with noexcept(...)
#if defined(PIKA_CLANG_VERSION)
            noexcept(noexcept(invoke_with_thread_local_cusolver_handle(
                sched, stream, f, std::forward<Ts>(ts)...)))
#endif
                -> decltype(invoke_with_thread_local_cusolver_handle(
                    sched, stream, f, std::forward<Ts>(ts)...))
        {
            return invoke_with_thread_local_cusolver_handle(
                sched, stream, f, std::forward<Ts>(ts)...);
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

    /// \brief The type of the \ref then_with_stream sender adaptor.
    struct then_with_stream_t final
    {
        /// \brief Create a \ref then_with_stream sender.
        ///
        /// \param sender The predecessor sender.
        /// \param f Callable that will be passed a \p cudaStream_t as the last argument. Values
        /// from \p sender are passed as references.
        template <typename Sender, typename F>
        constexpr PIKA_FORCEINLINE auto PIKA_STATIC_CALL_OPERATOR(Sender&& sender, F&& f)
        {
            return then_with_stream_detail::then_with_cuda_stream(std::forward<Sender>(sender),
                then_with_stream_detail::cuda_stream_callable<F>{std::forward<F>(f)});
        }

        /// \brief Partially bound sender. Expects a sender to be supplied later.
        template <typename F>
        constexpr PIKA_FORCEINLINE auto PIKA_STATIC_CALL_OPERATOR(F&& f)
        {
            return pika::execution::experimental::detail::partial_algorithm<then_with_stream_t, F>{
                std::forward<F>(f)};
        }
    };

    /// \brief Sender adaptor which calls \p f with CUDA stream.
    ///
    /// When the predecessor sender completes, calls \p f with a CUDA stream as the last argument
    /// after other values sent by the predecessor sender. This adaptor can only be used when the
    /// completion scheduler is a \ref cuda_scheduler. Other work may be scheduled concurrently on
    /// the stream passed to \p f. Values sent by the predecessor sender are passed as references to
    /// \p f and kept alive until the work submitted by \p f to the stream is completed. \p f may
    /// return as soon as work has been submitted, and a connected receiver will be signaled only
    /// once the kernels submitted to the stream have completed.
    inline constexpr then_with_stream_t then_with_stream{};

    /// \brief The type of the \ref then_with_cublas sender adaptor.
    struct then_with_cublas_t final
    {
        /// \brief Create a \ref then_with_cublas sender.
        ///
        /// \param sender The predecessor sender.
        /// \param f Callable that will be passed a \p cublasHandle_t as the first argument. Values
        /// from \p sender are passed as references.
        /// \param pointer_mode The \p cublasPointerMode_t used for the internal cuBLAS handle, or
        /// the equivalent for rocBLAS.
        template <typename Sender, typename F>
        constexpr PIKA_FORCEINLINE auto
        PIKA_STATIC_CALL_OPERATOR(Sender&& sender, F&& f, cublasPointerMode_t pointer_mode)
        {
            return then_with_stream_detail::then_with_cuda_stream(std::forward<Sender>(sender),
                then_with_stream_detail::cublas_handle_callable<F>{
                    std::forward<F>(f), pointer_mode});
        }

        /// \brief Partially bound sender. Expects a sender to be supplied later.
        template <typename F>
        constexpr PIKA_FORCEINLINE auto
        PIKA_STATIC_CALL_OPERATOR(F&& f, cublasPointerMode_t pointer_mode)
        {
            return pika::execution::experimental::detail::partial_algorithm<then_with_cublas_t, F,
                cublasPointerMode_t>{std::forward<F>(f), pointer_mode};
        }
    };

    /// \brief Sender adaptor which calls \p f with a cuBLAS handle.
    ///
    /// This sender is intended to be used to submit work using a cuBLAS handle. The stream
    /// associated to the handle may also be used to submit work. The handle is accessed through a
    /// \ref locked_cublas_handle and \p f should return as quickly as possible to avoid blocking
    /// other work from using the handle.
    ///
    /// The behaviour of synchronization and lifetimes are the same as for \ref then_with_stream,
    /// except that the handle is passed as the first argument to match the typical function
    /// signatures of cuBLAS functions.
    inline constexpr then_with_cublas_t then_with_cublas{};

    /// \brief The type of the \ref then_with_cusolver sender adaptor.
    struct then_with_cusolver_t final
    {
        /// \param sender The predecessor sender.
        /// \param f Callable that will be passed a \p cusolverDnHandle_t as the first argument.
        /// Values from \p sender are passed as references.
        template <typename Sender, typename F>
        constexpr PIKA_FORCEINLINE auto PIKA_STATIC_CALL_OPERATOR(Sender&& sender, F&& f)
        {
            return then_with_stream_detail::then_with_cuda_stream(std::forward<Sender>(sender),
                then_with_stream_detail::cusolver_handle_callable<F>{std::forward<F>(f)});
        }

        /// \brief Partially bound sender. Expects a sender to be supplied later.
        template <typename F>
        constexpr PIKA_FORCEINLINE auto PIKA_STATIC_CALL_OPERATOR(F&& f)
        {
            return pika::execution::experimental::detail::partial_algorithm<then_with_cusolver_t,
                F>{std::forward<F>(f)};
        }
    };

    /// \brief Sender adaptor which calls \p f with a cuSOLVER handle.
    ///
    /// This sender is intended to be used to submit work using a cuSOLVER handle. The stream
    /// associated to the handle may also be used to submit work. The handle is accessed through a
    /// \ref locked_cusolver_handle and \p f should return as quickly as possible to avoid blocking
    /// other work from using the handle.
    ///
    /// The behaviour of synchronization and lifetimes are the same as for \ref then_with_stream,
    /// except that the handle is passed as the first argument to match the typical function
    /// signatures of cuBLAS functions.
    inline constexpr then_with_cusolver_t then_with_cusolver{};
}    // namespace pika::cuda::experimental
