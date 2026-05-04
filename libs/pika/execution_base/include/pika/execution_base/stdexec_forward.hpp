//  Copyright (c) 2022 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#if defined(PIKA_HAVE_STDEXEC)
# include <stdexec/execution.hpp>
# if defined(PIKA_HAVE_STDEXEC_TRANSFORM_COMPLETION_SIGNATURES)
#  include <exec/completion_signatures.hpp>
# endif
namespace pika::execution::experimental {
    using namespace stdexec;
    using stdexec::get_completion_scheduler_t;

    // continue_on_t has been renamed to continues_on_t (valid in version from 02.2026)
# if !defined(PIKA_HAVE_STDEXEC_CONTINUES_ON)
    using continues_on_t = stdexec::continue_on_t;
    inline constexpr continues_on_t continues_on{};
# endif

    // empty_env is now deprecated in stdexec
# if defined(PIKA_HAVE_STDEXEC_ENV)
    using empty_env = stdexec::env<>;
# endif

    // transform_completion_signatures_of is deprecated in newer stdexec. The public replacement
    // is ::experimental::execution::transform_completion_signatures in
    // <exec/completion_signatures.hpp>, as named by the stdexec deprecation message. We use the
    // fully qualified namespace rather than the `exec` alias since the alias is not part of the
    // stable API. The leading :: is required because we are inside pika::execution::experimental,
    // where unqualified `experimental` would resolve to the enclosing namespace.
# if defined(PIKA_HAVE_STDEXEC_TRANSFORM_COMPLETION_SIGNATURES)
    namespace detail {
        // Default template-template transforms: keep each completion as-is, matching stdexec's
        // deprecated __cmplsigs::__default_set_value / __default_set_error.
        template <class... Ts>
        using default_set_value = stdexec::completion_signatures<stdexec::set_value_t(Ts...)>;

        template <class... Errs>
        using default_set_error = stdexec::completion_signatures<stdexec::set_error_t(Errs...)>;

        // The new exec::transform_completion_signatures takes function-object transforms. The
        // deprecated transform_completion_signatures_of took template-template transforms that
        // map a pack to a completion_signatures alias. This adapter lets callers keep the old
        // template-template API while forwarding to the new function-object API.
        template <template <class...> class F>
        struct template_alias_to_fn
        {
            template <class... Args>
            consteval auto operator()() const noexcept
            {
                return F<Args...>{};
            }
        };
    }    // namespace detail

    template <class Sndr, class Env = empty_env, class MoreSigs = stdexec::completion_signatures<>,
        template <class...> class ValueTransform = detail::default_set_value,
        template <class...> class ErrorTransform = detail::default_set_error>
    using transform_completion_signatures_of =
        decltype(::experimental::execution::transform_completion_signatures(
            stdexec::get_completion_signatures<Sndr, Env>(),
            detail::template_alias_to_fn<ValueTransform>{},
            detail::template_alias_to_fn<ErrorTransform>{},
            ::experimental::execution::keep_completion<stdexec::set_stopped_t>{}, MoreSigs{}));
# endif
}    // namespace pika::execution::experimental
#endif
