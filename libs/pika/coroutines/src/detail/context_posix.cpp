//  Copyright (c) 2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/config.hpp>
#include <pika/coroutines/detail/context_impl.hpp>

// The preprocessor conditions below are kept in sync with those used in
// context_impl.hpp

#if defined(PIKA_HAVE_BOOST_CONTEXT)

// left empty on purpose

#elif (defined(__linux) || defined(linux) || defined(__linux__) || defined(__FreeBSD__)) &&        \
    !defined(__bgq__) && !defined(__powerpc__) && !defined(__s390x__) && !defined(__arm__) &&      \
    !defined(__arm64__) && !defined(__aarch64__)

// left empty on purpose

#elif defined(_POSIX_VERSION) || defined(__bgq__) || defined(__powerpc__) || defined(__s390x__) || \
    defined(__arm__) || defined(__arm64__) || defined(__aarch64__)

# include <cstddef>

namespace pika::threads::coroutines::detail::posix {
    std::ptrdiff_t ucontext_context_impl_base::default_stack_size = SIGSTKSZ;
}    // namespace pika::threads::coroutines::detail::posix

#elif defined(PIKA_HAVE_FIBER_BASED_COROUTINES)

// left empty on purpose

#else

# error No default_context_impl available for this system

#endif    // PIKA_HAVE_BOOST_CONTEXT
