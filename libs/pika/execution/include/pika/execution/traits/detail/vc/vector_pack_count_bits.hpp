//  Copyright (c) 2007-2016 Hartmut Kaiser
//  Copyright (c) 2016 Matthias Kretz
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>

#if defined(PIKA_HAVE_DATAPAR_VC)
#include <cstddef>

#include <Vc/global.h>

#if defined(Vc_IS_VERSION_1) && Vc_IS_VERSION_1

#include <Vc/Vc>

namespace pika { namespace parallel { namespace traits {
    ///////////////////////////////////////////////////////////////////////
    template <typename T, typename Abi>
    PIKA_HOST_DEVICE PIKA_FORCEINLINE std::size_t count_bits(
        Vc::Mask<T, Abi> const& mask)
    {
        return mask.count();
    }
}}}    // namespace pika::parallel::traits

#else

#include <Vc/datapar>

namespace pika { namespace parallel { namespace traits {
    ///////////////////////////////////////////////////////////////////////
    template <typename T, typename Abi>
    PIKA_HOST_DEVICE PIKA_FORCEINLINE std::size_t count_bits(
        Vc::mask<T, Abi> const& mask)
    {
        return Vc::popcount(mask);
    }
}}}    // namespace pika::parallel::traits

#endif    // Vc_IS_VERSION_1

#endif
