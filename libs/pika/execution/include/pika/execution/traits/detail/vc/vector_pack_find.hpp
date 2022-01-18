//  Copyright (c) 2021 Srinivas Yadav
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>

#if defined(PIKA_HAVE_DATAPAR_VC)
#include <cstddef>

#include <Vc/Vc>
#include <Vc/global.h>

namespace pika { namespace parallel { namespace traits {
    ///////////////////////////////////////////////////////////////////////
    template <typename T, typename Abi>
    PIKA_HOST_DEVICE PIKA_FORCEINLINE int find_first_of(
        Vc::Mask<T, Abi> const& msk)
    {
        if (Vc::any_of(msk))
        {
            return msk.firstOne();
        }
        return -1;
    }
}}}    // namespace pika::parallel::traits

#endif
