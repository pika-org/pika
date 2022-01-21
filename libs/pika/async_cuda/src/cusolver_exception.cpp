//  Copyright (c) 2020 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#if defined(PIKA_HAVE_CUDA)
#include <pika/async_cuda/cusolver_exception.hpp>
#include <pika/async_cuda/custom_lapack_api.hpp>
#include <pika/errors/exception.hpp>

#include <string>

namespace pika::cuda::experimental {
    namespace detail {
        const char* cusolver_get_error_string(cusolverStatus_t error)
        {
            switch (error)
            {
            case CUSOLVER_STATUS_SUCCESS:
                return "CUSOLVER_STATUS_SUCCESS";
            case CUSOLVER_STATUS_NOT_INITIALIZED:
                return "CUSOLVER_STATUS_NOT_INITIALIZED";
            case CUSOLVER_STATUS_ALLOC_FAILED:
                return "CUSOLVER_STATUS_ALLOC_FAILED";
            case CUSOLVER_STATUS_INVALID_VALUE:
                return "CUSOLVER_STATUS_INVALID_VALUE";
            case CUSOLVER_STATUS_ARCH_MISMATCH:
                return "CUSOLVER_STATUS_ARCH_MISMATCH";
            case CUSOLVER_STATUS_EXECUTION_FAILED:
                return "CUSOLVER_STATUS_EXECUTION_FAILED";
            case CUSOLVER_STATUS_INTERNAL_ERROR:
                return "CUSOLVER_STATUS_INTERNAL_ERROR";
            case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
                return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
            case CUSOLVER_STATUS_MAPPING_ERROR:
                return "CUSOLVER_STATUS_MAPPING_ERROR";
            case CUSOLVER_STATUS_NOT_SUPPORTED:
                return "CUSOLVER_STATUS_NOT_SUPPORTED";
            case CUSOLVER_STATUS_ZERO_PIVOT:
                return "CUSOLVER_STATUS_ZERO_PIVOT";
            case CUSOLVER_STATUS_INVALID_LICENSE:
                return "CUSOLVER_STATUS_INVALID_LICENSE";
            case CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED:
                return "CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED";
            case CUSOLVER_STATUS_IRS_PARAMS_INVALID:
                return "CUSOLVER_STATUS_IRS_PARAMS_INVALID";
            case CUSOLVER_STATUS_IRS_INTERNAL_ERROR:
                return "CUSOLVER_STATUS_IRS_INTERNAL_ERROR";
            case CUSOLVER_STATUS_IRS_NOT_SUPPORTED:
                return "CUSOLVER_STATUS_IRS_NOT_SUPPORTED";
            case CUSOLVER_STATUS_IRS_OUT_OF_RANGE:
                return "CUSOLVER_STATUS_IRS_OUT_OF_RANGE";
            case CUSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES:
                return "CUSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_"
                       "GMRES";
            case CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED:
                return "CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED";
            case CUSOLVER_STATUS_IRS_PARAMS_INVALID_PREC:
                return "CUSOLVER_STATUS_IRS_PARAMS_INVALID_PREC";
            case CUSOLVER_STATUS_IRS_PARAMS_INVALID_REFINE:
                return "CUSOLVER_STATUS_IRS_PARAMS_INVALID_REFINE";
            case CUSOLVER_STATUS_IRS_PARAMS_INVALID_MAXITER:
                return "CUSOLVER_STATUS_IRS_PARAMS_INVALID_MAXITER";
            case CUSOLVER_STATUS_IRS_INFOS_NOT_DESTROYED:
                return "CUSOLVER_STATUS_IRS_INFOS_NOT_DESTROYED";
            case CUSOLVER_STATUS_IRS_MATRIX_SINGULAR:
                return "CUSOLVER_STATUS_IRS_MATRIX_SINGULAR";
            case CUSOLVER_STATUS_INVALID_WORKSPACE:
                return "CUSOLVER_STATUS_INVALID_WORKSPACE";
            }
            return "<unknown>";
        }
    }    // namespace detail

    cusolver_exception::cusolver_exception(
        const std::string& msg, cusolverStatus_t err)
      : pika::exception(pika::bad_function_call, msg)
      , err_(err)
    {
    }

    cusolverStatus_t cusolver_exception::get_cusolver_errorcode() const noexcept
    {
        return err_;
    }

    cusolverStatus_t check_cusolver_error(cusolverStatus_t err)
    {
        if (err != CUSOLVER_STATUS_SUCCESS)
        {
            auto msg = std::string("cuSOLVER function returned error code ") +
                std::to_string(err) + " (" +
                detail::cusolver_get_error_string(err) + ")";
            throw cusolver_exception(msg, err);
        }
        return err;
    }
}    // namespace pika::cuda::experimental
#endif
