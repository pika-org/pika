//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/config/compiler_specific.hpp>
#include <pika/exception.hpp>
#include <pika/execution.hpp>
#include <pika/execution_base/tests/algorithm_test_utils.hpp>
#include <pika/init.hpp>
#include <pika/mpi.hpp>
#include <pika/testing.hpp>

#include <atomic>
#include <cstdlib>
#include <mpi.h>
#include <string>
#include <utility>
#include <vector>

namespace ex = pika::execution::experimental;
namespace mpi = pika::mpi::experimental;
namespace tt = pika::this_thread::experimental;

// -----------------------------------------------------------------
// This overload is only used to check dispatching. It is not a useful implementation.
template <typename T>
auto tag_invoke(mpi::transform_mpi_t, custom_type<T>& c)
{
    c.tag_invoke_overload_called = true;
    return mpi::transform_mpi(ex::just(&c.x, 1, MPI_INT, 0, MPI_COMM_WORLD), MPI_Ibcast);
}

// -----------------------------------------------------------------
// These tests are in a separate function so that we can annotate the whole thing with
// disabled address sanitizer to work around (temporarily hide) unresolved stack corruption reports
PIKA_NO_SANITIZE_ADDRESS void test_exception_handler_code(MPI_Comm comm, MPI_Datatype datatype)
{
    // Failure path
    {
        // Exception with error sender
        bool exception_thrown = false;
        try
        {
            tt::sync_wait(mpi::transform_mpi(
                error_sender<int*, int, MPI_Datatype, int, MPI_Comm>{}, MPI_Ibcast));
            PIKA_TEST(false);
        }
        catch (std::runtime_error const& e)
        {
            PIKA_TEST_EQ(std::string(e.what()), std::string("error"));
            exception_thrown = true;
        }
        PIKA_TEST(exception_thrown);
    }

    {
        // Exception with const reference error sencder
        bool exception_thrown = false;
        try
        {
            tt::sync_wait(mpi::transform_mpi(
                const_reference_error_sender{}, [](MPI_Request*) { PIKA_TEST(false); }));
        }
        catch (std::runtime_error const& e)
        {
            PIKA_TEST_EQ(std::string(e.what()), std::string("error"));
            exception_thrown = true;
        }
        PIKA_TEST(exception_thrown);
    }

    {
        // Exception in the lambda
        bool exception_thrown = false;
        int data = 0, count = 1;
        auto s = mpi::transform_mpi(ex::just(&data, count, datatype, 0, comm),
            [](int* data, int count, MPI_Datatype datatype, int i, MPI_Comm comm,
                MPI_Request* request) {
                MPI_Ibcast(data, count, datatype, i, comm, request);
                throw std::runtime_error("error in lambda");
            });
        try
        {
            tt::sync_wait(PIKA_MOVE(s));
        }
        catch (std::runtime_error const& e)
        {
            PIKA_TEST_EQ(std::string(e.what()), std::string("error in lambda"));
            exception_thrown = true;
        }
        PIKA_TEST(exception_thrown);
        // Necessary to avoid a seg fault caused by MPI data going out of scope
        // too early when an exception occurred outside of MPI
        MPI_Barrier(comm);
    }

    {
        // Exception thrown through pika custom error handler that throws
        int *data = nullptr, count = 0;
        bool exception_thrown = false;
        try
        {
            tt::sync_wait(
                mpi::transform_mpi(ex::just(data, count, MPI_DATATYPE_NULL, -1, comm), MPI_Ibcast));
            PIKA_TEST(false);
        }
        catch (pika::exception const& e)
        {
            // Different MPI implementations print different error messages.
            bool err_ok = (e.get_error() == pika::error::bad_function_call);
            PIKA_TEST_MSG(err_ok, "Returned error code was not in expected list");
            std::vector<std::string> err_msgs = {"null datatype", "MPI_DATATYPE_NULL",
                "Invalid datatype", "MPI_ERR_TYPE", "Invalid root"};
            bool msg_ok = false;
            for (auto const& msg : err_msgs)
            {
                msg_ok |= (std::string(e.what()).find(msg) != std::string::npos);
            }
            PIKA_TEST_MSG(msg_ok, "Error message did not contain expected string");
            exception_thrown = true;
        }
        PIKA_TEST(exception_thrown);
    }
}

// -----------------------------------------------------------------
PIKA_NO_SANITIZE_ADDRESS void test_exception_no_handler(MPI_Comm comm)
{
    // Use the default error handler MPI_ERRORS_ARE_FATAL
    mpi::enable_polling enable_polling_no_errhandler;
    {
        // Exception thrown based on the returned error code
        int *data = nullptr, count = 0;
        bool exception_thrown = false;
        try
        {
            tt::sync_wait(
                mpi::transform_mpi(ex::just(data, count, MPI_DATATYPE_NULL, -1, comm), MPI_Ibcast));
            PIKA_TEST(false);
        }
        catch (std::runtime_error const&)
        {
            exception_thrown = true;
        }
        PIKA_TEST(exception_thrown);
    }
}

// -----------------------------------------------------------------
int pika_main()
{
    int size, rank;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);

    PIKA_TEST_MSG(size > 1, "This test requires N>1 mpi ranks");

    MPI_Datatype datatype = MPI_INT;

    {
        {
            // Use the custom error handler which throws exceptions on mpi errors
            mpi::enable_polling enable_polling(mpi::exception_mode::install_handler);
            // Success path
            {
                // MPI function pointer
                int data = 0, count = 1;
                if (rank == 0) { data = 42; }
                auto s = mpi::transform_mpi(ex::just(&data, count, datatype, 0, comm), MPI_Ibcast);
                tt::sync_wait(PIKA_MOVE(s));
                PIKA_TEST_EQ(data, 42);
            }

            {
                // Lambda
                int data = 0, count = 1;
                if (rank == 0) { data = 42; }
                auto s = mpi::transform_mpi(ex::just(&data, count, datatype, 0, comm),
                    [](int* data, int count, MPI_Datatype datatype, int i, MPI_Comm comm,
                        MPI_Request* request) {
                        return MPI_Ibcast(data, count, datatype, i, comm, request);
                    });
                tt::sync_wait(PIKA_MOVE(s));
                PIKA_TEST_EQ(data, 42);
            }

            {
                // Lambda returning void
                int data = 0, count = 1;
                if (rank == 0) { data = 42; }
                auto s = mpi::transform_mpi(ex::just(&data, count, datatype, 0, comm),
                    [](int* data, int count, MPI_Datatype datatype, int i, MPI_Comm comm,
                        MPI_Request* request) {
                        MPI_Ibcast(data, count, datatype, i, comm, request);
                    });
                tt::sync_wait(PIKA_MOVE(s));
                PIKA_TEST_EQ(data, 42);
            }

            // Values passed to transform_mpi should be kept alive by transform_mpi itself
            {
                int count = 1 << 20;
                auto s = ex::just(std::vector<int>{count, 0}, datatype, 0, comm) |
                    ex::drop_operation_state() |
                    mpi::transform_mpi([](auto& data, MPI_Datatype datatype, int i, MPI_Comm comm,
                                           MPI_Request* request) {
                        MPI_Ibcast(data.data(), data.size(), datatype, i, comm, request);
                    });
                tt::sync_wait(PIKA_MOVE(s));
            }

            {
                auto s = ex::just(custom_type_non_default_constructible_non_copyable{42}, datatype,
                             0, comm) |
                    ex::drop_operation_state() |
                    mpi::transform_mpi([](auto& data, MPI_Datatype datatype, int i, MPI_Comm comm,
                                           MPI_Request* request) {
                        MPI_Ibcast(&data.x, 1, datatype, i, comm, request);
                    });
                tt::sync_wait(PIKA_MOVE(s));
            }

            // transform_mpi should be able to handle reference types (by copying
            // them to the operation state)
            {
                int data = 0, count = 1;
                if (rank == 0) { data = 42; }
                auto s = mpi::transform_mpi(const_reference_sender<int>{count},
                    [&](int& count_transform_mpi, MPI_Request* request) {
                        PIKA_TEST(&count_transform_mpi != &count);
                        MPI_Ibcast(&data, count_transform_mpi, datatype, 0, comm, request);
                    });
                tt::sync_wait(PIKA_MOVE(s));
                PIKA_TEST_EQ(data, 42);
            }

            {
                // tag_invoke overload
                std::atomic<bool> tag_invoke_overload_called{false};
                custom_type<int> c{tag_invoke_overload_called, 0};
                if (rank == 0) { c.x = 3; }
                auto s = mpi::transform_mpi(c);
                tt::sync_wait(std::move(s));
                if (rank == 0) { PIKA_TEST_EQ(c.x, 3); }
                PIKA_TEST(tag_invoke_overload_called);
            }

            // Operator| overload
            {
                // MPI function pointer
                int data = 0, count = 1;
                if (rank == 0) { data = 42; }
                tt::sync_wait(
                    ex::just(&data, count, datatype, 0, comm) | mpi::transform_mpi(MPI_Ibcast));
                PIKA_TEST_EQ(data, 42);
            }

            // MPICH does not support throwing exceptions from error handlers (lock issues, see
            // https://github.com/pmodels/mpich/issues/7187)
#if !defined(MPICH)
            test_exception_handler_code(comm, datatype);

        }    // let the user polling go out of scope
        test_exception_no_handler(comm);
#else
        }
#endif
    }

    test_adl_isolation(mpi::transform_mpi(my_namespace::my_sender{}, [](MPI_Request*) {}));

    pika::finalize();
    return EXIT_SUCCESS;
}

//----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    int provided;
    int preferred = MPI_THREAD_MULTIPLE;
    MPI_Init_thread(&argc, &argv, preferred, &provided);
    PIKA_TEST_EQ(provided, preferred);

    // Start runtime and collect runtime exit status
    auto result = pika::init(pika_main, argc, argv);
    PIKA_TEST_EQ(result, 0);

    MPI_Finalize();
    return result;
}
