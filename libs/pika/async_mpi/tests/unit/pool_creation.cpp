//  Copyright (c) 2024 ETH Zurich
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

/// The purpose of this test is to ensure that a manually created pool,
/// can still be used even though by default, we can now let pika create an mpi pool for us
/// This test does one auto create and two manual creates to check if all works without fail
/// note that it is expected that this test will print warnings when a pool is created
/// but not actually needed by the completion mode

static const std::string random_pool_name1 = "abcd12345qwerty";
static const std::string random_pool_name2 = "ta-daa-500";

namespace ex = pika::execution::experimental;
namespace mpi = pika::mpi::experimental;
namespace tt = pika::this_thread::experimental;

// -----------------------------------------------------------------
int pika_main(const std::string& pool_name)
{
    int size, rank;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);

    MPI_Datatype datatype = MPI_INT;

    {
        // Register polling on our custom pool
        mpi::enable_polling enable_polling(mpi::exception_mode::install_handler, pool_name);
        // Success path
        {
            // MPI function pointer
            int data = 0, count = 1;
            if (rank == 0) { data = 42; }
            auto s = mpi::transform_mpi(ex::just(&data, count, datatype, 0, comm), MPI_Ibcast);
            tt::sync_wait(PIKA_MOVE(s));
            PIKA_TEST_EQ(data, 42);
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

    }    // let the user polling go out of scope

    {
        // Register polling on default pool
        mpi::enable_polling enable_polling(mpi::exception_mode::install_handler,
            pika::resource::get_partitioner().get_default_pool_name());
        // Success path
        {
            // MPI function pointer
            int data = 0, count = 1;
            if (rank == 0) { data = 42; }
            auto s = mpi::transform_mpi(ex::just(&data, count, datatype, 0, comm), MPI_Ibcast);
            tt::sync_wait(PIKA_MOVE(s));
            PIKA_TEST_EQ(data, 42);
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
    }    // let the user polling go out of scope

    pika::finalize();
    return EXIT_SUCCESS;
}

//----------------------------------------------------------------------------
void init_resource_partitioner_manual(
    pika::resource::partitioner& rp, pika::program_options::variables_map const&)
{
    // Disable idle backoff on our custom pool
    using pika::threads::scheduler_mode;
    auto mode = scheduler_mode::default_mode;
    mode = scheduler_mode(mode & ~scheduler_mode::enable_idle_backoff);

    // Create a thread pool with a single core that we will use for communication related tasks
    rp.create_thread_pool(
        random_pool_name1, pika::resource::scheduling_policy::local_priority_fifo, mode);
    rp.add_resource(rp.sockets()[0].cores()[0].pus()[0], random_pool_name1);
}

//----------------------------------------------------------------------------
void init_resource_partitioner_mpi(
    pika::resource::partitioner& rp, pika::program_options::variables_map const&)
{
    pika::mpi::experimental::detail::create_pool(
        rp, random_pool_name2, pika::resource::polling_pool_creation_mode::mode_force_create);
}

//----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    // ---------------------------------------
    // Let pika create a pool for mpi using default settings
    pika::init_params init_args1;
    init_args1.pool_creation_mode = pika::resource::polling_pool_creation_mode::mode_pika_decides;
    // Start runtime and collect runtime exit status
    auto result1 = pika::init([&]() { return pika_main(""); }, argc, argv, init_args1);
    PIKA_TEST_EQ(result1, 0);

    // ---------------------------------------
    // test if we can create and initialize an mpi pool ourselves
    // set runtime initialization callback to manually create our thread pool
    pika::init_params init_args2;
    init_args2.rp_callback = &init_resource_partitioner_manual;
    // Start runtime and collect runtime exit status
    auto result2 =
        pika::init([&]() { return pika_main(random_pool_name1); }, argc, argv, init_args2);
    PIKA_TEST_EQ(result2, 0);

    // ---------------------------------------
    // test if we can create and initialize an mpi pool using mpi::create_pool
    // set runtime initialization callback to create our thread pool using mpi internals
    pika::init_params init_args3;
    init_args3.rp_callback = &init_resource_partitioner_mpi;
    // Start runtime and collect runtime exit status
    auto result3 =
        pika::init([&]() { return pika_main(random_pool_name2); }, argc, argv, init_args3);
    PIKA_TEST_EQ(result3, 0);

    MPI_Finalize();
    return result1 && result2 && result3;
}
