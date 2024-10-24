//  Copyright (c) 2024 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/config/compiler_specific.hpp>
#include <pika/debugging/print.hpp>
#include <pika/exception.hpp>
#include <pika/execution.hpp>
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

static const std::string random_name1 = "abcd12345qwerty";
static const std::string random_name2 = "ta-daa-500";

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
    }    // let the polling go out of scope

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
        random_name1, pika::resource::scheduling_policy::local_priority_fifo, mode);
    rp.add_resource(rp.sockets()[0].cores()[0].pus()[0], random_name1);
}

//----------------------------------------------------------------------------
void init_resource_partitioner_mpi(
    pika::resource::partitioner& rp, pika::program_options::variables_map const&)
{
    pika::mpi::experimental::detail::create_pool(
        rp, random_name2, mpi::polling_pool_creation_mode::mode_force_create);
}

//----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    // -----------------
    // Init MPI
    int provided, preferred = MPI_THREAD_MULTIPLE;
    MPI_Init_thread(&argc, &argv, preferred, &provided);
    PIKA_TEST_EQ(provided, preferred);

    {
        // ---------------------------------------
        // Start runtime and collect runtime exit status, turn on auto pool creation
        pika::init_params init_args;
        init_args.cfg.emplace_back("pika.mpi.enable_pool=" + std::to_string(true));
        int result = pika::init([&]() { return pika_main(""); }, argc, argv, init_args);
        PIKA_TEST_EQ(result, 0);
    }

    {
        // ---------------------------------------
        // test if we can create and initialize an mpi pool ourselves
        // set runtime initialization callback to manually create our thread pool
        pika::init_params init_args;
        init_args.rp_callback = &init_resource_partitioner_manual;
        int result = pika::init([&]() { return pika_main(random_name1); }, argc, argv, init_args);
        PIKA_TEST_EQ(result, 0);
    }

    {
        // ---------------------------------------
        // test if we can create and initialize an mpi pool using mpi::create_pool
        // set runtime initialization callback to create our thread pool using mpi internals
        pika::init_params init_args;
        init_args.rp_callback = &init_resource_partitioner_mpi;
        int result = pika::init([&]() { return pika_main(random_name2); }, argc, argv, init_args);
        PIKA_TEST_EQ(result, 0);
    }

    return MPI_Finalize();
}
