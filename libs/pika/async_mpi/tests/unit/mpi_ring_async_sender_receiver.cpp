//  Copyright (c) 2019 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/assert.hpp>
#include <pika/execution.hpp>
#include <pika/future.hpp>
#include <pika/init.hpp>
#include <pika/mpi.hpp>
#include <pika/program_options.hpp>
#include <pika/testing.hpp>

#include <array>
#include <atomic>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <mpi.h>

// This test send a message from rank 0 to rank 1 and from R1->R2 in a ring
// until the last rank which sends it back to R0.and completes an iteration
//
// For benchmarking the test has been extended to allow many iterations
// but unfortunately, if we prepost too many receives, MPI has problems
// so we do 1000 iterations per main loop and another loop around that.

using pika::program_options::options_description;
using pika::program_options::value;
using pika::program_options::variables_map;

namespace ex = pika::execution::experimental;
namespace mpi = pika::mpi::experimental;
namespace tt = pika::this_thread::experimental;
namespace deb = pika::debug;

static bool output = true;

void msg_recv(int rank, int size, int /*to*/, int from, int token, unsigned tag)
{
    // to reduce string corruption on stdout from multiple threads
    // writing simultaneously, we use a stringstream as a buffer
    if (output)
    {
        std::ostringstream temp;
        temp << "Rank " << std::setfill(' ') << std::setw(3) << rank << " of "
             << std::setfill(' ') << std::setw(3) << size << " Recv token "
             << std::setfill(' ') << std::setw(3) << token << " from rank "
             << std::setfill(' ') << std::setw(3) << from << " tag "
             << std::setfill(' ') << std::setw(3) << tag;
        std::cout << temp.str() << std::endl;
    }
}

void msg_send(int rank, int size, int to, int /*from*/, int token, unsigned tag)
{
    if (output)
    {
        std::ostringstream temp;
        temp << "Rank " << std::setfill(' ') << std::setw(3) << rank << " of "
             << std::setfill(' ') << std::setw(3) << size << " Sent token "
             << std::setfill(' ') << std::setw(3) << token << " to   rank "
             << std::setfill(' ') << std::setw(3) << to << " tag "
             << std::setfill(' ') << std::setw(3) << tag;
        std::cout << temp.str() << std::endl;
    }
}

// this is called on an pika thread after the runtime starts up
int pika_main(pika::program_options::variables_map& vm)
{
    int rank, size;
    //
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // if comm size < 2 this test should fail
    // it needs to run on N>2 ranks to be useful
    PIKA_TEST_MSG(size > 1, "This test requires N>1 mpi ranks");

    const std::uint64_t iterations = vm["iterations"].as<std::uint64_t>();
    //
    output = vm.count("output") != 0;

    if (rank == 0 && output)
    {
        std::cout << "Rank " << std::setfill(' ') << std::setw(3) << rank
                  << " of " << std::setfill(' ') << std::setw(3) << size
                  << std::endl;
    }

    {
        size_t throttling =
            pika::mpi::experimental::get_max_requests_in_flight();
        if (throttling == size_t(-1))
        {
            pika::mpi::experimental::set_max_requests_in_flight(512);
        }

        // this needs to scope all uses of pika::mpi::experimental::executor
        pika::mpi::experimental::enable_user_polling enable_polling;

        // Ring send/recv around N ranks
        // Rank 0      : Send then Recv
        // Rank 1->N-1 : Recv then Send

        std::vector<int> tokens(iterations, -1);

        pika::chrono::detail::high_resolution_timer t;

        const std::uint64_t inner_iterations = 100;
        const std::uint64_t outer_iterations = iterations / inner_iterations;
        std::atomic<std::uint64_t> counter;

        std::uint64_t tag;
        for (std::uint64_t j = 0; (j != outer_iterations); ++j)
        {
            counter = inner_iterations;
            for (std::uint64_t i = 0; (i != inner_iterations); ++i)
            {
                tag = i + j * inner_iterations;
                tokens[tag] = (rank == 0) ? 1 : -1;
                int rank_from = (size + rank - 1) % size;
                int rank_to = (rank + 1) % size;

                // all ranks pre-post a receive, but when rank-0 receives it, we're done
                if (rank == 0)
                {
                    auto snd = ex::just(&tokens[tag], 1, MPI_INT, rank_from,
                                   tag, MPI_COMM_WORLD) |
                        mpi::transform_mpi(MPI_Irecv) |
                        ex::then([=, &tokens, &counter](int /*result*/) {
                            msg_recv(rank, size, rank_to, rank_from,
                                tokens[tag], tag);
                            --counter;
                        });
                    ex::start_detached(std::move(snd));
                }
                // when ranks>0 complete receives, send the message to next rank
                else
                {
                    auto recv_snd = ex::just(&tokens[tag], 1, MPI_INT,
                                        rank_from, tag, MPI_COMM_WORLD) |
                        mpi::transform_mpi(MPI_Irecv) |
                        ex::then([=, &tokens](int /*result*/) {
                            msg_recv(rank, size, rank_to, rank_from,
                                tokens[tag], tag);
                            // increment the token
                            ++tokens[tag];
                        });

                    auto send_snd =
                        ex::when_all(ex::just(&tokens[tag], 1, MPI_INT, rank_to,
                                         tag, MPI_COMM_WORLD),
                            std::move(recv_snd)) |
                        mpi::transform_mpi(MPI_Isend) |
                        ex::then([=, &tokens, &counter](int /*result*/) {
                            msg_send(rank, size, rank_to, rank_from,
                                tokens[tag], tag);
                            // ranks > 0 are done after sending token
                            --counter;
                        });
                    ex::start_detached(std::move(send_snd));
                }

                // rank 0 starts the process with a send
                if (rank == 0)
                {
                    auto snd0 = ex::just(&tokens[tag], 1, MPI_INT, rank_to, tag,
                                    MPI_COMM_WORLD) |
                        mpi::transform_mpi(MPI_Isend) |
                        ex::then([=, &tokens](int /*result*/) {
                            msg_send(rank, size, rank_to, rank_from,
                                tokens[tag], tag);
                        });
                    ex::start_detached(snd0);
                }
            }
            // block until messages are drained
            while (counter > 0)
            {
                pika::this_thread::yield();
            }
        }

        auto tt = pika::mpi::experimental::get_num_requests_in_flight();
        if (tt != 0)
        {
            std::cout << "Rank " << rank << " flight " << tt << " counter "
                      << counter << std::endl;
        }
        PIKA_ASSERT(pika::mpi::experimental::get_num_requests_in_flight() == 0);
        std::cout << "Rank " << rank << " reached end of test " << counter
                  << std::endl;

        if (rank == 0)
        {
            std::cout << "time " << t.elapsed() << std::endl;
        }

        // let the user polling go out of scope
    }
    return pika::finalize();
}

// the normal int main function that is called at startup and runs on an OS
// thread the user must call pika::init to start the pika runtime which
// will execute pika_main on an pika thread
int main(int argc, char* argv[])
{
    // Init MPI
    int provided = MPI_THREAD_MULTIPLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided != MPI_THREAD_MULTIPLE)
    {
        std::cout << "Provided MPI is not : MPI_THREAD_MULTIPLE " << provided
                  << std::endl;
    }

    // Configure application-specific options.
    options_description cmdline("usage: " PIKA_APPLICATION_STRING " [options]");

    // clang-format off
    cmdline.add_options()
        ("iterations",
            value<std::uint64_t>()->default_value(5000),
            "number of iterations to test")

        ("output", "display messages during test");
    // clang-format on

    // Initialize and run pika.
    pika::init_params init_args;
    init_args.desc_cmdline = cmdline;

    auto result = pika::init(pika_main, argc, argv, init_args);

    // Finalize MPI
    MPI_Finalize();

    return result || pika::util::report_errors();
}
