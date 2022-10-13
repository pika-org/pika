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

/*
 * This test exercises the MPI sender/receiver capabilities
 * Every rank sends a message to the next rank counting incrementally
 * (modulus the total number of ranks)
 * and the message is forwarded on to the next one, until it completes
 * a ring of all ranks and returns to it's origin.
 *
 * Thus rank 0, sends a message to rank 1, which is forwarded to 2, then 3
 * etc until it returns to 0. The initial message is tagged according to
 * the start rank and the iteration number, to avoid mismatching receives
 * when many messages are in flight at any given moment.
 *
 * example invocation
 * mpiexec -n 2 -report-bindings --oversubscribe --bind-to core --map-by node:PE=4
 * bin/mpi_ring_async_sender_receiver_test --pika:print-bind
 * --iterations=20 --in-flight-limit=8 --output
*/

using pika::program_options::options_description;
using pika::program_options::value;
using pika::program_options::variables_map;

namespace ex = pika::execution::experimental;
namespace mpi = pika::mpi::experimental;
namespace tt = pika::this_thread::experimental;
namespace deb = pika::debug;

static bool output = true;

// ------------------------------------------------------------
// a debug level of zero disables messages with a priority>0
// a debug level of N shows messages with priority<N
constexpr int debug_level = 9;
template <int Level>
static pika::debug::detail::print_threshold<Level, debug_level> msr_deb(
    "MPI_SR_");
// ------------------------------------------------------------

struct message_item
{
    std::array<std::uint64_t, 1024> databuf;
    //
    message_item(std::uint64_t val = 0)
    {
        std::fill(databuf.begin(), databuf.end(), val);
    }
};

// ------------------------------------------------------------
void rmsg_info(int rank, int size, int from, message_item& token, unsigned tag)
{
    // to reduce string corruption on stdout from multiple threads
    // writing simultaneously, we use a stringstream as a buffer
    if (output)
    {
        using namespace pika::debug::detail;
        msr_deb<0>.debug(str<>("Recv"), "Rank", dec<3>(rank), "of",
            dec<3>(size), "Recv token", dec<3>(token.databuf[0]), "from rank",
            dec<3>(from), "tag", dec<3>(tag));
    }
}

void smsg_info(int rank, int size, int to, message_item& token, unsigned tag)
{
    if (output)
    {
        using namespace pika::debug::detail;
        msr_deb<1>.debug(str<>("Send"), "Rank", dec<3>(rank), "of",
            dec<3>(size), "Send token", dec<3>(token.databuf[0]), "to   rank",
            dec<3>(to), "tag", dec<3>(tag));
    }
}

inline std::uint32_t next_rank(std::uint32_t rank, std::uint32_t size)
{
    return (rank + 1) % size;
}

inline std::uint32_t prev_rank(std::uint32_t rank, std::uint32_t size)
{
    return (rank - 1) % size;
}

inline std::uint32_t tag_no(
    std::uint32_t rank, std::uint32_t iteration, std::uint32_t n_iterations)
{
    return (rank * n_iterations) + iteration;
}

inline std::uint32_t recv_tag(std::uint32_t rank, std::uint32_t size,
    std::uint32_t iteration, std::uint32_t n_iterations)
{
    return (prev_rank(rank, size) * n_iterations) + iteration;
}

// this is called on an pika thread after the runtime starts up
int pika_main(pika::program_options::variables_map& vm)
{
    int rank, size;
    //
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // --------------------------
    // if not debugging standalone and comm size < 2 this test should fail
    // it needs to run on N>2 ranks to be useful
    // --------------------------
    if (vm.count("standalone") == 0)
    {
        PIKA_TEST_MSG(size > 1, "This test requires N>1 mpi ranks");
    }

    // --------------------------
    // Get user options/flags
    // --------------------------
    const std::uint64_t iterations = vm["iterations"].as<std::uint64_t>();
    output = vm.count("output") != 0;
    //
    auto throttling = vm["in-flight-limit"].as<std::uint32_t>();
    mpi::set_max_requests_in_flight(throttling, mpi::stream_type::user);

    // --------------------------
    // main scope with polling enabled
    // --------------------------
    {
        // the scope of the user polling must include all uses of transform_mpi
        mpi::enable_user_polling enable_polling;

        // we will send "tokens" : random data (with counters set by default)
        std::vector<message_item> tokens(iterations * size, size);
        for (auto& t : tokens)
        {
            t.databuf[0] = size;
        }

        pika::chrono::detail::high_resolution_timer t;

        // Each rank starts N rings, where N=iterations
        std::atomic<std::uint64_t> counter = iterations;

        // for each iteration (number of times we ring send)
        for (std::uint64_t i = 0; i < iterations; ++i)
        {
            // for each rank that originates a message
            for (int origin = 0; origin < size; origin++)
            {
                // post a ring receive for each rank at this iteration
                std::uint64_t tag = tag_no(origin, i, iterations);
                auto recv_snd = ex::just(&tokens[tag], sizeof(message_item),
                                    MPI_UNSIGNED_CHAR, prev_rank(rank, size),
                                    tag, MPI_COMM_WORLD) |
                    mpi::transform_mpi(MPI_Irecv, mpi::stream_type::receive) |
                    ex::then([rank, size, tag, origin, &tokens, &counter](
                                 int /*result*/) {
                        // output info
                        rmsg_info(rank, size, prev_rank(rank, size),
                            tokens[tag], tag);
                        // decrement the token
                        --tokens[tag].databuf[0];
                        // if this token came from us and has been all the way round the ring, just exit
                        if (origin == rank)
                        {
                            --counter;
                            if (output)
                            {
                                using namespace pika::debug::detail;
                                msr_deb<0>.debug(str<>("Complete"), "Rank",
                                    dec<3>(rank), "of", dec<3>(size),
                                    "Recv token",
                                    dec<3>(tokens[tag].databuf[0]), "tag",
                                    dec<3>(tag), "counter", dec<3>(counter));
                            }
                        }
                        // if this token is from another rank, then forward it on to the right
                        else
                        {
                            auto send_snd =
                                ex::just(&tokens[tag], sizeof(message_item),
                                    MPI_UNSIGNED_CHAR, next_rank(rank, size),
                                    tag, MPI_COMM_WORLD) |
                                mpi::transform_mpi(
                                    MPI_Isend, mpi::stream_type::send) |
                                ex::then([rank, size, tag, &tokens](
                                             int /*result*/) {
                                    // output info
                                    smsg_info(rank, size, next_rank(rank, size),
                                        tokens[tag], tag);
                                });
                            ex::start_detached(std::move(send_snd));
                        }
                    });
                ex::start_detached(std::move(recv_snd));
            }

            // start the ring (first message) message for this iteration/rank
            std::uint64_t tag = tag_no(rank, i, iterations);
            auto send_snd =
                ex::just(&tokens[tag], sizeof(message_item), MPI_UNSIGNED_CHAR,
                    next_rank(rank, size), tag, MPI_COMM_WORLD) |
                mpi::transform_mpi(MPI_Isend, mpi::stream_type::user) |
                ex::then([rank, size, tag, &tokens](int /*result*/) {
                    // output info
                    smsg_info(
                        rank, size, next_rank(rank, size), tokens[tag], tag);
                });
            ex::start_detached(std::move(send_snd));
        }

        // don't exit until all messages are drained
        while (counter > 0)
        {
            pika::this_thread::yield();
        }
        if (output)
        {
            using namespace pika::debug::detail;
            msr_deb<0>.debug(str<>("User Messages"), "Rank", dec<3>(rank), "of",
                dec<3>(size),
                dec<3>(
                    mpi::get_num_requests_in_flight(mpi::stream_type::user)));
        }

        // the user queue should always be empty by now since our counter tracks it
        PIKA_ASSERT(
            mpi::get_num_requests_in_flight(mpi::stream_type::user) == 0);

        // don't exit until messages that are still in flight are drained
        while (mpi::get_num_requests_in_flight(mpi::stream_type::send) +
                mpi::get_num_requests_in_flight(mpi::stream_type::receive) >
            0)
        {
            pika::this_thread::yield();
        }

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
            "number of iterations to test");

    cmdline.add_options()("in-flight-limit",
        pika::program_options::value<std::uint32_t>()->default_value(
            mpi::get_max_requests_in_flight()),
        "Apply a limit to the number of messages in flight.");

    cmdline.add_options()("output", "Display messages during test");

    cmdline.add_options()("standalone",
                          "Allow test to run with a single rank (debugging)");

    // clang-format on

    // Initialize and run pika.
    pika::init_params init_args;
    init_args.desc_cmdline = cmdline;

    auto result = pika::init(pika_main, argc, argv, init_args);
    PIKA_TEST_EQ(result, 0);

    // Finalize MPI
    MPI_Finalize();

    return result;
}
