//  Copyright (c) 2019 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/assert.hpp>
#include <pika/execution.hpp>
#if __has_include(<pika/executors/thread_pool_scheduler_queue_bypass.hpp>)
# include <pika/executors/thread_pool_scheduler_queue_bypass.hpp>
#endif
#include <pika/future.hpp>
#include <pika/init.hpp>
#include <pika/mpi.hpp>
#include <pika/program_options.hpp>
#include <pika/testing.hpp>
//
#include <boost/lockfree/stack.hpp>
//
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

static bool output = false;
static uint32_t mpi_task_transfer = false;
static std::uint32_t mpi_poll_size = 16;
// ------------------------------------------------------------
// a debug level of zero disables messages with a priority>0
// a debug level of N shows messages with priority<N
constexpr int debug_level = 9;
using namespace pika::debug::detail;
template <int Level>
static print_threshold<Level, debug_level> msr_deb("MPI_SR_");

// ------------------------------------------------------------
// caution: message_buffers will be constructed in-place in a buffer
// allocated with a larger size (set by message-bytes option)
struct message_buffer
{
    std::uint64_t token_val_;
    std::uint64_t size_;
    //
    std::uint64_t size()
    {
        return size_;
    }
    //
    message_buffer(std::uint64_t msize)
    {
        token_val_ = 65535;
        size_ = msize;
    }
};

// ------------------------------------------------------------
inline std::uint32_t next_rank(std::uint32_t rank, std::uint32_t size)
{
    return (rank + 1) % size;
}

inline std::uint32_t prev_rank(std::uint32_t rank, std::uint32_t size)
{
    return (rank + size - 1) % size;
}

// ------------------------------------------------------------
// when messages arrive/complete in random order, it might be the
// case that message N + x completes long before message N, so
// we don't reuse tags immmediately and instead offset them by
// a multiple to ensure we don't have two with the same id in flight
// at the same time
const int safety_factor = 5;

inline std::uint32_t tag_no(
    std::uint32_t rank, std::uint32_t iteration, std::uint32_t in_flight)
{
    return (rank * (in_flight * safety_factor)) +
        (iteration % (in_flight * safety_factor));
}

// ------------------------------------------------------------
enum class msg_type : std::uint32_t
{
    send = 0,
    recv = 1
};

// ------------------------------------------------------------
// utility function to print out info after send/recv completes
void msg_info(std::uint32_t rank, std::uint32_t size, msg_type mtype,
    message_buffer* buf, unsigned tag, const char* xmsg = nullptr)
{
    if (output)
    {
        using namespace pika::debug::detail;
        int other = (mtype == msg_type::send) ? next_rank(rank, size) :
                                                prev_rank(rank, size);
        const char* msg = (mtype == msg_type::send) ? "send" : "recv";
        std::stringstream temp;
        temp << "R " << dec<3>(rank) << "/" << dec<3>(size);
        msr_deb<0>.debug(str<>(msg), temp.str(), "token",
            dec<3>(buf->token_val_), "to/from", dec<3>(other), "tag",
            dec<3>(tag), xmsg);
    }
}

// ------------------------------------------------------------
// message buffers get reused from a stack
boost::lockfree::stack<message_buffer*, boost::lockfree::fixed_sized<false>>
    message_buffers(1024);

message_buffer* get_msg_buffer(std::uint64_t size)
{
    message_buffer* buffer;
    if (message_buffers.pop(buffer))
    {
        return buffer;
    }
    // allocate the amount of space we want
    void* data = new unsigned char[size];
    // construct out buffer object in that space
    return new (data) message_buffer(size);
}

void release_msg_buffer(message_buffer* buffer)
{
    message_buffers.push(buffer);
}

// ------------------------------------------------------------
// this is called on an pika thread after the runtime starts up
int pika_main(pika::program_options::variables_map& vm)
{
    int rank, size;
    //
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // enable MPI error handler to throw pika::mpi_exception
    pika::mpi::experimental::init(false, "", true);

    // --------------------------
    // if not debugging standalone and comm size < 2 this test should fail
    // it needs to run on N>2 ranks to be useful
    // --------------------------
    if (vm.count("standalone") == 0)
    {
        PIKA_TEST_MSG(size > 1,
            "This test requires N>1 mpi ranks, use --standalone for single "
            "node testing");
    }

    // --------------------------
    // Get user options/flags
    // --------------------------
    const std::uint64_t iterations = vm["iterations"].as<std::uint64_t>();
    output = vm.count("output") != 0;
    mpi_task_transfer = vm["mpi-task-transfer"].as<std::uint32_t>();
    //
    auto in_flight = vm["in-flight-limit"].as<std::uint32_t>();
    mpi::set_max_requests_in_flight(in_flight, mpi::stream_type::user);

    mpi_poll_size = vm["mpi-polling-size"].as<std::uint32_t>();
    mpi::set_max_mpi_polling_size(mpi_poll_size);

    std::uint64_t message_size = vm["message-bytes"].as<std::uint64_t>();

    // --------------------------
    // main scope with polling enabled
    // --------------------------
    {
        // the scope of the user polling must include all uses of transform_mpi
        mpi::enable_user_polling enable_polling;

        // To prevent the application exiting the main scope of mpi polling
        // whilst there are messages in flight, we will count each send/recv and
        // wait until all are done
        // each iteration initiate ring = send + recv
        // each iteration participate in = recv + send (for n_ranks - 1)
        // #nummessages = (2 * iterations) + (size-1)*2*iterations
        std::atomic<std::uint64_t> counter = size * 2 * iterations;

        // start a timer
        pika::chrono::detail::high_resolution_timer t;

        // for each iteration (number of times we ring send)
        for (std::uint64_t i = 0; i < iterations; ++i)
        {
            // for each rank that originates a message
            for (int origin = 0; origin < size; origin++)
            {
                // post a ring receive for each rank at this iteration
                auto buf = get_msg_buffer(message_size);
                std::uint64_t tag = tag_no(origin, i, in_flight);
                auto snd1 = ex::just(&*buf, message_size, MPI_UNSIGNED_CHAR,
                                prev_rank(rank, size), tag, MPI_COMM_WORLD) |
                    mpi::transform_mpi(MPI_Irecv, mpi::stream_type::receive);
                // rx msg came from us, and has completed the ring
                if (origin == rank)
                {
                    auto recv_snd = snd1 | ex::then([=, &counter](int /*res*/) {
                        // output info
                        msg_info(rank, size, msg_type::recv, buf, tag);
                        counter--;
                        if (output)
                        {
                            using namespace pika::debug::detail;
                            msr_deb<0>.debug(str<>("Complete"), "Rank",
                                dec<3>(rank), "of", dec<3>(size), "Recv token",
                                dec<3>(buf->token_val_), "tag", dec<3>(tag),
                                "counter", dec<3>(counter));
                        }
                        release_msg_buffer(buf);
                    });
                    msg_info(rank, size, msg_type::recv, buf, tag, "post");
                    ex::start_detached(std::move(recv_snd));
                }
                // msgs from other ranks get forwarded to the next in the ring
                else
                {
                    // to prevent continuations running on a polling thread
                    // transfer explicitly to a new pika task
                    ex::any_sender<int> as1;
                    if (mpi_task_transfer == 0)
                    {
                        as1 = snd1;
                    }
                    else if (mpi_task_transfer == 1)
                    {
                        as1 = snd1 |
                            ex::transfer(
                                ex::with_priority(ex::thread_pool_scheduler{},
                                    pika::execution::thread_priority::high));
                    }
                    if (mpi_task_transfer == 2)
                    {
                        as1 = snd1 |
                            ex::transfer(ex::with_priority(
                                ex::thread_pool_scheduler_queue_bypass{},
                                pika::execution::thread_priority::high));
                    }
                    auto recv_snd = as1 | ex::then([=, &counter](int /*res*/) {
                        // output info
                        msg_info(rank, size, msg_type::recv, buf, tag);
                        counter--;

                        // decrement the token we received
                        --buf->token_val_;
                        if (buf->token_val_ !=
                            (size - std::uint64_t(rank + size - origin)) % size)
                        {
                            using namespace pika::debug::detail;
                            msr_deb<0>.debug(str<>("Recv"), "Rank",
                                dec<3>(rank), "of", dec<3>(size), "Recv token",
                                dec<3>(buf->token_val_), "from rank",
                                dec<3>(prev_rank(rank, size)), "tag",
                                dec<3>(tag));
                        }
                        assert(buf->token_val_ ==
                            (size - std::uint64_t(rank + size - origin)) %
                                size);

                        auto snd1 =
                            ex::just(&*buf, message_size, MPI_UNSIGNED_CHAR,
                                next_rank(rank, size), tag, MPI_COMM_WORLD) |
                            mpi::transform_mpi(
                                MPI_Isend, mpi::stream_type::send);

                        auto send_snd = snd1 |
                            // ex::transfer(ex::thread_pool_scheduler_queue_bypass{}) |
                            ex::then([=, &counter](int /*res*/) {
                                // output info
                                msg_info(rank, size, msg_type::send, buf, tag);
                                counter--;
                                release_msg_buffer(buf);
                                //                                if (output) msr_deb<0>.debug(str<>("Yielding after forwarding"));
                                //                                pika::this_thread::yield();
                                //                                if (output) msr_deb<0>.debug(str<>("Done Yielding after forwarding"));
                            });
                        msg_info(rank, size, msg_type::send, buf, tag, "post");
                        ex::start_detached(std::move(send_snd));
                    });
                    msg_info(rank, size, msg_type::recv, buf, tag, "post");
                    ex::start_detached(std::move(recv_snd));
                }
            }

            // start the ring (first message) message for this iteration/rank
            auto buf = get_msg_buffer(message_size);
            buf->token_val_ = size;
            std::uint64_t tag = tag_no(rank, i, in_flight);
            auto send_snd =
#if SPAWN_AS_NEW_TASK
                ex::transfer_just(ex::thread_pool_scheduler{}, &*buf,
                    message_size, MPI_UNSIGNED_CHAR, next_rank(rank, size), tag,
                    MPI_COMM_WORLD) |
#else
                ex::just(&*buf, message_size, MPI_UNSIGNED_CHAR,
                    next_rank(rank, size), tag, MPI_COMM_WORLD) |
#endif
                mpi::transform_mpi(MPI_Isend, mpi::stream_type::user) |
                ex::then([=, &counter](int /*result*/) {
                    // output info
                    msg_info(rank, size, msg_type::send, buf, tag);
                    counter--;
                    release_msg_buffer(buf);
                });
            msg_info(rank, size, msg_type::send, buf, tag, "post");
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

        double elapsed = t.elapsed();

        std::cout << "Rank " << rank << " reached end of test " << counter
                  << std::endl;

        if (rank == 0)
        {
            // a complete set of results formatted for plotting
            std::stringstream temp;
            char const* msg = "CSVData, "
                              "{1}, in_flight, {2}, ranks, {3}, threads, {4}, "
                              "iterations, {5}, task_transfer, {6}, "
                              "message-size, {7}, polling-size, {8}, time";
            pika::util::format_to(temp, msg, in_flight, size,
                pika::get_num_worker_threads(), iterations, mpi_task_transfer,
                message_size, mpi_poll_size, elapsed)
                << std::endl;
            std::cout << temp.str();
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
    cmdline.add_options()("iterations",
        value<std::uint64_t>()->default_value(5000),
        "number of iterations to test");

    cmdline.add_options()("in-flight-limit",
        pika::program_options::value<std::uint32_t>()->default_value(
            mpi::get_max_requests_in_flight()),
        "Apply a limit to the number of messages in flight.");

    cmdline.add_options()("mpi-polling-size",
        pika::program_options::value<std::uint32_t>()->default_value(16),
        "The maximum number of mpi request completions to handle per poll.");

    cmdline.add_options()("message-bytes",
        pika::program_options::value<std::uint64_t>()->default_value(64),
        "Specify the buffer size to use for messages (min 16).");

    cmdline.add_options()("mpi-task-transfer",
        pika::program_options::value<std::uint32_t>()->default_value(1),
        "0: poll callbacks on raw thread,\n"
        "1: poll callbacks on scheduler,\n"
        "2: poll callbacks on scheduler_bypass");

    cmdline.add_options()("output",
        "Display messages during test");

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
