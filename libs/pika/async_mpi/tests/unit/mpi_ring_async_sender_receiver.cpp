//  Copyright (c) 2019 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/assert.hpp>
#include <pika/execution.hpp>
#if __has_include(<pika/executors/thread_pool_scheduler_queue_bypass.hpp>)
#include <pika/executors/thread_pool_scheduler_queue_bypass.hpp>
#endif
#include <pika/future.hpp>
#include <pika/init.hpp>
#include <pika/mpi.hpp>
#include <pika/program_options.hpp>
#include <pika/testing.hpp>
//
#include <boost/lockfree/stack.hpp>
#include <fmt/format.h>
#include <fmt/printf.h>
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
static uint32_t mpi_task_transfer = mpi::get_completion_mode();
static std::uint32_t mpi_poll_size = 16;
std::atomic<std::uint64_t> counter;

// ------------------------------------------------------------
// a debug level of zero disables messages with a priority>0
// a debug level of N shows messages with priority<N
using namespace pika::debug::detail;
template <int Level>
static print_threshold<Level, 1> msr_deb("MPI_SR_");

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
// we don't reuse tags immediately and instead offset them by
// a multiple to ensure we don't have two with the same id in flight
// at the same time
inline std::uint32_t tag_no(std::uint64_t rank, std::uint64_t iteration, std::uint32_t ranks)
{
    std::int64_t tag = (rank + (iteration * ranks)) & 0xffffffff;
    msr_deb<7>.debug(
        str<>("generating tag"), std::uint32_t(tag), "rank/s", rank, ranks, "iteration", iteration);
    return std::uint32_t(tag);
}

// ------------------------------------------------------------
enum class msg_type : std::uint32_t
{
    send = 0,
    recv = 1
};

// ------------------------------------------------------------
// utility function to print out info after send/recv completes
void msg_info(std::uint32_t rank, std::uint32_t size, msg_type mtype, std::uint32_t token, unsigned tag,
    std::uint32_t round, std::uint32_t step, const char* xmsg = nullptr)
{
    if (output)
    {
        using namespace pika::debug::detail;
        int other = (mtype == msg_type::send) ? next_rank(rank, size) : prev_rank(rank, size);
        const char* msg = (mtype == msg_type::send) ? "send" : "recv";
        std::stringstream temp;
        temp << dec<3>(rank) << "/" << dec<3>(size);
        // clang-format off
        msr_deb<1>.debug(str<>(temp.str().c_str())
                         , "token", hex<4>(token)
                         , "<-/->", dec<3>(other)
                         , "tag", dec<3>(tag)
                         , "round", dec<3>(round)
                         , "step", dec<3>(step)
                         , msg
                         , (xmsg == nullptr) ? "" : xmsg
                         , "counter", std::int64_t(counter));
        // clang-format on
    }
}

// ------------------------------------------------------------
// message buffers get reused from a stack
boost::lockfree::stack<message_buffer*, boost::lockfree::fixed_sized<false>> message_buffers(1024);
std::atomic<std::uint64_t> message_buffers_size_{0};

message_buffer* get_msg_buffer(std::uint64_t size)
{
    message_buffer* buffer;
    if (!message_buffers.pop(buffer))
    {
        // allocate the amount of space we want
        void* data = new unsigned char[size];
        // construct out buffer object in that space
        buffer = new (data) message_buffer(size);
    }
    // set initial token to some easy to spot default
    buffer->token_val_ = 0xc0de;
    message_buffers_size_++;
    msr_deb<6>.debug(str<>("message_buffers"), std::uint64_t(message_buffers_size_.load()));
    return buffer;
}

void release_msg_buffer(message_buffer* buffer)
{
    --message_buffers_size_;
    message_buffers.push(buffer);
    msr_deb<6>.debug(str<>("message_buffers"), std::uint64_t(message_buffers_size_.load()));
}

namespace pika::execution {
    template <typename Sender>
    auto async(Sender&& sender)
    {
        namespace ex = pika::execution::experimental;
        auto sched = ex::thread_pool_scheduler{&pika::resource::get_thread_pool("default")};
        auto snd = ex::schedule(sched) |
            ex::then([sender = std::move(sender)]() mutable { ex::start_detached(std::move(sender)); });
        ex::start_detached(std::move(snd));
    }
}    // namespace pika::execution

struct receiver
{
    std::uint64_t rank, orank, size, tag, num_rounds, message_size;
    message_buffer* buf;

    receiver(int rank, int orank, int size, int tag, int num_rounds, int message_size, message_buffer* buf)
      : rank(rank)
      , orank(orank)
      , size(size)
      , tag(tag)
      , num_rounds(num_rounds)
      , message_size(message_size)
      , buf(buf)
    {
    }

    void operator()(int /*res*/)
    {
        // global counter for sanity checking of the test itself
        counter--;

        // increment the token we received
        auto old_token = buf->token_val_++;

        // each message travels around the ring num_rounds times,
        // which round is the message currently on?
        std::uint64_t round = buf->token_val_ / size;
        std::uint64_t step = buf->token_val_ % size;

        // if the message has arrived back at the starting rank
        // and has completed all rounds, it is finished
        if (step == 0 && round == num_rounds)
        {
            msg_info(rank, size, msg_type::recv, buf->token_val_, tag, round, step, "complete");
            release_msg_buffer(buf);
        }
        // if the message (not originating from this rank) needs to be forwarded,
        // but is on its final round, do not post a receive as it won't come back to us
        else if (step != 0 && (round == num_rounds - 1))
        {
            msg_info(rank, size, msg_type::recv, buf->token_val_, tag, round, step, "forward(no rx)");
            // forward the message with the (already) incremented token
            auto tx_snd2 =
                ex::just(&*buf, message_size, MPI_UNSIGNED_CHAR, next_rank(rank, size), tag, MPI_COMM_WORLD) |
                mpi::transform_mpi(MPI_Isend, mpi::stream_type::user_1) |
                ex::then([round, step, buf = buf, tag = tag, rank = rank, size = size](int /*result*/) {
                    counter--;
                    msg_info(rank, size, msg_type::send, buf->token_val_, tag, round, step, "forwardsent");
                    release_msg_buffer(buf);
                });

            // launch the receive for the next msg and launch the sending forward
            msr_deb<6>.debug(str<>("start_detached"), "tx_snd2", step, round);
            pika::execution::async(std::move(tx_snd2));
        }
        else
        {
            // the message needs to be forwarded and we need to post a new
            // receive to handle it when it comes back on the next round
            // we can reuse the same buffer for the send and for the receive,
            // because the send always goes out before the receive comes in
            // even if there is only one rank in the ring
            msg_info(rank, size, msg_type::send, buf->token_val_, tag, round, step, "forward");
            receiver reclambda(rank, orank, size, tag, num_rounds, message_size, buf);

            // prepost receive the recursive lambda will
            // be used to handle it in the same way as this one is
            auto rx_snd2 =
                ex::just(&*buf, message_size, MPI_UNSIGNED_CHAR, prev_rank(rank, size), tag, MPI_COMM_WORLD) |
                mpi::transform_mpi(MPI_Irecv, mpi::stream_type::receive_2) | ex::then(reclambda);

            // forward the message with the (already) incremented token
            auto tx_snd2 =
                ex::just(&*buf, message_size, MPI_UNSIGNED_CHAR, next_rank(rank, size), tag, MPI_COMM_WORLD) |
                mpi::transform_mpi(MPI_Isend, mpi::stream_type::user_1) |
                ex::then([round, step, buf = buf, tag = tag, rank = rank, size = size](int /*result*/) {
                    counter--;
                    msg_info(rank, size, msg_type::send, buf->token_val_, tag, round, step, "forwardsent");
                    //                release_msg_buffer(buf);
                });

            // launch the receive for the next msg and launch the sending forward
            msr_deb<6>.debug(str<>("start_detached"), "rx_snd2", step, round);
            pika::execution::async(std::move(rx_snd2));
            msr_deb<6>.debug(str<>("start_detached"), "tx_snd2", step, round);
            pika::execution::async(std::move(tx_snd2));
        }
    }
};

// ------------------------------------------------------------
int call_mpi_irecv(
    void* buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Request* request)
{
    int res = MPI_Irecv(buf, count, datatype, source, tag, comm, request);
    msr_deb<6>.debug(str<>("MPI_Irecv"), dec<5>(count));
    return res;
}

// ------------------------------------------------------------
// this is called on an pika thread after the runtime starts up
int pika_main(pika::program_options::variables_map& vm)
{
    std::int32_t rank, size;
    //
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // enable MPI error handler to throw pika::mpi_exception
    pika::mpi::experimental::init(false, mpi::get_pool_name(), true);

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
    const std::uint64_t num_rounds = vm["rounds"].as<std::uint64_t>();
    output = vm.count("output") != 0;
    //
    auto in_flight = vm["in-flight-limit"].as<std::uint32_t>();
    mpi::set_max_requests_in_flight(in_flight, mpi::stream_type::user_1);

    mpi_poll_size = vm["mpi-polling-size"].as<std::uint32_t>();
    mpi::set_max_polling_size(mpi_poll_size);

    std::uint64_t message_size = vm["message-bytes"].as<std::uint64_t>();

    // --------------------------
    // main scope with polling enabled
    // --------------------------
    {
        // the scope of the user polling must include all uses of transform_mpi
        // this needs to scope all uses of mpi::experimental::executor
        std::string poolname = "default";
        if (pika::resource::pool_exists(mpi::get_pool_name()))
        {
            poolname = mpi::get_pool_name();
        }
        mpi::enable_user_polling enable_polling(poolname);

        // To prevent the application exiting the main scope of mpi polling
        // whilst there are messages in flight, we will count each send/recv and
        // wait until all are done
        // each iteration initiate ring = send + recv
        // each iteration participate in = recv + send (for n_ranks - 1)
        // #nummessages = (2 * iterations) + (size-1)*2*iterations
        counter = 2 * size * num_rounds * iterations;

        // start a timer
        pika::chrono::detail::high_resolution_timer t;

        // for each iteration (number of times we loop over the ranks)
        for (std::uint64_t i = 0; i < iterations; ++i)
        {
            // every rank will start a ring, and forward messages from
            // other ranks (prev) in the ring
            for (std::int32_t orank = 0; orank < size; orank++)
            {
                // post a ring receive for each rank at this iteration
                // the message is always received from the previous rank
                auto buf = get_msg_buffer(message_size);
                std::uint64_t tag = tag_no(std::uint64_t(orank), std::uint64_t(i), std::uint32_t(size));
                msg_info(rank, size, msg_type::recv, buf->token_val_, tag, 0, 0, "ringrecv");

                // a handler for a receive that recursively posts receives and handles them
                receiver reclambda(rank, orank, size, tag, num_rounds, message_size, buf);
                // create chain of senders to make the mpi recv and handle it
                auto rx_snd1 = ex::just(&*buf, message_size, MPI_UNSIGNED_CHAR, prev_rank(rank, size), tag,
                                   MPI_COMM_WORLD) |
                    mpi::transform_mpi(MPI_Irecv, mpi::stream_type::receive_1) | ex::then(reclambda);
                msr_deb<6>.debug(str<>("start_detached"), "rx_snd1", i, orank);
                pika::execution::async(std::move(rx_snd1));
            }

            // start the ring (first message) message for this iteration/rank
            auto buf = get_msg_buffer(message_size);
            buf->token_val_ = 0;
            std::uint64_t tag = tag_no(std::uint64_t(rank), std::uint64_t(i), std::uint32_t(size));
            auto send_snd =
                ex::just(&*buf, message_size, MPI_UNSIGNED_CHAR, next_rank(rank, size), tag, MPI_COMM_WORLD) |
                mpi::transform_mpi(MPI_Isend, mpi::stream_type::user_1) |
                ex::then([rank, size, buf, tag](int /*result*/) {
                    counter--;
                    msg_info(rank, size, msg_type::send, buf->token_val_, tag, 0, 0, "initsent");
                    release_msg_buffer(buf);
                });
            msg_info(rank, size, msg_type::send, buf->token_val_, tag, 0, 0, "init");
            msr_deb<6>.debug(str<>("start_detached"), "send_snd", i);
            pika::execution::async(std::move(send_snd));
        }

        // don't exit until all messages are drained
        while (counter > 0)
        {
            pika::this_thread::yield();
        }
        if (output)
        {
            using namespace pika::debug::detail;
            // clang-format off
            msr_deb<0>.debug(str<>("User Messages")
                , "Rank", dec<3>(rank), "of", dec<3>(size)
                , "Counter", hex<8>(std::uint64_t(counter.load()))
                , "in-flight u_1", dec<3>(
                    mpi::get_num_requests_in_flight(mpi::stream_type::user_1))
                , "in-flight s_1", dec<3>(
                    mpi::get_num_requests_in_flight(mpi::stream_type::send_1))
                , "in-flight r_1", dec<3>(
                    mpi::get_num_requests_in_flight(mpi::stream_type::receive_1))
                , "in-flight r_2", dec<3>(
                    mpi::get_num_requests_in_flight(mpi::stream_type::receive_2)));
            // clang-format on
        }

        // the user queue should always be empty by now since our counter tracks it
        PIKA_ASSERT(mpi::get_num_requests_in_flight(mpi::stream_type::user_1) == 0);
        PIKA_ASSERT(mpi::get_num_requests_in_flight(mpi::stream_type::send_1) == 0);
        PIKA_ASSERT(mpi::get_num_requests_in_flight(mpi::stream_type::receive_1) == 0);
        PIKA_ASSERT(mpi::get_num_requests_in_flight(mpi::stream_type::receive_2) == 0);

        // don't exit until messages that are still in flight are drained
        while (mpi::get_num_requests_in_flight(mpi::stream_type::send_1) +
                mpi::get_num_requests_in_flight(mpi::stream_type::receive_1) >
            0)
        {
            pika::this_thread::yield();
        }

        double elapsed = t.elapsed();

        std::cout << "Rank " << rank << " reached end of test " << counter << std::endl;

        if (rank == 0)
        {
            // a complete set of results formatted for plotting
            std::stringstream temp;
            constexpr char const* msg = "CSVData, "
                                        "{}, in_flight, {}, ranks, {}, threads, {}, "
                                        "iterations, {}, task_transfer, {}, "
                                        "message-size, {}, polling-size, {}, time";
            fmt::print(temp, msg, in_flight, size, pika::get_num_worker_threads(), iterations,
                mpi::get_completion_mode(), message_size, mpi_poll_size, elapsed);
            std::cout << temp.str();
        }
        // let the user polling go out of scope
    }
    return pika::finalize();
}

//----------------------------------------------------------------------------
void init_resource_partitioner_handler(
    pika::resource::partitioner& rp, pika::program_options::variables_map const& vm)
{
    // Don't create the MPI pool if the user disabled it
    if (vm["no-mpi-pool"].as<bool>())
    {
        mpi::set_pool_name("default");
        return;
    }

    // Don't create the MPI pool if there is a single process
    int ntasks;
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
    //    if (ntasks == 1)
    //        return;

    // Disable idle backoff on the MPI pool
    using pika::threads::scheduler_mode;
    auto mode = scheduler_mode::default_mode;
    mode = scheduler_mode(mode & ~scheduler_mode::enable_idle_backoff);

    // Create a thread pool with a single core that we will use for all
    // communication related tasks
    rp.create_thread_pool(mpi::get_pool_name(), pika::resource::scheduling_policy::unspecified, mode);

    rp.add_resource(rp.numa_domains()[0].cores()[0].pus()[0], mpi::get_pool_name());
}

//----------------------------------------------------------------------------
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
        std::cout << "Provided MPI is not : MPI_THREAD_MULTIPLE " << provided << std::endl;
    }

    // Configure application-specific options.
    options_description cmdline("usage: " PIKA_APPLICATION_STRING " [options]");

    // clang-format off
    cmdline.add_options()("iterations",
        value<std::uint64_t>()->default_value(5000),
        "number of iterations to test");

    cmdline.add_options()("rounds",
        value<std::uint64_t>()->default_value(5),
        "number of times around the ring each message should flow");

    cmdline.add_options()("in-flight-limit",
        pika::program_options::value<std::uint32_t>()->default_value(
            mpi::get_max_requests_in_flight()),
        "Apply a limit to the number of messages in flight.");

    cmdline.add_options()("no-mpi-pool", pika::program_options::bool_switch(),
        "Disable the MPI pool.");

    cmdline.add_options()("mpi-polling-size",
        pika::program_options::value<std::uint32_t>()->default_value(16),
        "The maximum number of mpi request completions to handle per poll.");

    cmdline.add_options()("message-bytes",
        pika::program_options::value<std::uint64_t>()->default_value(64),
        "Specify the buffer size to use for messages (min 16).");

    cmdline.add_options()("output",
        "Display messages during test");

    cmdline.add_options()("standalone",
        "Allow test to run with a single rank (debugging)");
    // clang-format on

    // Initialize and run pika.
    pika::init_params init_args;
    init_args.desc_cmdline = cmdline;
    // Set the callback to init thread_pools
    init_args.rp_callback = &init_resource_partitioner_handler;

    auto result = pika::init(pika_main, argc, argv, init_args);
    PIKA_TEST_EQ(result, 0);

    // Finalize MPI
    MPI_Finalize();
    return result;
}
