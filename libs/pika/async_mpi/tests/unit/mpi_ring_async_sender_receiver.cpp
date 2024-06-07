//  Copyright (c) 2023 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/assert.hpp>
#include <pika/command_line_handling/get_env_var_as.hpp>
#include <pika/debugging/print.hpp>
#include <pika/execution.hpp>
#include <pika/init.hpp>
#include <pika/mpi.hpp>
#include <pika/program_options.hpp>
#include <pika/synchronization/counting_semaphore.hpp>
#include <pika/testing.hpp>
#include <pika/thread.hpp>
//
#include <boost/lockfree/stack.hpp>
#include <fmt/format.h>
#include <fmt/printf.h>
//
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
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

using namespace pika::debug::detail;

namespace ex = pika::execution::experimental;
namespace mpix = pika::mpi::experimental;

static bool output = false;
static std::uint32_t mpi_poll_size = 16;
std::atomic<std::uint32_t> counter;
std::unique_ptr<pika::counting_semaphore<>> limiter;

// ------------------------------------------------------------
// a debug level of zero disables messages with a priority>0
// a debug level of N shows messages with priority<N
template <int Level>
static print_threshold<Level, 0> msr_deb("MPI_SR_");

// ------------------------------------------------------------
// caution: message_buffers will be constructed in-place in a buffer
// allocated with a larger size (set by message-bytes option)
struct header
{
    std::uint32_t token_val_;
    std::uint32_t tag;
    std::uint32_t iteration;
    std::uint32_t round;
    std::uint32_t step;
    std::uint32_t origin_rank;
    std::uint32_t size_;
};

struct message_buffer
{
    header header_;
    //
    std::uint32_t size() { return header_.size_; }
    //
    explicit message_buffer(header& h)
    {
        assert(h.size_ > sizeof(header));
        header_ = h;
    }
};

// ------------------------------------------------------------
inline std::uint32_t next_rank(std::uint32_t rank, std::uint32_t size) { return (rank + 1) % size; }

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
inline std::uint32_t make_tag(std::uint32_t rank, std::uint32_t iteration, std::uint32_t ranks)
{
    std::int64_t tag = (rank + (iteration * ranks)) & 0xffff'ffff;
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
void msg_info(
    std::uint32_t rank, std::uint32_t size, msg_type mtype, header h, const char* xmsg = nullptr)
{
    if (output)
    {
        int other = (mtype == msg_type::send) ? next_rank(rank, size) : prev_rank(rank, size);
        const char* msg = (mtype == msg_type::send) ? "send" : "recv";
        std::stringstream temp;
        temp << dec<3>(rank) << "/" << dec<3>(size);
        // clang-format off
        msr_deb<1>.debug(str<>(temp.str().c_str())
                         , "token", hex<4>(h.token_val_)
                         , dec<3>(rank), "<-->", dec<3>(other), msg
                         , "tag", dec<3>(h.tag)
                         , "iteration", dec<3>(h.iteration)
                         , "round", dec<3>(h.round)
                         , "step", dec<3>(h.step)
                         , "origin", dec<3>(h.origin_rank)
                         , str<12>((xmsg == nullptr) ? "" : xmsg)
                         , "counter", std::int64_t(counter));
        // clang-format on
    }
}

// ------------------------------------------------------------
// message buffers get reused from a stack
boost::lockfree::stack<message_buffer*, boost::lockfree::fixed_sized<false>> message_buffers(1024);
std::atomic<std::uint32_t> message_buffers_size_{0};
#define BUFFER_CACHE

message_buffer* get_msg_buffer(header h)
{
    message_buffer* buffer;
#ifdef BUFFER_CACHE
    if (message_buffers.pop(buffer))
    {
        // setup the header
        buffer->header_ = h;
    }
    else
#endif
    {
        // allocate the amount of space we want
        void* data = new unsigned char[h.size_];
        // construct our buffer object in that space
        buffer = new (data) message_buffer(h);
    }
    // set initial token to some easy to spot default
    message_buffers_size_++;
    msr_deb<6>.debug(str<>("message_buffers"), std::uint32_t(message_buffers_size_.load()));
    return buffer;
}

void release_msg_buffer(message_buffer* buffer)
{
    --message_buffers_size_;
#ifdef BUFFER_CACHE
    message_buffers.push(buffer);
#else
    char* data = reinterpret_cast<char*>(buffer);
    delete[] data;
#endif
    msr_deb<6>.debug(str<>("message_buffers"), std::uint32_t(message_buffers_size_.load()));
}

struct buffer_cleaner_upper
{
    message_buffer* buffer;
    ~buffer_cleaner_upper()
    {
        while (message_buffers.pop(buffer))
        {
            char* data = reinterpret_cast<char*>(buffer);
            delete[] data;
        }
    }
};

static buffer_cleaner_upper dummy;

struct message_receiver
{
    std::uint32_t rank, orank, size, tag, num_rounds, message_size;
    message_buffer* buf;

    message_receiver(std::uint32_t rank, std::uint32_t orank, std::uint32_t size, std::uint32_t tag,
        std::uint32_t num_rounds, std::uint32_t message_size, message_buffer* buf)
      : rank(rank)
      , orank(orank)
      , size(size)
      , tag(tag)
      , num_rounds(num_rounds)
      , message_size(message_size)
      , buf(buf)
    {
    }

    void operator()(/*int res*/)
    {
        PIKA_ASSERT(tag == buf->header_.tag);

        // global counter for sanity checking of the test itself
        counter--;

        // increment the token we received
        buf->header_.token_val_++;

        // each message travels around the ring num_rounds times,
        // which round is the message currently on?
        buf->header_.round = buf->header_.token_val_ / size;
        buf->header_.step = buf->header_.token_val_ % size;

        msr_deb<2>.debug(str<>("operator"), rank, size, "tag", dec<3>(buf->header_.tag),
            "iteration", buf->header_.iteration, "round", buf->header_.round, "step",
            buf->header_.step);

        // if the message has arrived back at the starting rank
        // and has completed all rounds, it is finished
        if (buf->header_.round == num_rounds)
        {
            if (buf->header_.step != 0)
            {
                throw std::runtime_error("Ring should have terminated before now");
            }
            msg_info(rank, size, msg_type::recv, buf->header_, "complete");
            // msr_deb<0>.debug(str<>("release"), buf->header_.iteration);
            msr_deb<2>.debug(str<>("release"), "tag", buf->header_.tag);
            release_msg_buffer(buf);
            limiter->release();
        }
        else
        {
            header hcopy = buf->header_;

            // origin rank (step==0) receives one more than the other steps/ranks
            if (buf->header_.step == 0 || (buf->header_.round < (num_rounds - 1)))
            {
                msg_info(rank, size, msg_type::recv, buf->header_, "recv_R");
                // post a receive to handle it when it comes back on the next round
                // we can reuse the same buffer for the forwarding send and for the receive,
                // because the send _must_ complete before the receive
                // even if there is only one rank in the ring
                message_receiver reclambda(rank, orank, size, tag, num_rounds, message_size, buf);

                // the recursive lambda will handle it
                auto rx_snd2 = ex::just(buf, message_size, MPI_UNSIGNED_CHAR, prev_rank(rank, size),
                                   tag, MPI_COMM_WORLD) |
                    mpix::transform_mpi(MPI_Irecv /*, mpix::stream_type::receive_2*/) |
                    ex::then(std::move(reclambda));
                // launch the receive for the msg on the next round
                msr_deb<6>.debug(
                    str<>("start_detached"), "rx_snd2", buf->header_.round, buf->header_.step);
                ex::start_detached(std::move(rx_snd2));
            }
            else { release_msg_buffer(buf); }

            // prepare new send buffer for forwarding message on
            auto buf2 = get_msg_buffer(hcopy);
            msg_info(rank, size, msg_type::send, buf2->header_, "send_R");
            auto tx_snd2 = ex::just(buf2, message_size, MPI_UNSIGNED_CHAR, next_rank(rank, size),
                               tag, MPI_COMM_WORLD) |
                mpix::transform_mpi(MPI_Isend /*, mpix::stream_type::send_2*/) |
                ex::then([buf2 = buf2, rank = rank, size = size](/*int result*/) {
                    counter--;
                    msg_info(rank, size, msg_type::send, buf2->header_, "forwarded");
                    release_msg_buffer(buf2);
                });

            // launch the forwarding send for the current round
            msr_deb<6>.debug(
                str<>("start_detached"), "tx_snd2", buf2->header_.round, buf2->header_.step);
            ex::start_detached(std::move(tx_snd2));
        }
    }
};

// ------------------------------------------------------------
int call_mpi_irecv(void* buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm,
    MPI_Request* request)
{
    int res = MPI_Irecv(buf, count, datatype, source, tag, comm, request);
    msr_deb<6>.debug(str<>("MPI_Irecv"), dec<5>(count));
    return res;
}

// ------------------------------------------------------------
// this is called on a pika thread after the runtime starts up
int pika_main(pika::program_options::variables_map& vm)
{
    // Do not initialize mpi (we do that ourselves), do install an error handler
    mpix::init(false, true);
    // Setup mpi polling on default pool, enable exceptions and init mpi internals
    mpix::register_polling();
    //
    std::int32_t rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // --------------------------
    // if not debugging standalone and comm size < 2 this test should fail
    // it needs to run on N>2 ranks to be useful
    // --------------------------
    if (vm.count("standalone") == 0)
    {
        PIKA_TEST_MSG(
            size > 1, "This test requires N>1 mpi ranks, use --standalone for single node testing");
    }

    // --------------------------
    // Get user options/flags
    // --------------------------
    const std::uint32_t iterations = vm["iterations"].as<std::uint32_t>();
    const std::uint32_t num_rounds = vm["rounds"].as<std::uint32_t>();
    output = vm.count("output") != 0;
    //
    std::uint32_t in_flight = 65536;
    if (vm.count("in-flight-limit")) { in_flight = vm["in-flight-limit"].as<std::uint32_t>(); }
    limiter = std::make_unique<pika::counting_semaphore<>>(in_flight);

    mpi_poll_size = vm["mpi-polling-size"].as<std::uint32_t>();
    mpix::set_max_polling_size(mpi_poll_size);

    std::uint32_t message_size = vm["message-bytes"].as<std::uint32_t>();

    // --------------------------
    // main scope with polling enabled
    // --------------------------
    {
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
        for (std::uint32_t i = 0; i < iterations; ++i)
        {
            msr_deb<3>.debug(str<>("acquire"), i);
            limiter->acquire();

            // every rank starts a ring send,
            // on first round - every rank receives a message from all other ranks
            for (std::uint32_t orank = 0; orank < std::uint32_t(size); orank++)
            {
                // post a ring receive for each rank at this iteration
                // the message is always received from the previous rank
                // and has the tag associated with the originating rank
                std::uint32_t tag =
                    make_tag(std::uint32_t(orank), std::uint32_t(i), std::uint32_t(size));
                auto rbuf = get_msg_buffer(header{0, tag, i, 0, 0, orank, message_size});
                msg_info(rank, size, msg_type::recv, rbuf->header_, "recv");

                // a handler for a receive that recursively posts receives and handles them
                message_receiver reclambda(rank, orank, size, tag, num_rounds, message_size, rbuf);
                // create chain of senders to make the mpi recv and handle it
                auto rx_snd1 = ex::just(rbuf, message_size, MPI_UNSIGNED_CHAR,
                                   prev_rank(rank, size), tag, MPI_COMM_WORLD) |
                    mpix::transform_mpi(MPI_Irecv /*, mpix::stream_type::receive_1*/) |
                    ex::then(std::move(reclambda));
                msr_deb<6>.debug(str<>("start_detached"), "rx_snd1", i, orank);
                ex::start_detached(std::move(rx_snd1));
            }

            // start the ring (first message) message for this iteration/rank
            std::uint32_t tag =
                make_tag(std::uint32_t(rank), std::uint32_t(i), std::uint32_t(size));
            auto sbuf = get_msg_buffer(header{0, tag, i, 0, 0, std::uint32_t(rank), message_size});
            auto send_snd = ex::just(sbuf, message_size, MPI_UNSIGNED_CHAR, next_rank(rank, size),
                                tag, MPI_COMM_WORLD) |
                mpix::transform_mpi(MPI_Isend /*, mpix::stream_type::send_1*/) |
                ex::then([rank, size, sbuf](/*int res*/) {
                    counter--;
                    msg_info(rank, size, msg_type::send, sbuf->header_, "sent");
                    release_msg_buffer(sbuf);
                });
            msg_info(rank, size, msg_type::send, sbuf->header_, "send");
            msr_deb<6>.debug(str<>("start_detached"), "send_snd", i);
            ex::start_detached(std::move(send_snd));
        }

        // don't exit until all messages are drained
        while (counter > 0) { pika::this_thread::yield(); }
        if (output)
        {
            // clang-format off
            msr_deb<0>.debug(str<>("User Messages")
                , "Rank", dec<3>(rank), "of", dec<3>(size)
                , "Counter", hex<8>(std::uint32_t(counter.load()))
                , "in-flight", dec<3>(mpix::get_work_count()));
            // clang-format on
        }

        // the user queue should always be empty by now since our counter tracks it
        PIKA_TEST_EQ(mpix::get_work_count(), static_cast<std::size_t>(0));

        double elapsed = t.elapsed();

        if (rank == 0)
        {
            // a complete set of results formatted for plotting
            std::stringstream temp;
            constexpr char const* msg =
                "CSVData-2, in_flight, {}, ranks, {}, threads, {}, iterations, {}, rounds, {}, "
                "completion_mode, {}, message-size, {}, polling-size, {}, time, {}, {}";
            fmt::print(temp, msg, in_flight, size, pika::get_num_worker_threads(), iterations,
                num_rounds, mpix::get_completion_mode(), message_size, mpi_poll_size, elapsed,
                vm["pp-info"].as<std::string>());
            std::cout << temp.str() << std::endl;
        }
        // let the user polling go out of scope
    }

    pika::finalize();
    return EXIT_SUCCESS;
}

//----------------------------------------------------------------------------
void init_resource_partitioner_handler(
    pika::resource::partitioner&, pika::program_options::variables_map const& vm)
{
    // Don't create an MPI pool if the user disabled it
    auto pool_mode = mpix::pool_create_mode::pika_decides;
    if (vm["no-mpi-pool"].as<bool>()) { pool_mode = mpix::pool_create_mode::force_no_create; }

    mpix::enable_optimizations(vm["mpi-optimizations"].as<bool>());
    std::cout << "init_resource_partitioner_handler enable optimizations "
              << vm["mpi-optimizations"].as<bool>() << std::endl;

    msr_deb<2>.debug(str<>("init RP"), "create_pool");
    mpix::create_pool("", pool_mode);
}

//----------------------------------------------------------------------------
// the normal int main function that is called at startup and runs on an OS
// thread the user must call pika::init to start the pika runtime which
// will execute pika_main on a pika thread
int main(int argc, char* argv[])
{
    // Configure application-specific options.
    options_description cmdline("usage: " PIKA_APPLICATION_STRING " [options]");

    // clang-format off
    cmdline.add_options()("iterations",
        value<std::uint32_t>()->default_value(5000),
        "number of iterations to test");

    cmdline.add_options()("rounds",
        value<std::uint32_t>()->default_value(5),
        "number of times around the ring each message should flow");

    cmdline.add_options()("in-flight-limit",
        pika::program_options::value<std::uint32_t>()->default_value(128),
        "Apply a limit to the number of messages in flight.");

    cmdline.add_options()("no-mpi-pool", pika::program_options::bool_switch(),
        "Disable the MPI pool.");

    cmdline.add_options()("mpi-polling-size",
        pika::program_options::value<std::uint32_t>()->default_value(16),
        "The maximum number of mpi request completions to handle per poll.");

    cmdline.add_options()("message-bytes",
        pika::program_options::value<std::uint32_t>()->default_value(64),
        "Specify the buffer size to use for messages (min 16).");

    cmdline.add_options()("output",
        "Display messages during test");

    cmdline.add_options()("standalone",
        "Allow test to run with a single rank (debugging)");

    cmdline.add_options()("mpi-optimizations", pika::program_options::bool_switch(),
        "do not change pool, or thread level");

    cmdline.add_options()("csv", pika::program_options::bool_switch()->default_value(false),
                     "Enable CSV output of values");

    cmdline.add_options()("pp-info", pika::program_options::value<std::string>()->default_value(""),
                     "Info for postprocessing scripts appended to csv output (if enabled)");
    // clang-format on

    // -----------------
    // process command line options early so we can use them for mpi_init
    namespace po = pika::program_options;
    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(cmdline).allow_unregistered().run(), vm);
    po::notify(vm);

    // -----------------
    // Init MPI
    int provided, preferred = mpix::get_preferred_thread_mode();
    MPI_Init_thread(&argc, &argv, preferred, &provided);
    if (provided != preferred)
    {
        msr_deb<0>.error(str<>("Caution"), "Provided MPI is not as requested");
    }

    // -----------------
    // Initialize and run pika.
    pika::init_params init_args;
    init_args.desc_cmdline = cmdline;
    // Set the callback to init thread_pools
    init_args.rp_callback = &init_resource_partitioner_handler;

    auto result = pika::init(pika_main, argc, argv, init_args);
    PIKA_TEST_EQ(result, 0);

    // -----------------
    // Finalize MPI
    MPI_Finalize();
    return result;
}
