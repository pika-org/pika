//  Copyright (c) 2022 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/execution.hpp>
#include <pika/init.hpp>
#include <pika/mpi.hpp>
#include <pika/program_options.hpp>
#include <pika/testing.hpp>

#include <pika/config.hpp>
#include <pika/execution_base/this_thread.hpp>
#include <pika/modules/format.hpp>
#include <pika/mpi_base/mpi_environment.hpp>
#include <pika/threading_base/print.hpp>
//
#define RANK_OUTPUT (rank == 0)

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

// a debug level of zero disables messages with a priority>0
// a debug level of N shows messages with priority<N
constexpr int debug_level = 0;

// cppcheck-suppress ConfigurationNotChecked
template <int Level>
static pika::debug::detail::print_threshold<Level, debug_level> nws_deb(
    "STORAGE");

//----------------------------------------------------------------------------
//
// Each locality allocates a buffer of memory which is used to host transfers
//
using local_storage_type = std::vector<char>;
static local_storage_type local_send_storage;
static local_storage_type local_recv_storage;

//----------------------------------------------------------------------------
// namespace aliases
//----------------------------------------------------------------------------
namespace ex = pika::execution::experimental;
namespace mpi = pika::mpi::experimental;
namespace tt = pika::this_thread::experimental;
namespace deb = pika::debug::detail;

//----------------------------------------------------------------------------
// namespace aliases
//----------------------------------------------------------------------------
struct test_options
{
    std::uint64_t local_storage_MB;
    std::uint64_t transfer_size_B;
    std::uint32_t threads;
    std::uint32_t num_seconds;
    std::uint32_t in_flight_limit;

    bool warmup;
    bool final;
};

//----------------------------------------------------------------------------
struct perf
{
    size_t rank;
    size_t nranks;
    double dataMB;
    double time;
    double IOPs;
    std::string mode;
};

//----------------------------------------------------------------------------
void display(perf const& data, const test_options& options)
{
    std::stringstream temp;
    double IOPs_s = data.IOPs / data.time;
    double BW = data.dataMB / data.time;
    if (data.rank == 0 && !options.warmup)
    {
        temp << "\n";
        temp << "Rank                 : " << data.rank << " / " << data.nranks
             << "\n";
        temp << "Total time           : " << data.time << "\n";
        temp << "Memory Transferred   : " << data.dataMB << " MB\n";
        temp << "Number of local IOPs : " << data.IOPs << "\n";
        temp << "IOPs/s (local)       : " << IOPs_s << "\n";
        temp << "Aggregate BW         : " << BW << " MB/s";
        temp << "\n";
        if (/*options.final && */ !options.warmup)
        {
            // a complete set of results that our python matplotlib script will ingest
            char const* msg =
                "CSVData, {1}, network, "
                "{2}, ranks, {3}, threads, {4}, Memory, {5}, IOPsize, {6}, "
                "IOPS/s, {7}, BW(MB/s), {8}, in_flight, {9}";
            pika::util::format_to(temp, msg, data.mode.c_str(), "pika-mpi",
                data.nranks, options.threads, data.dataMB,
                options.transfer_size_B, IOPs_s, BW, options.in_flight_limit)
                << std::endl;
        }
        std::cout << temp.str() << std::endl;
    }
}

//----------------------------------------------------------------------------
static std::vector<perf> performance_figures;
static std::atomic<size_t> results_received{0};
static test_options options;

//----------------------------------------------------------------------------
static std::atomic<std::uint32_t> sends_in_flight;
static std::atomic<std::uint32_t> recvs_in_flight;

//----------------------------------------------------------------------------
// Test speed of write/put
void test_send_recv(std::uint32_t rank, std::uint32_t nranks, std::mt19937& gen,
    std::uniform_int_distribution<std::uint64_t>& random_offset,
    std::uniform_int_distribution<std::uint64_t>& random_slot,
    test_options& options)
{
    static deb::print_threshold<1, debug_level> write_arr(" WRITE ");

    // this needs to scope all uses of mpi::experimental::executor
    std::string poolname = "default";
    if (pika::resource::pool_exists("mpi"))
        poolname = "mpi";
    mpi::enable_user_polling enable_polling(poolname);

    pika::scoped_annotation annotate("test_write");
    std::stringstream temp;
    temp << deb::hostname_print_helper();
    std::string name = temp.str() + "-test-write.prof";
    std::replace(name.begin(), name.end(), ':', '-');

    results_received = 0;
    sends_in_flight = 0;
    recvs_in_flight = 0;
    std::uint64_t messages_sent = 0;
    //
    if (rank == 0)
    {
        std::cout << (options.warmup ? "Warmup   " : "Progress ");
    }

    // generate an array of location offsets where we are going to send data
    constexpr size_t array_size = 1024;
    std::array<std::size_t, array_size> offsets, sends, recvs;
    std::generate(
        offsets.begin(), offsets.end(), [&]() { return random_offset(gen); });
    std::transform(offsets.begin(), offsets.end(), sends.begin(),
        [&](std::size_t& offset) { return (rank + offset) % nranks; });
    std::transform(offsets.begin(), offsets.end(), recvs.begin(),
        [&](std::size_t& offset) { return (rank - offset) % nranks; });

    std::string tempstr = "rank " + std::to_string(rank);
    write_arr.array(
        tempstr + " # offset  ", &offsets[0], (std::min)(array_size, 32ul));
    write_arr.array(
        tempstr + " # send    ", &sends[0], (std::min)(array_size, 32ul));
    write_arr.array(
        tempstr + " # recv    ", &recvs[0], (std::min)(array_size, 32ul));

    // ----------------------------------------------------------------
    nws_deb<2>.debug("Entering Barrier at start of write on rank", rank);
    MPI_Barrier(MPI_COMM_WORLD);
    nws_deb<2>.debug("Passed Barrier at start of write on rank", rank);
    // ----------------------------------------------------------------

    // create a timer object with a 1s tick
    auto debug_timer = nws_deb<0>.make_timer(1);
    int debug_tick = 0;

    std::vector<MPI_Request> recvsr(nranks);
    std::vector<MPI_Request> sendsr(nranks);
    std::vector<MPI_Status> recvss(nranks);
    std::vector<MPI_Status> sendss(nranks);
    recvsr[rank] = MPI_REQUEST_NULL;
    sendsr[rank] = MPI_REQUEST_NULL;
    std::vector<std::uint64_t> counts(nranks, 0);

    //
    // Start main message sending loop
    //
    pika::chrono::detail::high_resolution_timer exec_timer;
    std::uint64_t final_count = (std::numeric_limits<std::uint64_t>().max)();
    bool count_complete = false;
    // loop for allowed time : sending and receiving
    do
    {
        // read from a random slot on this node
        // (and ideally write to a random slot on remote node)
        int read_slot = random_slot(gen);
        int write_slot = random_slot(gen);

        std::uint32_t memory_offset_recv =
            static_cast<std::uint32_t>(read_slot * options.transfer_size_B);
        std::uint32_t memory_offset_send =
            static_cast<std::uint32_t>(write_slot * options.transfer_size_B);

        // next message to recv from here
        std::uint32_t recv_rank = recvs[messages_sent % array_size];
        // next message to send goes here
        std::uint32_t send_rank = sends[messages_sent % array_size];
        //
        int tag = messages_sent & 0xffff;
        {
            // we are about to recv, so increment our counter
            ++recvs_in_flight;
            // clang-format off
            nws_deb<5>.debug(deb::str<>("posting recv"),
                             "Rank ", deb::dec<3>(rank),
                             "recv block ", deb::hex<8>(read_slot),
                             "<- rank", deb::dec<4>(recv_rank),
                             "tag", deb::hex<4>(tag),
                             "recv in flight", deb::dec<4>(recvs_in_flight),
                             "send in flight", deb::dec<4>(sends_in_flight));
            // clang-format on
            void* buffer_to_recv = &local_recv_storage[memory_offset_recv];
            auto rsnd = ex::just(buffer_to_recv, options.transfer_size_B,
                            MPI_UNSIGNED_CHAR, recv_rank, tag, MPI_COMM_WORLD) |
                mpi::transform_mpi(
                    MPI_Irecv, /*mpi::mpi_stream::*/ mpi::receive_stream) |
                ex::then([&](int result) {
                    --recvs_in_flight;
                    nws_deb<5>.debug(deb::str<>("recv complete"),
                        "recv in flight", recvs_in_flight, "send in flight",
                        sends_in_flight);
                    return result;
                });
            ex::start_detached(std::move(rsnd));

            // we are about to send, so increment our counter
            ++sends_in_flight;
            // clang-format off
            nws_deb<5>.debug(deb::str<>("posting send"),
                             "Rank ", deb::dec<3>(rank),
                             "send block ", deb::hex<8>(write_slot),
                             "-> rank", deb::dec<4>(send_rank),
                             "tag", deb::hex<4>(tag),
                             "recv in flight", deb::dec<4>(recvs_in_flight),
                             "send in flight", deb::dec<4>(sends_in_flight));
            // clang-format on
            void* buffer_to_send = &local_send_storage[memory_offset_send];
            auto ssnd = ex::just(buffer_to_send, options.transfer_size_B,
                            MPI_UNSIGNED_CHAR, send_rank, tag, MPI_COMM_WORLD) |
                mpi::transform_mpi(
                    MPI_Isend, mpi::/*mpi_stream::*/ send_stream) |
                ex::then([&](int result) {
                    --sends_in_flight;
                    nws_deb<5>.debug(deb::str<>("send complete"),
                        "recv in flight", recvs_in_flight, "send in flight",
                        sends_in_flight);
                    return result;
                });
            ex::start_detached(std::move(ssnd));
        }
        messages_sent++;
        //
        if (RANK_OUTPUT && debug_timer.trigger())
        {
            std::cout << ((debug_tick++ % 10 == 9) ? "x" : ".") << std::flush;
        }

        bool time_up = (exec_timer.elapsed() >= options.num_seconds);

        //
        if (time_up && !count_complete)
        {
            for (std::uint32_t i = 0; i < nranks; ++i)
            {
                if (i != rank)
                {
                    MPI_Irecv(&counts[i], 1, MPI_INT32_T, i, 0, MPI_COMM_WORLD,
                        &recvsr[i]);
                    MPI_Isend(&messages_sent, 1, MPI_INT32_T, i, 0,
                        MPI_COMM_WORLD, &sendsr[i]);
                }
            }
            for (std::uint32_t i = 0; i < nranks; ++i)
            {
                MPI_Waitall(nranks, sendsr.data(), sendss.data());
                MPI_Waitall(nranks, recvsr.data(), recvss.data());
            }
            // reduction step
            final_count = messages_sent;
            std::for_each(counts.begin(), counts.end(), [&](std::uint64_t c) {
                final_count = (std::max)(final_count, c);
            });
            nws_deb<2>.debug(
                "Rank ", rank, "count max", deb::dec<8>(final_count));
            //
            count_complete = true;
        }

    } while (messages_sent < final_count);

    nws_deb<2>.debug(
        "Time elapsed", "on rank", rank, "counter", deb::dec<8>(messages_sent));

    // block until no messages are in flight
    nws_deb<2>.debug(deb::str<>("pre final"), "Rank ", rank, "recvs in flight",
        deb::dec<>(recvs_in_flight), "sends in flight",
        deb::dec<>(sends_in_flight));
    //
    while ((sends_in_flight.load() + recvs_in_flight.load()) > 0)
    {
        mpi::detail::poll();
        if (debug_timer.trigger())
            nws_deb<5>.debug(
                "Rank ", rank, "counter", deb::dec<8>(messages_sent));
    }
    //
    nws_deb<2>.debug(deb::str<>("final"), "Rank ", rank, "recvs in flight",
        deb::dec<>(recvs_in_flight), "sends in flight",
        deb::dec<>(sends_in_flight));

    // ----------------------------------------------------------------
    double MB = nranks *
        static_cast<double>(messages_sent * options.transfer_size_B) /
        (1024 * 1024);
    double IOPs = static_cast<double>(messages_sent);
    double Time = exec_timer.elapsed();

    if (rank == 0)
        std::cout << std::endl;

    // ----------------------------------------------------------------
    nws_deb<2>.debug(
        "Entering Barrier before update_performance on rank", rank);
    MPI_Barrier(MPI_COMM_WORLD);
    nws_deb<2>.debug("Passed Barrier before update_performance on rank", rank);
    // ----------------------------------------------------------------

    perf p{
        rank, nranks, MB, Time, IOPs, options.warmup ? "warmup" : "send/recv "};
    display(p, options);
}

//----------------------------------------------------------------------------
// Main test loop which randomly sends packets of data from one locality to
// another looping over the entire buffer address space and timing the total
// transmit/receive time to see how well we're doing.
int pika_main(pika::program_options::variables_map& vm)
{
    // Disable idle backoff on the default pool
    pika::threads::remove_scheduler_mode(
        pika::threads::policies::enable_idle_backoff);

    pika::util::mpi_environment mpi_env;
    pika::runtime* rt = pika::get_runtime_ptr();
    pika::util::runtime_configuration cfg = rt->get_config();
    mpi_env.init(nullptr, nullptr, cfg);

    pika::chrono::detail::high_resolution_timer timer_main;
    nws_deb<2>.debug(deb::str<>("PIKA main"));
    //
    std::string name = mpi_env.get_processor_name();
    std::uint64_t rank = mpi_env.rank();
    std::uint64_t nranks = mpi_env.size();
    std::size_t current = pika::get_worker_thread_num();

    if (rank == 0)
    {
        char const* msg = "hello world from OS-thread {:02} on locality "
                          "{:04} rank {:04} hostname {}";
        pika::util::format_to(std::cout, msg, current, pika::get_locality_id(),
            rank, name.c_str())
            << std::endl
            << std::endl;
    }

    // extract command line argument
    options.transfer_size_B =
        static_cast<std::uint64_t>(vm["transferKB"].as<double>() * 1024);
    options.local_storage_MB = vm["localMB"].as<std::uint64_t>();
    options.num_seconds = vm["seconds"].as<std::uint32_t>();
    options.in_flight_limit = vm["in-flight-limit"].as<std::uint32_t>();
    options.threads = pika::get_os_thread_count();
    options.warmup = false;
    options.final = false;

    mpi::detail::set_max_requests_in_flight(options.in_flight_limit);
    nws_deb<1>.debug("set_max_requests_in_flight", rank,
        deb::dec<04>(options.in_flight_limit));

    nws_deb<1>.debug("Allocating local storage on rank", rank, "MB",
        deb::dec<03>(options.local_storage_MB));
    local_send_storage.resize(options.local_storage_MB * 1024 * 1024);
    local_recv_storage.resize(options.local_storage_MB * 1024 * 1024);
    for (std::uint64_t i = 0; i < local_send_storage.size();
         i += sizeof(std::uint64_t))
    {
        // each block is filled with the rank and block number
        std::uint64_t temp = (rank << 32) + i / options.transfer_size_B;
        *reinterpret_cast<std::uint64_t*>(&local_send_storage[i]) = temp;
        *reinterpret_cast<std::uint64_t*>(&local_recv_storage[i]) = temp;
    }
    //
    std::uint64_t num_slots = static_cast<std::uint64_t>(1024) *
        static_cast<std::uint64_t>(1024) * options.local_storage_MB /
        options.transfer_size_B;
    nws_deb<6>.debug(
        "num ranks ", nranks, ", num_slots ", num_slots, " on rank", rank);
    //
    int64_t random_number = 0;
    if (rank == 0)
    {
        std::random_device rand;
        random_number = rand();
        nws_deb<1>.debug("Sending Seed", deb::dec<8>(random_number));
        for (uint i = 1; i < nranks; ++i)
        {
            MPI_Send(&random_number, 1, MPI_INT64_T, i, 0, MPI_COMM_WORLD);
        }
    }
    else
    {
        MPI_Recv(&random_number, 1, MPI_INT64_T, 0, 0, MPI_COMM_WORLD,
            MPI_STATUS_IGNORE);
        nws_deb<1>.debug("Received Seed", deb::dec<8>(random_number));
    }
    std::mt19937 gen(random_number);
    std::uniform_int_distribution<std::uint64_t> random_offset(
        1, (int) nranks - 1);
    std::uniform_int_distribution<std::uint64_t> random_slot(
        0, (int) num_slots - 1);

    // ----------------------------------------------------------------
    if (rank == 0)
        nws_deb<1>.debug("Completed initialization on", rank);
    // ----------------------------------------------------------------
    nws_deb<1>.debug("Entering startup_barrier on rank", rank);
    MPI_Barrier(MPI_COMM_WORLD);
    nws_deb<1>.debug("Passed startup_barrier on rank", rank);
    // ----------------------------------------------------------------

    test_options warmup = options;
    warmup.num_seconds = 1;
    warmup.warmup = true;

    if (rank == 0)
        nws_deb<1>.debug("Starting warmup", rank);
    nws_deb<1>.debug("test_write warmup on rank", rank);

    test_send_recv(rank, nranks, gen, random_offset, random_slot, warmup);
    nws_deb<1>.debug("warmup complete on rank", rank);

    test_send_recv(rank, nranks, gen, random_offset, random_slot, options);

    // ----------------------------------------------------------------
    nws_deb<1>.debug("Entering end_barrier on rank", rank);
    MPI_Barrier(MPI_COMM_WORLD);
    nws_deb<1>.debug("Passed end_barrier on rank", rank);
    // ----------------------------------------------------------------

    nws_deb<2>.debug("Deleting local storage ", rank);
    // free up any allocated memory (clear() does not free memory)
    local_recv_storage.clear();
    local_recv_storage.shrink_to_fit();
    local_send_storage.clear();
    local_send_storage.shrink_to_fit();

    nws_deb<2>.debug("Calling finalize ", rank);
    return pika::finalize();
}

//----------------------------------------------------------------------------
void init_resource_partitioner_handler(pika::resource::partitioner& rp,
    pika::program_options::variables_map const& vm)
{
    // Don't create the MPI pool if the user disabled it
    if (vm["no-mpi-pool"].as<bool>())
        return;

    // Don't create the MPI pool if there is a single process
    int ntasks;
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
    if (ntasks == 1)
        return;

    // Disable idle backoff on the MPI pool
    using pika::threads::policies::scheduler_mode;
    auto mode = scheduler_mode::default_mode;
    mode = scheduler_mode(mode & ~scheduler_mode::enable_idle_backoff);

    // Create a thread pool with a single core that we will use for all
    // communication related tasks
    rp.create_thread_pool(
        "mpi", pika::resource::scheduling_policy::local_priority_fifo, mode);
    rp.add_resource(rp.numa_domains()[0].cores()[0].pus()[0], "mpi");
}

//----------------------------------------------------------------------------
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

    // Configure application-specific options
    // some of these are not used currently but left for future tweaking
    pika::program_options::options_description cmdline(
        "Usage: " PIKA_APPLICATION_STRING " [options]");

    cmdline.add_options()("no-mpi-pool", pika::program_options::bool_switch(),
        "Disable the MPI pool.");

    cmdline.add_options()("in-flight-limit",
        pika::program_options::value<std::uint32_t>()->default_value(
            mpi::detail::get_max_requests_in_flight(mpi::stream_index(-1))),
        "Apply a limit to the number of messages in flight.");

    cmdline.add_options()("localMB",
        pika::program_options::value<std::uint64_t>()->default_value(256),
        "Sets the storage capacity (in MB) on each node.\n"
        "The total storage will be num_ranks * localMB");

    cmdline.add_options()("transferKB",
        pika::program_options::value<double>()->default_value(512),
        "Sets the default block transfer size (in KB).\n"
        "Each put/get IOP will be this size");

    cmdline.add_options()("seconds",
        pika::program_options::value<std::uint32_t>()->default_value(5),
        "The number of seconds to run each iteration for.\n");

    nws_deb<6>.debug(3, "Calling pika::init");
    pika::init_params init_args;
    init_args.desc_cmdline = cmdline;
    // Set the callback to init the thread_pools
    init_args.rp_callback = &init_resource_partitioner_handler;

    auto res = pika::init(pika_main, argc, argv, init_args);
    MPI_Finalize();

    // This test should just run without crashing
    PIKA_TEST(true);
    return res;
}
