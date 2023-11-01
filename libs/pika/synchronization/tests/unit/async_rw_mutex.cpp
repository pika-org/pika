//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/assert.hpp>
#include <pika/async_rw_mutex.hpp>
#include <pika/execution.hpp>
#include <pika/init.hpp>
#include <pika/mutex.hpp>
#include <pika/testing.hpp>

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

using pika::execution::experimental::async_rw_mutex;
using pika::execution::experimental::execute;
using pika::execution::experimental::start_detached;
using pika::execution::experimental::then;
using pika::execution::experimental::thread_pool_scheduler;
using pika::execution::experimental::transfer;
using pika::execution::experimental::when_all;
using pika::this_thread::experimental::sync_wait;

unsigned int seed = std::random_device{}();

///////////////////////////////////////////////////////////////////////////////
// Custom type with a more restricted interface in the base class. Used for
// testing that the read-only type of async_rw_mutex can be customized.
class mytype_base
{
protected:
    std::size_t x = 0;

public:
    mytype_base() = default;
    mytype_base(mytype_base&&) = default;
    mytype_base& operator=(mytype_base&&) = default;
    mytype_base(mytype_base const&) = delete;
    mytype_base& operator=(mytype_base const&) = delete;

    std::size_t const& read() const { return x; }
};

class mytype : public mytype_base
{
public:
    mytype() = default;
    mytype(mytype&&) = default;
    mytype& operator=(mytype&&) = default;
    mytype(mytype const&) = delete;
    mytype& operator=(mytype const&) = delete;

    std::size_t& readwrite() { return x; }
};

// Struct with call operators used for checking that the correct types are sent
// from the async_rw_mutex senders.
struct checker
{
    const bool expect_readonly;
    const std::size_t expected_predecessor_value;
    std::atomic<std::size_t>& count;
    const std::size_t count_min;
    const std::size_t count_max = count_min;

    // Access types are differently tagged for read-only and read-write access.
    using void_read_access_type = typename async_rw_mutex<void>::read_access_type;
    using void_readwrite_access_type = typename async_rw_mutex<void>::readwrite_access_type;

    void operator()(void_read_access_type)
    {
        PIKA_ASSERT(expect_readonly);
        PIKA_TEST_RANGE(++count, count_min, count_max);
    }

    void operator()(void_readwrite_access_type)
    {
        PIKA_ASSERT(!expect_readonly);
        PIKA_TEST_RANGE(++count, count_min, count_max);
    }

    // Non-void access types must be convertible to (const) references of the
    // types on which the async_rw_mutex is templated.
    using size_t_read_access_type = typename async_rw_mutex<std::size_t>::read_access_type;
    using size_t_readwrite_access_type =
        typename async_rw_mutex<std::size_t>::readwrite_access_type;

    void operator()(size_t_read_access_type x)
    {
        PIKA_TEST(expect_readonly);
        PIKA_TEST_EQ(x.get(), expected_predecessor_value);
        PIKA_TEST_RANGE(++count, count_min, count_max);
    }

    void operator()(size_t_readwrite_access_type x)
    {
        PIKA_ASSERT(!expect_readonly);
        PIKA_TEST_EQ(x.get(), expected_predecessor_value);
        PIKA_TEST_RANGE(++count, count_min, count_max);
        ++x.get();
    }

    // Non-void access types must be convertible to (const) references of the
    // types on which the async_rw_mutex is templated.
    using mytype_read_access_type = typename async_rw_mutex<mytype, mytype_base>::read_access_type;
    using mytype_readwrite_access_type =
        typename async_rw_mutex<mytype, mytype_base>::readwrite_access_type;

    void operator()(mytype_read_access_type x)
    {
        PIKA_TEST(expect_readonly);
        PIKA_TEST_EQ(x.get().read(), expected_predecessor_value);
        PIKA_TEST_RANGE(++count, count_min, count_max);
    }

    void operator()(mytype_readwrite_access_type x)
    {
        PIKA_ASSERT(!expect_readonly);
        PIKA_TEST_EQ(x.get().read(), expected_predecessor_value);
        PIKA_TEST_RANGE(++count, count_min, count_max);
        ++(x.get().readwrite());
    }
};

template <typename Executor, typename Senders>
void submit_senders(Executor&& exec, Senders& senders)
{
    for (auto& sender : senders)
    {
        execute(exec, [sender = std::move(sender)]() mutable { sync_wait(std::move(sender)); });
    }
}

template <typename ReadWriteT, typename ReadT = ReadWriteT>
void test_single_read_access(async_rw_mutex<ReadWriteT, ReadT> rwm)
{
    std::atomic<bool> called{false};
    sync_wait(rwm.read() | then([&](auto) { called = true; }));
    PIKA_TEST(called);
}

template <typename ReadWriteT, typename ReadT = ReadWriteT>
void test_single_readwrite_access(async_rw_mutex<ReadWriteT, ReadT> rwm)
{
    std::atomic<bool> called{false};
    sync_wait(rwm.readwrite() | then([&](auto) { called = true; }));
    PIKA_TEST(called);
}

template <typename ReadWriteT, typename ReadT = ReadWriteT>
void test_moved(async_rw_mutex<ReadWriteT, ReadT> rwm)
{
    // The destructor of an empty async_rw_mutex should not attempt to keep any
    // values alive
    auto rwm2 = std::move(rwm);
    std::atomic<bool> called{false};
    sync_wait(rwm2.read() | then([&](auto) { called = true; }));
    PIKA_TEST(called);
}

template <typename ReadWriteT, typename ReadT = ReadWriteT>
void test_multiple_accesses(async_rw_mutex<ReadWriteT, ReadT> rwm, std::size_t iterations)
{
    thread_pool_scheduler exec{};

    std::atomic<std::size_t> count{0};

    // Read-only and read-write access return senders of different types
    using r_sender_type =
        std::decay_t<decltype(rwm.read() | transfer(exec) | then(checker{true, 0, count, 0}))>;
    using rw_sender_type = std::decay_t<decltype(rwm.readwrite() | transfer(exec) |
        then(checker{false, 0, count, 0}))>;
    std::vector<r_sender_type> r_senders;
    std::vector<rw_sender_type> rw_senders;

    std::mt19937 r(seed);
    std::uniform_int_distribution<std::size_t> d_senders(1, 10);

    std::size_t expected_count = 0;
    std::size_t expected_predecessor_count = 0;

    auto sender_helper = [&](bool readonly) {
        std::size_t const num_senders = d_senders(r);
        std::size_t const min_expected_count = expected_count + 1;
        std::size_t const max_expected_count = expected_count + num_senders;
        expected_count += num_senders;
        for (std::size_t j = 0; j < num_senders; ++j)
        {
            if (readonly)
            {
                r_senders.push_back(rwm.read() | transfer(exec) |
                    then(checker{readonly, expected_predecessor_count, count, min_expected_count,
                        max_expected_count}));
            }
            else
            {
                rw_senders.push_back(rwm.readwrite() | transfer(exec) |
                    then(checker{readonly, expected_predecessor_count, count, min_expected_count,
                        max_expected_count}));
                // Only read-write access is allowed to change the value
                ++expected_predecessor_count;
            }
        }
    };

    for (std::size_t i = 0; i < iterations; ++i)
    {
        // Alternate between read-only and read-write access
        sender_helper(true);
        sender_helper(false);
    }

    // Asynchronously submit the senders
    submit_senders(exec, r_senders);
    submit_senders(exec, rw_senders);

    // The destructor does not block, so we block here manually
    sync_wait(rwm.readwrite());
}

template <typename ReadWriteT, typename ReadT = ReadWriteT>
void test_shared_state_deadlock(async_rw_mutex<ReadWriteT, ReadT> rwm)
{
    // This tests that when synchronously waiting for read-only access, the
    // shared state from the previous read-write access is not kept alive
    // unnecessarily, thus stopping the read-only access from happening.
    //
    // This test should simply not deadlock. If the previous state is held alive
    // by the async_rw_mutex itself the second sync_wait will never complete.
    bool readwrite_access = false;
    bool read_access = false;
    sync_wait(rwm.readwrite() | then([&](auto) { readwrite_access = true; }));
    sync_wait(rwm.read() | then([&](auto) { read_access = true; }));
    PIKA_TEST(readwrite_access);
    PIKA_TEST(read_access);
}

template <typename ReadWriteT, typename ReadT = ReadWriteT>
void test_read_sender_copyable(async_rw_mutex<ReadWriteT, ReadT> rwm)
{
    std::size_t read_accesses = 0;
    auto f = [&](auto) { ++read_accesses; };
    sync_wait(rwm.read() | then(f));
    auto s = rwm.read();
    sync_wait(s | then(f));
    sync_wait(s | then(f));
    sync_wait(std::move(s) | then(f));
    PIKA_TEST_EQ(read_accesses, std::size_t(4));
}

template <typename ReadWriteT, typename ReadT = ReadWriteT>
void test_multiple_when_all(async_rw_mutex<ReadWriteT, ReadT> rwm)
{
    {
        auto s1 = rwm.readwrite() | then([](auto) {});
        auto s2 = rwm.readwrite() | then([](auto) {});
        auto s3 = rwm.readwrite() | then([](auto) {});
        sync_wait(when_all(std::move(s1), std::move(s2), std::move(s3)));
    }

    {
        auto s1 = rwm.readwrite() | then([](auto) {});
        auto s2 = rwm.readwrite() | then([](auto) {});
        auto s3 = rwm.readwrite() | then([](auto) {});
        sync_wait(when_all(std::move(s3), std::move(s2), std::move(s1)));
    }

    {
        auto s1 = rwm.readwrite() | then([](auto) {});
        auto s2 = rwm.readwrite() | then([](auto) {});
        auto s3 = rwm.readwrite() | then([](auto) {});
        sync_wait(when_all(std::move(s3), std::move(s1), std::move(s2)));
    }
}

///////////////////////////////////////////////////////////////////////////////
int pika_main(pika::program_options::variables_map& vm)
{
    if (vm.count("seed")) { seed = vm["seed"].as<unsigned int>(); }

    test_single_read_access(async_rw_mutex<void>{});
    test_single_read_access(async_rw_mutex<std::size_t>{0});
    test_single_read_access(async_rw_mutex<mytype, mytype_base>{mytype{}});

    test_single_readwrite_access(async_rw_mutex<void>{});
    test_single_readwrite_access(async_rw_mutex<std::size_t>{0});
    test_single_readwrite_access(async_rw_mutex<mytype, mytype_base>{mytype{}});

    test_moved(async_rw_mutex<void>{});
    test_moved(async_rw_mutex<std::size_t>{0});
    test_moved(async_rw_mutex<mytype, mytype_base>{mytype{}});

#if defined(PIKA_HAVE_VERIFY_LOCKS)
    constexpr std::size_t iterations = 50;
#else
    constexpr std::size_t iterations = 1000;
#endif
    test_multiple_accesses(async_rw_mutex<void>{}, iterations);
    test_multiple_accesses(async_rw_mutex<std::size_t>{0}, iterations);
    test_multiple_accesses(async_rw_mutex<mytype, mytype_base>{mytype{}}, iterations);

    test_shared_state_deadlock(async_rw_mutex<void>{});
    test_shared_state_deadlock(async_rw_mutex<std::size_t>{0});
    test_shared_state_deadlock(async_rw_mutex<mytype, mytype_base>{mytype{}});

    test_read_sender_copyable(async_rw_mutex<void>{});
    test_read_sender_copyable(async_rw_mutex<std::size_t>{0});
    test_read_sender_copyable(async_rw_mutex<mytype, mytype_base>{mytype{}});

    test_multiple_when_all(async_rw_mutex<void>{});
    test_multiple_when_all(async_rw_mutex<std::size_t>{0});
    test_multiple_when_all(async_rw_mutex<mytype, mytype_base>{mytype{}});

    pika::finalize();
    return EXIT_SUCCESS;
}

int main(int argc, char* argv[])
{
    pika::init_params i;
    pika::program_options::options_description desc_cmdline(
        "usage: " PIKA_APPLICATION_STRING " [options]");
    desc_cmdline.add_options()("seed,s", pika::program_options::value<unsigned int>(),
        "the random number generator seed to use for this run");
    i.desc_cmdline = desc_cmdline;

    PIKA_TEST_EQ(pika::init(pika_main, argc, argv, i), 0);
    return 0;
}
