//  Copyright (c) 2014-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/init.hpp>
#include <pika/modules/testing.hpp>
#include <pika/parallel/algorithms/uninitialized_default_construct.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "test_utils.hpp"

struct default_constructable
{
    default_constructable()
      : value_(42)
    {
    }
    std::int32_t value_;
};

struct value_constructable
{
    std::int32_t value_;
};

std::size_t const data_size = 10007;

////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_uninitialized_default_construct_n(ExPolicy policy, IteratorTag)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    typedef default_constructable* base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    default_constructable* p = (default_constructable*) std::malloc(
        data_size * sizeof(default_constructable));
    std::memset(
        static_cast<void*>(p), 0xcd, data_size * sizeof(default_constructable));

    pika::uninitialized_default_construct_n(policy, iterator(p), data_size);

    std::size_t count = 0;
    std::for_each(p, p + data_size, [&count](default_constructable v1) {
        PIKA_TEST_EQ(v1.value_, 42);
        ++count;
    });
    PIKA_TEST_EQ(count, data_size);

    std::free(p);
}

template <typename ExPolicy, typename IteratorTag>
void test_uninitialized_default_construct_n_async(ExPolicy policy, IteratorTag)
{
    typedef default_constructable* base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    default_constructable* p = (default_constructable*) std::malloc(
        data_size * sizeof(default_constructable));
    std::memset(
        static_cast<void*>(p), 0xcd, data_size * sizeof(default_constructable));

    auto f =
        pika::uninitialized_default_construct_n(policy, iterator(p), data_size);
    f.wait();

    std::size_t count = 0;
    std::for_each(p, p + data_size, [&count](default_constructable v1) {
        PIKA_TEST_EQ(v1.value_, 42);
        ++count;
    });
    PIKA_TEST_EQ(count, data_size);

    std::free(p);
}

template <typename ExPolicy, typename IteratorTag>
void test_uninitialized_default_construct_n2(ExPolicy policy, IteratorTag)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    typedef value_constructable* base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    value_constructable* p = (value_constructable*) std::malloc(
        data_size * sizeof(value_constructable));
    std::memset(
        static_cast<void*>(p), 0xcd, data_size * sizeof(value_constructable));

    pika::uninitialized_default_construct_n(policy, iterator(p), data_size);

    std::size_t count = 0;
    std::for_each(p, p + data_size, [&count](value_constructable v1) {
        PIKA_TEST_EQ(v1.value_, (std::int32_t) 0xcdcdcdcd);
        ++count;
    });
    PIKA_TEST_EQ(count, data_size);

    std::free(p);
}

template <typename ExPolicy, typename IteratorTag>
void test_uninitialized_default_construct_n_async2(ExPolicy policy, IteratorTag)
{
    typedef value_constructable* base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    value_constructable* p = (value_constructable*) std::malloc(
        data_size * sizeof(value_constructable));
    std::memset(
        static_cast<void*>(p), 0xcd, data_size * sizeof(value_constructable));

    auto f =
        pika::uninitialized_default_construct_n(policy, iterator(p), data_size);
    f.wait();

    std::size_t count = 0;
    std::for_each(p, p + data_size, [&count](value_constructable v1) {
        PIKA_TEST_EQ(v1.value_, (std::int32_t) 0xcdcdcdcd);
        ++count;
    });
    PIKA_TEST_EQ(count, data_size);

    std::free(p);
}

template <typename IteratorTag>
void test_uninitialized_default_construct_n()
{
    using namespace pika::execution;

    test_uninitialized_default_construct_n(seq, IteratorTag());
    test_uninitialized_default_construct_n(par, IteratorTag());
    test_uninitialized_default_construct_n(par_unseq, IteratorTag());

    test_uninitialized_default_construct_n_async(seq(task), IteratorTag());
    test_uninitialized_default_construct_n_async(par(task), IteratorTag());

    test_uninitialized_default_construct_n2(seq, IteratorTag());
    test_uninitialized_default_construct_n2(par, IteratorTag());
    test_uninitialized_default_construct_n2(par_unseq, IteratorTag());

    test_uninitialized_default_construct_n_async(seq(task), IteratorTag());
    test_uninitialized_default_construct_n_async2(par(task), IteratorTag());
}

void uninitialized_default_construct_n_test()
{
    test_uninitialized_default_construct_n<std::random_access_iterator_tag>();
    test_uninitialized_default_construct_n<std::forward_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_uninitialized_default_construct_n_exception(
    ExPolicy policy, IteratorTag)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    typedef test::count_instances_v<default_constructable> data_type;
    typedef data_type* base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    data_type* p = (data_type*) std::malloc(data_size * sizeof(data_type));
    std::memset(static_cast<void*>(p), 0xcd, data_size * sizeof(data_type));

    std::atomic<std::size_t> throw_after(std::rand() % data_size);    //-V104
    std::size_t throw_after_ = throw_after.load();

    data_type::instance_count.store(0);
    data_type::max_instance_count.store(0);

    bool caught_exception = false;
    try
    {
        pika::uninitialized_default_construct_n(policy,
            decorated_iterator(p,
                [&throw_after]() {
                    if (throw_after-- == 0)
                        throw std::runtime_error("test");
                }),
            data_size);
        PIKA_TEST(false);
    }
    catch (pika::exception_list const& e)
    {
        caught_exception = true;
        test::test_num_exceptions<ExPolicy, IteratorTag>::call(policy, e);
    }
    catch (...)
    {
        PIKA_TEST(false);
    }

    PIKA_TEST(caught_exception);
    PIKA_TEST_EQ(test::count_instances::instance_count.load(), std::size_t(0));
    PIKA_TEST_LTE(throw_after_, data_type::max_instance_count.load());

    std::free(p);
}

template <typename ExPolicy, typename IteratorTag>
void test_uninitialized_default_construct_n_exception_async(
    ExPolicy policy, IteratorTag)
{
    typedef test::count_instances_v<default_constructable> data_type;
    typedef data_type* base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    data_type* p = (data_type*) std::malloc(data_size * sizeof(data_type));
    std::memset(static_cast<void*>(p), 0xcd, data_size * sizeof(data_type));

    std::atomic<std::size_t> throw_after(std::rand() % data_size);    //-V104
    std::size_t throw_after_ = throw_after.load();

    data_type::instance_count.store(0);
    data_type::max_instance_count.store(0);

    bool caught_exception = false;
    bool returned_from_algorithm = false;
    try
    {
        auto f = pika::uninitialized_default_construct_n(policy,
            decorated_iterator(p,
                [&throw_after]() {
                    if (throw_after-- == 0)
                        throw std::runtime_error("test");
                }),
            data_size);

        returned_from_algorithm = true;
        f.get();

        PIKA_TEST(false);
    }
    catch (pika::exception_list const& e)
    {
        caught_exception = true;
        test::test_num_exceptions<ExPolicy, IteratorTag>::call(policy, e);
    }
    catch (...)
    {
        PIKA_TEST(false);
    }

    PIKA_TEST(caught_exception);
    PIKA_TEST(returned_from_algorithm);
    PIKA_TEST_EQ(test::count_instances::instance_count.load(), std::size_t(0));
    PIKA_TEST_LTE(throw_after_, data_type::max_instance_count.load());

    std::free(p);
}

template <typename IteratorTag>
void test_uninitialized_default_construct_n_exception()
{
    using namespace pika::execution;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_uninitialized_default_construct_n_exception(seq, IteratorTag());
    test_uninitialized_default_construct_n_exception(par, IteratorTag());

    test_uninitialized_default_construct_n_exception_async(
        seq(task), IteratorTag());
    test_uninitialized_default_construct_n_exception_async(
        par(task), IteratorTag());
}

void uninitialized_default_construct_n_exception_test()
{
    test_uninitialized_default_construct_n_exception<
        std::random_access_iterator_tag>();
    test_uninitialized_default_construct_n_exception<
        std::forward_iterator_tag>();
}

////////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_uninitialized_default_construct_n_bad_alloc(
    ExPolicy policy, IteratorTag)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    typedef test::count_instances_v<default_constructable> data_type;
    typedef data_type* base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    data_type* p = (data_type*) std::malloc(data_size * sizeof(data_type));
    std::memset(static_cast<void*>(p), 0xcd, data_size * sizeof(data_type));

    std::atomic<std::size_t> throw_after(std::rand() % data_size);    //-V104
    std::size_t throw_after_ = throw_after.load();

    data_type::instance_count.store(0);
    data_type::max_instance_count.store(0);

    bool caught_bad_alloc = false;
    try
    {
        pika::uninitialized_default_construct_n(policy,
            decorated_iterator(p,
                [&throw_after]() {
                    if (throw_after-- == 0)
                        throw std::bad_alloc();
                }),
            data_size);

        PIKA_TEST(false);
    }
    catch (std::bad_alloc const&)
    {
        caught_bad_alloc = true;
    }
    catch (...)
    {
        PIKA_TEST(false);
    }

    PIKA_TEST(caught_bad_alloc);
    PIKA_TEST_EQ(test::count_instances::instance_count.load(), std::size_t(0));
    PIKA_TEST_LTE(throw_after_, data_type::max_instance_count.load());

    std::free(p);
}

template <typename ExPolicy, typename IteratorTag>
void test_uninitialized_default_construct_n_bad_alloc_async(
    ExPolicy policy, IteratorTag)
{
    typedef test::count_instances_v<default_constructable> data_type;
    typedef data_type* base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    data_type* p = (data_type*) std::malloc(data_size * sizeof(data_type));
    std::memset(static_cast<void*>(p), 0xcd, data_size * sizeof(data_type));

    std::atomic<std::size_t> throw_after(std::rand() % data_size);    //-V104
    std::size_t throw_after_ = throw_after.load();

    data_type::instance_count.store(0);
    data_type::max_instance_count.store(0);

    bool caught_bad_alloc = false;
    bool returned_from_algorithm = false;
    try
    {
        auto f = pika::uninitialized_default_construct_n(policy,
            decorated_iterator(p,
                [&throw_after]() {
                    if (throw_after-- == 0)
                        throw std::bad_alloc();
                }),
            data_size);

        returned_from_algorithm = true;
        f.get();

        PIKA_TEST(false);
    }
    catch (std::bad_alloc const&)
    {
        caught_bad_alloc = true;
    }
    catch (...)
    {
        PIKA_TEST(false);
    }

    PIKA_TEST(caught_bad_alloc);
    PIKA_TEST(returned_from_algorithm);
    PIKA_TEST_EQ(test::count_instances::instance_count.load(), std::size_t(0));
    PIKA_TEST_LTE(throw_after_, data_type::max_instance_count.load());

    std::free(p);
}

template <typename IteratorTag>
void test_uninitialized_default_construct_n_bad_alloc()
{
    using namespace pika::execution;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_uninitialized_default_construct_n_bad_alloc(seq, IteratorTag());
    test_uninitialized_default_construct_n_bad_alloc(par, IteratorTag());

    test_uninitialized_default_construct_n_bad_alloc_async(
        seq(task), IteratorTag());
    test_uninitialized_default_construct_n_bad_alloc_async(
        par(task), IteratorTag());
}

void uninitialized_default_construct_n_bad_alloc_test()
{
    test_uninitialized_default_construct_n_bad_alloc<
        std::random_access_iterator_tag>();
    test_uninitialized_default_construct_n_bad_alloc<
        std::forward_iterator_tag>();
}

int pika_main(pika::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int) std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    uninitialized_default_construct_n_test();
    uninitialized_default_construct_n_exception_test();
    uninitialized_default_construct_n_bad_alloc_test();
    return pika::finalize();
}

int main(int argc, char* argv[])
{
    // add command line option which controls the random number generator seed
    using namespace pika::program_options;
    options_description desc_commandline(
        "Usage: " PIKA_APPLICATION_STRING " [options]");

    desc_commandline.add_options()("seed,s", value<unsigned int>(),
        "the random number generator seed to use for this run");

    // By default this test should run on all available cores
    std::vector<std::string> const cfg = {"pika.os_threads=all"};

    // Initialize and run pika
    pika::init_params init_args;
    init_args.desc_cmdline = desc_commandline;
    init_args.cfg = cfg;

    PIKA_TEST_EQ_MSG(pika::init(pika_main, argc, argv, init_args), 0,
        "pika main exited with non-zero status");

    return pika::util::report_errors();
}
