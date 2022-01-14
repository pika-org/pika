//  Copyright (c) 2014-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/init.hpp>
#include <pika/modules/testing.hpp>
#include <pika/parallel/algorithms/destroy.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "test_utils.hpp"

std::atomic<std::size_t> destruct_count(0);
unsigned int seed = std::random_device{}();
std::mt19937 gen(seed);

struct destructable
{
    destructable()
      : value_(0)
    {
    }

    ~destructable()
    {
        ++destruct_count;
    }

    std::uint32_t value_;
};

std::size_t const data_size = 10007;

////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_destroy_n(IteratorTag)
{
    typedef destructable* base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    destructable* p =
        (destructable*) std::malloc(data_size * sizeof(destructable));

    // value-initialize data in array
    std::for_each(p, p + data_size, [](destructable& d) {
        ::new (static_cast<void*>(std::addressof(d))) destructable;
    });

    destruct_count.store(0);

    pika::destroy_n(iterator(p), data_size);

    PIKA_TEST_EQ(destruct_count.load(), data_size);

    std::free(p);
}

template <typename ExPolicy, typename IteratorTag>
void test_destroy_n(ExPolicy policy, IteratorTag)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    typedef destructable* base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    destructable* p =
        (destructable*) std::malloc(data_size * sizeof(destructable));

    // value-initialize data in array
    std::for_each(p, p + data_size, [](destructable& d) {
        ::new (static_cast<void*>(std::addressof(d))) destructable;
    });

    destruct_count.store(0);

    pika::destroy_n(std::forward<ExPolicy>(policy), iterator(p), data_size);

    PIKA_TEST_EQ(destruct_count.load(), data_size);

    std::free(p);
}

template <typename ExPolicy, typename IteratorTag>
void test_destroy_n_async(ExPolicy&& policy, IteratorTag)
{
    typedef destructable* base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    destructable* p =
        (destructable*) std::malloc(data_size * sizeof(destructable));

    // value-initialize data in array
    std::for_each(p, p + data_size, [](destructable& d) {
        ::new (static_cast<void*>(std::addressof(d))) destructable;
    });

    destruct_count.store(0);

    auto f =
        pika::destroy_n(std::forward<ExPolicy>(policy), iterator(p), data_size);
    f.wait();

    PIKA_TEST_EQ(destruct_count.load(), data_size);

    std::free(p);
}

template <typename IteratorTag>
void test_destroy_n()
{
    test_destroy_n(IteratorTag());

    test_destroy_n(pika::execution::seq, IteratorTag());
    test_destroy_n(pika::execution::par, IteratorTag());
    test_destroy_n(pika::execution::par_unseq, IteratorTag());

    test_destroy_n_async(
        pika::execution::seq(pika::execution::task), IteratorTag());
    test_destroy_n_async(
        pika::execution::par(pika::execution::task), IteratorTag());
}

void destroy_n_test()
{
    test_destroy_n<std::random_access_iterator_tag>();
    test_destroy_n<std::forward_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_destroy_n_exception(IteratorTag)
{
    typedef test::count_instances_v<destructable> data_type;
    typedef data_type* base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    data_type* p = (data_type*) std::malloc(data_size * sizeof(data_type));

    data_type::instance_count.store(0);
    data_type::max_instance_count.store(0);

    // value-initialize data in array
    std::for_each(p, p + data_size, [](data_type& d) {
        ::new (static_cast<void*>(std::addressof(d))) data_type;
    });

    PIKA_TEST_EQ(data_type::instance_count.load(), data_size);

    std::uniform_int_distribution<> dis(0, data_size - 1);
    std::atomic<std::size_t> throw_after(dis(gen));    //-V104
    std::size_t throw_after_ = throw_after.load();

    bool caught_exception = false;
    try
    {
        pika::destroy_n(decorated_iterator(p,
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
        test::test_num_exceptions<pika::execution::sequenced_policy,
            IteratorTag>::call(pika::execution::seq, e);
    }
    catch (...)
    {
        PIKA_TEST(false);
    }

    PIKA_TEST(caught_exception);
    PIKA_TEST_LTE(data_type::instance_count.load(),
        std::size_t(data_size - throw_after_));

    std::free(p);
}

template <typename ExPolicy, typename IteratorTag>
void test_destroy_n_exception(ExPolicy&& policy, IteratorTag)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    typedef test::count_instances_v<destructable> data_type;
    typedef data_type* base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    data_type* p = (data_type*) std::malloc(data_size * sizeof(data_type));

    data_type::instance_count.store(0);
    data_type::max_instance_count.store(0);

    // value-initialize data in array
    std::for_each(p, p + data_size, [](data_type& d) {
        ::new (static_cast<void*>(std::addressof(d))) data_type;
    });

    PIKA_TEST_EQ(data_type::instance_count.load(), data_size);

    std::uniform_int_distribution<> dis(0, data_size - 1);
    std::atomic<std::size_t> throw_after(dis(gen));    //-V104
    std::size_t throw_after_ = throw_after.load();

    bool caught_exception = false;
    try
    {
        pika::destroy_n(policy,
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
    PIKA_TEST_LTE(data_type::instance_count.load(),
        std::size_t(data_size - throw_after_));

    std::free(p);
}

template <typename ExPolicy, typename IteratorTag>
void test_destroy_n_exception_async(ExPolicy&& policy, IteratorTag)
{
    typedef test::count_instances_v<destructable> data_type;
    typedef data_type* base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    data_type* p = (data_type*) std::malloc(data_size * sizeof(data_type));

    data_type::instance_count.store(0);
    data_type::max_instance_count.store(0);

    // value-initialize data in array
    std::for_each(p, p + data_size, [](data_type& d) {
        ::new (static_cast<void*>(std::addressof(d))) data_type;
    });

    PIKA_TEST_EQ(data_type::instance_count.load(), data_size);

    std::uniform_int_distribution<> dis(0, data_size - 1);
    std::atomic<std::size_t> throw_after(dis(gen));    //-V104
    std::size_t throw_after_ = throw_after.load();

    bool caught_exception = false;
    bool returned_from_algorithm = false;
    try
    {
        auto f = pika::destroy_n(policy,
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
    PIKA_TEST_LTE(data_type::instance_count.load(),
        std::size_t(data_size - throw_after_));

    std::free(p);
}

template <typename IteratorTag>
void test_destroy_n_exception()
{
    test_destroy_n_exception(IteratorTag());

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_destroy_n_exception(pika::execution::seq, IteratorTag());
    test_destroy_n_exception(pika::execution::par, IteratorTag());

    test_destroy_n_exception_async(
        pika::execution::seq(pika::execution::task), IteratorTag());
    test_destroy_n_exception_async(
        pika::execution::par(pika::execution::task), IteratorTag());
}

void destroy_n_exception_test()
{
    test_destroy_n_exception<std::random_access_iterator_tag>();
    test_destroy_n_exception<std::forward_iterator_tag>();
}

////////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_destroy_n_bad_alloc(ExPolicy&& policy, IteratorTag)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    typedef test::count_instances_v<destructable> data_type;
    typedef data_type* base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    data_type* p = (data_type*) std::malloc(data_size * sizeof(data_type));

    data_type::instance_count.store(0);
    data_type::max_instance_count.store(0);

    // value-initialize data in array
    std::for_each(p, p + data_size, [](data_type& d) {
        ::new (static_cast<void*>(std::addressof(d))) data_type;
    });

    PIKA_TEST_EQ(data_type::instance_count.load(), data_size);

    std::uniform_int_distribution<> dis(0, data_size - 1);
    std::atomic<std::size_t> throw_after(dis(gen));    //-V104
    std::size_t throw_after_ = throw_after.load();

    bool caught_bad_alloc = false;
    try
    {
        pika::destroy_n(policy,
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
    PIKA_TEST_LTE(data_type::instance_count.load(),
        std::size_t(data_size - throw_after_));

    std::free(p);
}

template <typename ExPolicy, typename IteratorTag>
void test_destroy_n_bad_alloc_async(ExPolicy&& policy, IteratorTag)
{
    typedef test::count_instances_v<destructable> data_type;
    typedef data_type* base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    data_type* p = (data_type*) std::malloc(data_size * sizeof(data_type));

    data_type::instance_count.store(0);
    data_type::max_instance_count.store(0);

    // value-initialize data in array
    std::for_each(p, p + data_size, [](data_type& d) {
        ::new (static_cast<void*>(std::addressof(d))) data_type;
    });

    PIKA_TEST_EQ(data_type::instance_count.load(), data_size);

    std::uniform_int_distribution<> dis(0, data_size - 1);
    std::atomic<std::size_t> throw_after(dis(gen));    //-V104
    std::size_t throw_after_ = throw_after.load();

    bool caught_bad_alloc = false;
    bool returned_from_algorithm = false;
    try
    {
        auto f = pika::destroy_n(policy,
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
    PIKA_TEST_LTE(data_type::instance_count.load(),
        std::size_t(data_size - throw_after_));

    std::free(p);
}

template <typename IteratorTag>
void test_destroy_n_bad_alloc()
{
    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_destroy_n_bad_alloc(pika::execution::seq, IteratorTag());
    test_destroy_n_bad_alloc(pika::execution::par, IteratorTag());

    test_destroy_n_bad_alloc_async(
        pika::execution::seq(pika::execution::task), IteratorTag());
    test_destroy_n_bad_alloc_async(
        pika::execution::par(pika::execution::task), IteratorTag());
}

void destroy_n_bad_alloc_test()
{
    test_destroy_n_bad_alloc<std::random_access_iterator_tag>();
    test_destroy_n_bad_alloc<std::forward_iterator_tag>();
}

int pika_main(pika::program_options::variables_map& vm)
{
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    gen.seed(seed);

    destroy_n_test();
    destroy_n_exception_test();
    destroy_n_bad_alloc_test();
    return pika::local::finalize();
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
    pika::local::init_params init_args;
    init_args.desc_cmdline = desc_commandline;
    init_args.cfg = cfg;

    PIKA_TEST_EQ_MSG(pika::local::init(pika_main, argc, argv, init_args), 0,
        "pika main exited with non-zero status");

    return pika::util::report_errors();
}
