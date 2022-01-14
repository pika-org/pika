//  Copyright (c) 2007-2015 Hartmut Kaiser
//  Copyright (c) 2021 Chuanqiu He
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/execution/traits/is_execution_policy.hpp>
#include <pika/iterator_support/iterator_range.hpp>
#include <pika/iterator_support/tests/iter_sent.hpp>
#include <pika/local/init.hpp>
#include <pika/modules/testing.hpp>
#include <pika/parallel/container_algorithms/rotate.hpp>

#include <cstddef>
#include <iostream>
#include <iterator>
#include <numeric>
#include <string>
#include <vector>

#include "test_utils.hpp"

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_rotate_copy_sent(IteratorTag)
{
    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d1(c.size());
    std::vector<std::size_t> d2(c.size());    //-V656

    std::iota(std::begin(c), std::end(c), std::rand());

    std::size_t mid_pos = std::rand() % c.size();

    auto mid = std::begin(c);
    std::advance(mid, mid_pos);

    pika::ranges::rotate_copy(std::begin(c), mid,
        sentinel<std::size_t>{*(std::end(c) - 1)}, std::begin(d1));

    auto mid_base = std::begin(c);
    std::advance(mid_base, mid_pos);

    std::rotate_copy(std::begin(c), mid_base, std::end(c) - 1, std::begin(d2));

    std::size_t count = 0;
    PIKA_TEST(std::equal(std::begin(d1), std::end(d1) - 1, std::begin(d2),
        [&count](std::size_t v1, std::size_t v2) -> bool {
            PIKA_TEST_EQ(v1, v2);
            ++count;
            return v1 == v2;
        }));
    PIKA_TEST_EQ(count, d1.size() - 1);
}

template <typename ExPolicy, typename IteratorTag>
void test_rotate_copy_sent(ExPolicy policy, IteratorTag)
{
    static_assert(pika::is_execution_policy_v<ExPolicy>,
        "pika::is_execution_policy<ExPolicy>::value");

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d1(c.size());
    std::vector<std::size_t> d2(c.size());    //-V656

    std::iota(std::begin(c), std::end(c), std::rand());

    std::size_t mid_pos = std::rand() % c.size();

    auto mid = std::begin(c);
    std::advance(mid, mid_pos);

    pika::ranges::rotate_copy(policy, std::begin(c), mid,
        sentinel<std::size_t>{*(std::end(c) - 1)}, std::begin(d1));

    auto mid_base = std::begin(c);
    std::advance(mid_base, mid_pos);

    std::rotate_copy(std::begin(c), mid_base, std::end(c) - 1, std::begin(d2));

    std::size_t count = 0;
    PIKA_TEST(std::equal(std::begin(d1), std::end(d1) - 1, std::begin(d2),
        [&count](std::size_t v1, std::size_t v2) -> bool {
            PIKA_TEST_EQ(v1, v2);
            ++count;
            return v1 == v2;
        }));
    PIKA_TEST_EQ(count, d1.size() - 1);
}

//add test w/o ExPolicy
template <typename IteratorTag>
void test_rotate_copy(IteratorTag)
{
    using base_iterator = std::vector<std::size_t>::iterator;
    using iterator = test::test_iterator<base_iterator, IteratorTag>;
    using test_vector =
        test::test_container<std::vector<std::size_t>, IteratorTag>;

    test_vector c(10007);
    std::vector<std::size_t> d1(c.size());
    std::vector<std::size_t> d2(c.size());    //-V656

    std::iota(std::begin(c.base()), std::end(c.base()), std::rand());

    std::size_t mid_pos = std::rand() % c.size();

    auto mid = std::begin(c);
    std::advance(mid, mid_pos);

    pika::ranges::rotate_copy(c, iterator(mid), std::begin(d1));

    auto mid_base = std::begin(c.base());
    std::advance(mid_base, mid_pos);

    std::rotate_copy(
        std::begin(c.base()), mid_base, std::end(c.base()), std::begin(d2));

    std::size_t count = 0;
    PIKA_TEST(std::equal(std::begin(d1), std::end(d1), std::begin(d2),
        [&count](std::size_t v1, std::size_t v2) -> bool {
            PIKA_TEST_EQ(v1, v2);
            ++count;
            return v1 == v2;
        }));
    PIKA_TEST_EQ(count, d1.size());
}

template <typename ExPolicy, typename IteratorTag>
void test_rotate_copy(ExPolicy policy, IteratorTag)
{
    static_assert(pika::is_execution_policy_v<ExPolicy>,
        "pika::is_execution_policy<ExPolicy>::value");

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d1(c.size());
    std::vector<std::size_t> d2(c.size());    //-V656

    std::iota(std::begin(c), std::end(c), std::rand());

    std::size_t mid_pos = std::rand() % c.size();

    auto mid = std::begin(c);
    std::advance(mid, mid_pos);

    pika::ranges::rotate_copy(policy, c, mid, std::begin(d1));

    auto mid_base = std::begin(c);
    std::advance(mid_base, mid_pos);

    std::rotate_copy(std::begin(c), mid_base, std::end(c), std::begin(d2));

    std::size_t count = 0;
    PIKA_TEST(std::equal(std::begin(d1), std::end(d1), std::begin(d2),
        [&count](std::size_t v1, std::size_t v2) -> bool {
            PIKA_TEST_EQ(v1, v2);
            ++count;
            return v1 == v2;
        }));
    PIKA_TEST_EQ(count, d1.size());
}

template <typename ExPolicy, typename IteratorTag>
void test_rotate_copy_async(ExPolicy p, IteratorTag)
{
    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d1(c.size());
    std::vector<std::size_t> d2(c.size());    //-V656

    std::iota(std::begin(c), std::end(c), std::rand());

    std::size_t mid_pos = std::rand() % c.size();

    auto mid = std::begin(c);
    std::advance(mid, mid_pos);

    auto f = pika::ranges::rotate_copy(p, c, mid, std::begin(d1));
    f.wait();

    auto mid_base = std::begin(c);
    std::advance(mid_base, mid_pos);

    std::rotate_copy(std::begin(c), mid_base, std::end(c), std::begin(d2));

    std::size_t count = 0;
    PIKA_TEST(std::equal(std::begin(d1), std::end(d1), std::begin(d2),
        [&count](std::size_t v1, std::size_t v2) -> bool {
            PIKA_TEST_EQ(v1, v2);
            ++count;
            return v1 == v2;
        }));
    PIKA_TEST_EQ(count, d1.size());
}

template <typename IteratorTag>
void test_rotate_copy()
{
    using namespace pika::execution;
    test_rotate_copy_sent(IteratorTag());
    test_rotate_copy_sent(seq, IteratorTag());
    test_rotate_copy_sent(par, IteratorTag());
    test_rotate_copy_sent(par_unseq, IteratorTag());

    test_rotate_copy(seq, IteratorTag());
    test_rotate_copy(par, IteratorTag());
    test_rotate_copy(par_unseq, IteratorTag());

    test_rotate_copy_async(seq(task), IteratorTag());
    test_rotate_copy_async(par(task), IteratorTag());
}

void rotate_copy_test()
{
    test_rotate_copy<std::random_access_iterator_tag>();
    test_rotate_copy<std::forward_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_rotate_copy_exception(IteratorTag)
{
    using base_iterator = std::vector<std::size_t>::iterator;
    using decorated_iterator =
        test::decorated_iterator<base_iterator, IteratorTag>;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::iota(std::begin(c), std::end(c), std::rand());

    base_iterator mid = std::begin(c);

    // move at least one element to guarantee an exception to be thrown
    std::size_t delta =
        (std::max)(std::rand() % c.size(), std::size_t(1));    //-V104
    std::advance(mid, delta);

    bool caught_exception = false;
    try
    {
        pika::ranges::rotate_copy(
            pika::util::make_iterator_range(
                decorated_iterator(
                    std::begin(c), []() { throw std::runtime_error("test"); }),
                decorated_iterator(std::end(c))),
            decorated_iterator(mid), std::begin(d));
        PIKA_TEST(false);
    }
    catch (pika::exception_list const& e)
    {
        caught_exception = true;
    }
    catch (...)
    {
        PIKA_TEST(false);
    }

    PIKA_TEST(caught_exception);
}

template <typename ExPolicy, typename IteratorTag>
void test_rotate_copy_exception(ExPolicy policy, IteratorTag)
{
    static_assert(pika::is_execution_policy_v<ExPolicy>,
        "pika::is_execution_policy<ExPolicy>::value");

    using base_iterator = std::vector<std::size_t>::iterator;
    using decorated_iterator =
        test::decorated_iterator<base_iterator, IteratorTag>;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::iota(std::begin(c), std::end(c), std::rand());

    base_iterator mid = std::begin(c);

    // move at least one element to guarantee an exception to be thrown
    std::size_t delta =
        (std::max)(std::rand() % c.size(), std::size_t(1));    //-V104
    std::advance(mid, delta);

    bool caught_exception = false;
    try
    {
        pika::ranges::rotate_copy(policy,
            pika::util::make_iterator_range(
                decorated_iterator(
                    std::begin(c), []() { throw std::runtime_error("test"); }),
                decorated_iterator(std::end(c))),
            decorated_iterator(mid), std::begin(d));
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
}

template <typename ExPolicy, typename IteratorTag>
void test_rotate_copy_exception_async(ExPolicy p, IteratorTag)
{
    using base_iterator = std::vector<std::size_t>::iterator;
    using decorated_iterator =
        test::decorated_iterator<base_iterator, IteratorTag>;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::iota(std::begin(c), std::end(c), std::rand());

    base_iterator mid = std::begin(c);

    // move at least one element to guarantee an exception to be thrown
    std::size_t delta =
        (std::max)(std::rand() % c.size(), std::size_t(1));    //-V104
    std::advance(mid, delta);

    bool caught_exception = false;
    bool returned_from_algorithm = false;
    try
    {
        auto f = pika::ranges::rotate_copy(p,
            pika::util::make_iterator_range(
                decorated_iterator(
                    std::begin(c), []() { throw std::runtime_error("test"); }),
                decorated_iterator(std::end(c))),
            decorated_iterator(mid), std::begin(d));
        returned_from_algorithm = true;
        f.get();

        PIKA_TEST(false);
    }
    catch (pika::exception_list const& e)
    {
        caught_exception = true;
        test::test_num_exceptions<ExPolicy, IteratorTag>::call(p, e);
    }
    catch (...)
    {
        PIKA_TEST(false);
    }

    PIKA_TEST(caught_exception);
    PIKA_TEST(returned_from_algorithm);
}

template <typename IteratorTag>
void test_rotate_copy_exception()
{
    using namespace pika::execution;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_rotate_copy_exception(IteratorTag());
    test_rotate_copy_exception(seq, IteratorTag());
    test_rotate_copy_exception(par, IteratorTag());

    test_rotate_copy_exception_async(seq(task), IteratorTag());
    test_rotate_copy_exception_async(par(task), IteratorTag());
}

void rotate_copy_exception_test()
{
    test_rotate_copy_exception<std::random_access_iterator_tag>();
    test_rotate_copy_exception<std::forward_iterator_tag>();
}

//////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_rotate_copy_bad_alloc(IteratorTag)
{
    using base_iterator = std::vector<std::size_t>::iterator;
    using decorated_iterator =
        test::decorated_iterator<base_iterator, IteratorTag>;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::iota(std::begin(c), std::end(c), std::rand());

    base_iterator mid = std::begin(c);

    // move at least one element to guarantee an exception to be thrown
    std::size_t delta =
        (std::max)(std::rand() % c.size(), std::size_t(1));    //-V104
    std::advance(mid, delta);

    bool caught_bad_alloc = false;
    try
    {
        pika::ranges::rotate_copy(pika::util::make_iterator_range(
                                     decorated_iterator(std::begin(c),
                                         []() { throw std::bad_alloc(); }),
                                     decorated_iterator(std::end(c))),
            decorated_iterator(mid), std::begin(d));
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
}

template <typename ExPolicy, typename IteratorTag>
void test_rotate_copy_bad_alloc(ExPolicy policy, IteratorTag)
{
    static_assert(pika::is_execution_policy_v<ExPolicy>,
        "pika::is_execution_policy<ExPolicy>::value");

    using base_iterator = std::vector<std::size_t>::iterator;
    using decorated_iterator =
        test::decorated_iterator<base_iterator, IteratorTag>;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::iota(std::begin(c), std::end(c), std::rand());

    base_iterator mid = std::begin(c);

    // move at least one element to guarantee an exception to be thrown
    std::size_t delta =
        (std::max)(std::rand() % c.size(), std::size_t(1));    //-V104
    std::advance(mid, delta);

    bool caught_bad_alloc = false;
    try
    {
        pika::ranges::rotate_copy(policy,
            pika::util::make_iterator_range(
                decorated_iterator(
                    std::begin(c), []() { throw std::bad_alloc(); }),
                decorated_iterator(std::end(c))),
            decorated_iterator(mid), std::begin(d));
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
}

template <typename ExPolicy, typename IteratorTag>
void test_rotate_copy_bad_alloc_async(ExPolicy p, IteratorTag)
{
    using base_iterator = std::vector<std::size_t>::iterator;
    using decorated_iterator =
        test::decorated_iterator<base_iterator, IteratorTag>;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::iota(std::begin(c), std::end(c), std::rand());

    base_iterator mid = std::begin(c);

    // move at least one element to guarantee an exception to be thrown
    std::size_t delta =
        (std::max)(std::rand() % c.size(), std::size_t(1));    //-V104
    std::advance(mid, delta);

    bool caught_bad_alloc = false;
    bool returned_from_algorithm = false;
    try
    {
        auto f = pika::ranges::rotate_copy(p,
            pika::util::make_iterator_range(
                decorated_iterator(
                    std::begin(c), []() { throw std::bad_alloc(); }),
                decorated_iterator(std::end(c))),
            decorated_iterator(mid), std::begin(d));
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
}

template <typename IteratorTag>
void test_rotate_copy_bad_alloc()
{
    using namespace pika::execution;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_rotate_copy_bad_alloc(IteratorTag());
    test_rotate_copy_bad_alloc(seq, IteratorTag());
    test_rotate_copy_bad_alloc(par, IteratorTag());

    test_rotate_copy_bad_alloc_async(seq(task), IteratorTag());
    test_rotate_copy_bad_alloc_async(par(task), IteratorTag());
}

void rotate_copy_bad_alloc_test()
{
    test_rotate_copy_bad_alloc<std::random_access_iterator_tag>();
    test_rotate_copy_bad_alloc<std::forward_iterator_tag>();
}

int pika_main(pika::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int) std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    rotate_copy_test();
    rotate_copy_exception_test();
    rotate_copy_bad_alloc_test();
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
