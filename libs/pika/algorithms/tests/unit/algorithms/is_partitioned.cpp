//  Copyright (c) 2015 Daniel Bourgeois
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/init.hpp>
#include <pika/modules/testing.hpp>
#include <pika/parallel/algorithms/is_partitioned.hpp>

#include <cstddef>
#include <iostream>
#include <iterator>
#include <random>
#include <string>
#include <vector>

#include "test_utils.hpp"

////////////////////////////////////////////////////////////////////////////////
int seed = std::random_device{}();
std::mt19937 gen(seed);
std::uniform_int_distribution<> dis(0, 99);

template <typename ExPolicy, typename IteratorTag>
void test_partitioned1(ExPolicy policy, IteratorTag)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    //fill first half of array with even numbers and second half
    //with odd numbers
    std::fill(std::begin(c), std::begin(c) + c.size() / 2, 2 * (dis(gen)));
    std::fill(std::begin(c) + c.size() / 2, std::end(c), 2 * (dis(gen)) + 1);

    bool parted = pika::is_partitioned(policy, iterator(std::begin(c)),
        iterator(std::end(c)), [](std::size_t n) { return n % 2 == 0; });

    PIKA_TEST(parted);
}

template <typename ExPolicy, typename IteratorTag>
void test_partitioned1_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    //fill first half of array with even numbers and second half
    //with odd numbers
    std::fill(std::begin(c), std::begin(c) + c.size() / 2, 2 * (dis(gen)));
    std::fill(std::begin(c) + c.size() / 2, std::end(c), 2 * (dis(gen)) + 1);

    pika::future<bool> f = pika::is_partitioned(p, iterator(std::begin(c)),
        iterator(std::end(c)), [](std::size_t n) { return n % 2 == 0; });
    f.wait();

    PIKA_TEST(f.get());
}

template <typename IteratorTag>
void test_partitioned1()
{
    using namespace pika::execution;
    test_partitioned1(seq, IteratorTag());
    test_partitioned1(par, IteratorTag());
    test_partitioned1(par_unseq, IteratorTag());

    test_partitioned1_async(seq(task), IteratorTag());
    test_partitioned1_async(par(task), IteratorTag());
}

void partitioned_test1()
{
    test_partitioned1<std::random_access_iterator_tag>();
    test_partitioned1<std::forward_iterator_tag>();
}

////////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_partitioned2(ExPolicy policy, IteratorTag)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c_odd(10007);
    //fill all of array with odds
    std::fill(std::begin(c_odd), std::end(c_odd), 2 * (dis(gen)) + 1);
    std::vector<std::size_t> c_even(10007);
    //fill all of array with evens
    std::fill(std::begin(c_even), std::end(c_even), 2 * (dis(gen)));

    bool parted_odd = pika::is_partitioned(policy, iterator(std::begin(c_odd)),
        iterator(std::end(c_odd)), [](std::size_t n) { return n % 2 == 0; });
    bool parted_even = pika::is_partitioned(policy, iterator(std::begin(c_even)),
        iterator(std::end(c_even)), [](std::size_t n) { return n % 2 == 0; });

    PIKA_TEST(parted_odd);
    PIKA_TEST(parted_even);
}

template <typename ExPolicy, typename IteratorTag>
void test_partitioned2_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c_odd(10007);
    //fill all of array with odds
    std::fill(std::begin(c_odd), std::end(c_odd), 2 * (dis(gen)) + 1);
    std::vector<std::size_t> c_even(10007);
    //fill all of array with evens
    std::fill(std::begin(c_even), std::end(c_even), 2 * (dis(gen)));

    pika::future<bool> f_odd = pika::is_partitioned(p,
        iterator(std::begin(c_odd)), iterator(std::end(c_odd)),
        [](std::size_t n) { return n % 2 == 0; });
    pika::future<bool> f_even = pika::is_partitioned(p,
        iterator(std::begin(c_even)), iterator(std::end(c_even)),
        [](std::size_t n) { return n % 2 == 0; });

    f_odd.wait();
    PIKA_TEST(f_odd.get());
    f_even.wait();
    PIKA_TEST(f_even.get());
}

template <typename IteratorTag>
void test_partitioned2()
{
    using namespace pika::execution;
    test_partitioned2(seq, IteratorTag());
    test_partitioned2(par, IteratorTag());
    test_partitioned2(par_unseq, IteratorTag());

    test_partitioned2_async(seq(task), IteratorTag());
    test_partitioned2_async(par(task), IteratorTag());
}

void partitioned_test2()
{
    test_partitioned2<std::random_access_iterator_tag>();
    test_partitioned2<std::forward_iterator_tag>();
}

////////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_partitioned3(ExPolicy policy, IteratorTag)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c_beg(10007);
    //fill first half of array with even numbers and second half
    //with odd numbers
    std::fill(std::begin(c_beg), std::begin(c_beg) + c_beg.size() / 2,
        2 * (dis(gen)));
    std::fill(std::begin(c_beg) + c_beg.size() / 2, std::end(c_beg),
        2 * (dis(gen)) + 1);
    std::vector<size_t> c_end = c_beg;
    //add odd number to the beginning
    c_beg[0] -= 1;
    //add even number to end
    c_end[c_end.size() - 1] -= 1;

    bool parted1 = pika::is_partitioned(policy, iterator(std::begin(c_beg)),
        iterator(std::end(c_beg)), [](std::size_t n) { return n % 2 == 0; });
    bool parted2 = pika::is_partitioned(policy, iterator(std::begin(c_end)),
        iterator(std::end(c_end)), [](std::size_t n) { return n % 2 == 0; });

    PIKA_TEST(!parted1);
    PIKA_TEST(!parted2);
}

template <typename ExPolicy, typename IteratorTag>
void test_partitioned3_async(ExPolicy p, IteratorTag)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c_beg(10007);
    //fill first half of array with even numbers and second half
    //with odd numbers
    std::fill(std::begin(c_beg), std::begin(c_beg) + c_beg.size() / 2,
        2 * (dis(gen)));
    std::fill(std::begin(c_beg) + c_beg.size() / 2, std::end(c_beg),
        2 * (dis(gen)) + 1);
    std::vector<size_t> c_end = c_beg;
    //add odd number to the beginning
    c_beg[0] -= 1;
    //add even number to end
    c_end[c_end.size() - 1] -= 1;

    pika::future<bool> f_beg = pika::is_partitioned(p,
        iterator(std::begin(c_beg)), iterator(std::end(c_beg)),
        [](std::size_t n) { return n % 2 == 0; });
    pika::future<bool> f_end = pika::is_partitioned(p,
        iterator(std::begin(c_end)), iterator(std::end(c_end)),
        [](std::size_t n) { return n % 2 == 0; });

    f_beg.wait();
    PIKA_TEST(!f_beg.get());
    f_end.wait();
    PIKA_TEST(!f_end.get());
}

template <typename IteratorTag>
void test_partitioned3()
{
    using namespace pika::execution;
    test_partitioned3(seq, IteratorTag());
    test_partitioned3(par, IteratorTag());
    test_partitioned3(par_unseq, IteratorTag());

    test_partitioned3_async(seq(task), IteratorTag());
    test_partitioned3_async(par(task), IteratorTag());
}

void partitioned_test3()
{
    test_partitioned3<std::random_access_iterator_tag>();
    test_partitioned3<std::forward_iterator_tag>();
}

////////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_partitioned_exception(ExPolicy policy, IteratorTag)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(10007);
    //fill first half of array with even numbers and second half
    //with odd numbers
    std::fill(std::begin(c), std::begin(c) + c.size() / 2, 2 * (dis(gen)));
    std::fill(std::begin(c) + c.size() / 2, std::end(c), 2 * (dis(gen)) + 1);

    bool caught_exception = false;
    try
    {
        pika::is_partitioned(policy,
            decorated_iterator(
                std::begin(c), []() { throw std::runtime_error("test"); }),
            decorated_iterator(
                std::end(c), []() { throw std::runtime_error("test"); }),
            [](std::size_t n) { return n % 2 == 0; });
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
void test_partitioned_async_exception(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(10007);
    //fill first half of array with even numbers and second half
    //with odd numbers
    std::fill(std::begin(c), std::begin(c) + c.size() / 2, 2 * (dis(gen)));
    std::fill(std::begin(c) + c.size() / 2, std::end(c), 2 * (dis(gen)) + 1);

    bool caught_exception = false;
    try
    {
        pika::future<bool> f = pika::is_partitioned(p,
            decorated_iterator(
                std::begin(c), []() { throw std::runtime_error("test"); }),
            decorated_iterator(
                std::end(c), []() { throw std::runtime_error("test"); }),
            [](std::size_t n) { return n % 2 == 0; });
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
}

template <typename IteratorTag>
void test_partitioned_exception()
{
    using namespace pika::execution;
    //If the execution policy object is of type vector_execution_policy,
    //  std::terminate shall be called. Therefore we do not test exceptions
    //  with a vector execution policy
    test_partitioned_exception(seq, IteratorTag());
    test_partitioned_exception(par, IteratorTag());

    test_partitioned_async_exception(seq(task), IteratorTag());
    test_partitioned_async_exception(par(task), IteratorTag());
}

void partitioned_exception_test()
{
    test_partitioned_exception<std::random_access_iterator_tag>();
    test_partitioned_exception<std::forward_iterator_tag>();
}

////////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_partitioned_bad_alloc(ExPolicy policy, IteratorTag)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(10007);
    //fill first half of array with even numbers and second half
    //with odd numbers
    std::fill(std::begin(c), std::begin(c) + c.size() / 2, 2 * (dis(gen)));
    std::fill(std::begin(c) + c.size() / 2, std::end(c), 2 * (dis(gen)) + 1);

    bool caught_bad_alloc = false;
    try
    {
        pika::is_partitioned(policy,
            decorated_iterator(std::begin(c), []() { throw std::bad_alloc(); }),
            decorated_iterator(std::end(c), []() { throw std::bad_alloc(); }),
            [](std::size_t n) { return n % 2 == 0; });
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
void test_partitioned_async_bad_alloc(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(10007);
    //fill first half of array with even numbers and second half
    //with odd numbers
    std::fill(std::begin(c), std::begin(c) + c.size() / 2, 2 * (dis(gen)));
    std::fill(std::begin(c) + c.size() / 2, std::end(c), 2 * (dis(gen)) + 1);

    bool caught_bad_alloc = false;
    try
    {
        pika::future<bool> f = pika::is_partitioned(p,
            decorated_iterator(std::begin(c), []() { throw std::bad_alloc(); }),
            decorated_iterator(std::end(c), []() { throw std::bad_alloc(); }),
            [](std::size_t n) { return n % 2 == 0; });

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
}

template <typename IteratorTag>
void test_partitioned_bad_alloc()
{
    using namespace pika::execution;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_partitioned_bad_alloc(par, IteratorTag());
    test_partitioned_bad_alloc(seq, IteratorTag());

    test_partitioned_async_bad_alloc(seq(task), IteratorTag());
    test_partitioned_async_bad_alloc(par(task), IteratorTag());
}

void partitioned_bad_alloc_test()
{
    test_partitioned_bad_alloc<std::random_access_iterator_tag>();
    test_partitioned_bad_alloc<std::forward_iterator_tag>();
}

int pika_main()
{
    partitioned_test1();
    partitioned_test2();
    partitioned_test3();
    partitioned_exception_test();
    partitioned_bad_alloc_test();

    return pika::local::finalize();
}

int main(int argc, char* argv[])
{
    using namespace pika::program_options;
    options_description desc_commandline(
        "Usage: " PIKA_APPLICATION_STRING " [options]");

    std::vector<std::string> const cfg = {"pika.os_threads=all"};

    pika::local::init_params init_args;
    init_args.desc_cmdline = desc_commandline;
    init_args.cfg = cfg;

    PIKA_TEST_EQ_MSG(pika::local::init(pika_main, argc, argv, init_args), 0,
        "pika main exited with non-zero status");

    return pika::util::report_errors();
}
