//  Copyright (c) 2014 Grant Mercer
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/init.hpp>
#include <pika/modules/testing.hpp>
#include <pika/parallel/algorithms/find.hpp>

#include <cstddef>
#include <functional>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "test_utils.hpp"

////////////////////////////////////////////////////////////////////////////
unsigned int seed = std::random_device{}();
std::mt19937 gen(seed);
std::uniform_int_distribution<> dis(3, 102);
std::uniform_int_distribution<> dist(7, 106);

template <typename IteratorTag>
void test_find_end1(IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    // fill vector with random values above 2
    std::fill(std::begin(c), std::end(c), dis(gen));
    // create subsequence in middle of vector
    c[c.size() / 2] = 1;
    c[c.size() / 2 + 1] = 2;

    std::size_t h[] = {1, 2};

    iterator index = pika::find_end(iterator(std::begin(c)),
        iterator(std::end(c)), std::begin(h), std::end(h));

    iterator test_index = std::find_end(iterator(std::begin(c)),
        iterator(std::end(c)), std::begin(h), std::end(h));

    PIKA_TEST(index == test_index);
}

template <typename ExPolicy, typename IteratorTag>
void test_find_end1(ExPolicy&& policy, IteratorTag)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    // fill vector with random values above 2
    std::fill(std::begin(c), std::end(c), dis(gen));
    // create subsequence in middle of vector
    c[c.size() / 2] = 1;
    c[c.size() / 2 + 1] = 2;

    std::size_t h[] = {1, 2};

    iterator index = pika::find_end(policy, iterator(std::begin(c)),
        iterator(std::end(c)), std::begin(h), std::end(h));

    iterator test_index = std::find_end(iterator(std::begin(c)),
        iterator(std::end(c)), std::begin(h), std::end(h));

    PIKA_TEST(index == test_index);
}

template <typename ExPolicy, typename IteratorTag>
void test_find_end1_async(ExPolicy&& p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    // fill vector with random values above 2
    std::fill(std::begin(c), std::end(c), dis(gen));
    // create subsequence in middle of vector
    c[c.size() / 2] = 1;
    c[c.size() / 2 + 1] = 2;

    std::size_t h[] = {1, 2};

    pika::future<iterator> f = pika::find_end(p, iterator(std::begin(c)),
        iterator(std::end(c)), std::begin(h), std::end(h));
    f.wait();

    iterator test_index = std::find_end(iterator(std::begin(c)),
        iterator(std::end(c)), std::begin(h), std::end(h));

    PIKA_TEST(f.get() == test_index);
}

template <typename IteratorTag>
void test_find_end1()
{
    using namespace pika::execution;

    test_find_end1(IteratorTag());

    test_find_end1(seq, IteratorTag());
    test_find_end1(par, IteratorTag());
    test_find_end1(par_unseq, IteratorTag());

    test_find_end1_async(seq(task), IteratorTag());
    test_find_end1_async(par(task), IteratorTag());
}

void find_end_test1()
{
    test_find_end1<std::random_access_iterator_tag>();
    test_find_end1<std::forward_iterator_tag>();
}

template <typename IteratorTag>
void test_find_end2(IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    // fill vector with random values about 2
    std::fill(std::begin(c), std::end(c), dis(gen));
    // create subsequence at start and end
    c[0] = 1;
    c[1] = 2;
    c[c.size() - 1] = 2;
    c[c.size() - 2] = 1;

    std::size_t h[] = {1, 2};

    iterator index = pika::find_end(iterator(std::begin(c)),
        iterator(std::end(c)), std::begin(h), std::end(h));

    iterator test_index = std::find_end(iterator(std::begin(c)),
        iterator(std::end(c)), std::begin(h), std::end(h));

    PIKA_TEST(index == test_index);
}

template <typename ExPolicy, typename IteratorTag>
void test_find_end2(ExPolicy&& policy, IteratorTag)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    // fill vector with random values about 2
    std::fill(std::begin(c), std::end(c), dis(gen));
    // create subsequence at start and end
    c[0] = 1;
    c[1] = 2;
    c[c.size() - 1] = 2;
    c[c.size() - 2] = 1;

    std::size_t h[] = {1, 2};

    iterator index = pika::find_end(policy, iterator(std::begin(c)),
        iterator(std::end(c)), std::begin(h), std::end(h));

    iterator test_index = std::find_end(iterator(std::begin(c)),
        iterator(std::end(c)), std::begin(h), std::end(h));

    PIKA_TEST(index == test_index);
}

template <typename ExPolicy, typename IteratorTag>
void test_find_end2_async(ExPolicy&& p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    // fill vector with random values above 2
    std::vector<std::size_t> c(10007);
    std::fill(std::begin(c), std::end(c), dis(gen));

    // create subsequence at start and end
    c[0] = 1;
    c[1] = 2;
    c[c.size() - 1] = 2;
    c[c.size() - 2] = 1;

    std::size_t h[] = {1, 2};

    pika::future<iterator> f = pika::find_end(p, iterator(std::begin(c)),
        iterator(std::end(c)), std::begin(h), std::end(h));
    f.wait();

    iterator test_index = std::find_end(iterator(std::begin(c)),
        iterator(std::end(c)), std::begin(h), std::end(h));

    PIKA_TEST(f.get() == test_index);
}

template <typename IteratorTag>
void test_find_end2()
{
    using namespace pika::execution;

    test_find_end2(IteratorTag());

    test_find_end2(seq, IteratorTag());
    test_find_end2(par, IteratorTag());
    test_find_end2(par_unseq, IteratorTag());

    test_find_end2_async(seq(task), IteratorTag());
    test_find_end2_async(par(task), IteratorTag());
}

void find_end_test2()
{
    test_find_end2<std::random_access_iterator_tag>();
    test_find_end2<std::forward_iterator_tag>();
}

template <typename IteratorTag>
void test_find_end3(IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    // fill vector with random values above 2
    std::vector<std::size_t> c(10007);
    std::fill(std::begin(c), std::end(c), dis(gen));

    // create subsequence large enough to always be split into multiple partitions
    std::iota(std::begin(c), std::begin(c) + c.size() / 16 + 1, 1);
    std::size_t sub_size = c.size() / 16 + 1;

    std::vector<std::size_t> h(sub_size);
    std::iota(std::begin(h), std::end(h), 1);

    iterator index = pika::find_end(iterator(std::begin(c)),
        iterator(std::end(c)), std::begin(h), std::end(h));

    iterator test_index = std::find_end(iterator(std::begin(c)),
        iterator(std::end(c)), std::begin(h), std::end(h));

    PIKA_TEST(index == test_index);
}

template <typename ExPolicy, typename IteratorTag>
void test_find_end3(ExPolicy&& policy, IteratorTag)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    // fill vector with random values above 2
    std::vector<std::size_t> c(10007);
    std::fill(std::begin(c), std::end(c), dis(gen));

    // create subsequence large enough to always be split into multiple partitions
    std::iota(std::begin(c), std::begin(c) + c.size() / 16 + 1, 1);
    std::size_t sub_size = c.size() / 16 + 1;

    std::vector<std::size_t> h(sub_size);
    std::iota(std::begin(h), std::end(h), 1);

    iterator index = pika::find_end(policy, iterator(std::begin(c)),
        iterator(std::end(c)), std::begin(h), std::end(h));

    iterator test_index = std::find_end(iterator(std::begin(c)),
        iterator(std::end(c)), std::begin(h), std::end(h));

    PIKA_TEST(index == test_index);
}

template <typename ExPolicy, typename IteratorTag>
void test_find_end3_async(ExPolicy&& p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    // fill vector with random values above 6
    std::vector<std::size_t> c(10007);
    std::fill(std::begin(c), std::end(c), dist(gen));

    // create subsequence large enough to always be split into multiple partitions
    std::iota(std::begin(c), std::begin(c) + c.size() / 16 + 1, 1);
    std::size_t sub_size = c.size() / 16 + 1;

    std::vector<std::size_t> h(sub_size);
    std::iota(std::begin(h), std::end(h), 1);

    // create only two partitions, splitting the desired sub sequence into
    // separate partitions.
    pika::future<iterator> f = pika::find_end(p, iterator(std::begin(c)),
        iterator(std::end(c)), std::begin(h), std::end(h));
    f.wait();

    iterator test_index = std::find_end(iterator(std::begin(c)),
        iterator(std::end(c)), std::begin(h), std::end(h));

    PIKA_TEST(f.get() == test_index);
}

template <typename IteratorTag>
void test_find_end3()
{
    using namespace pika::execution;

    test_find_end3(IteratorTag());

    test_find_end3(seq, IteratorTag());
    test_find_end3(par, IteratorTag());
    test_find_end3(par_unseq, IteratorTag());

    test_find_end3_async(seq(task), IteratorTag());
    test_find_end3_async(par(task), IteratorTag());
}

void find_end_test3()
{
    test_find_end3<std::random_access_iterator_tag>();
    test_find_end3<std::forward_iterator_tag>();
}

template <typename IteratorTag>
void test_find_end4(IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    // fill vector with random values above 2
    std::fill(std::begin(c), std::end(c), dis(gen));

    // create subsequence in middle of vector
    c[c.size() / 2] = 1;
    c[c.size() / 2 + 1] = 2;

    std::size_t h[] = {1, 2};

    iterator index = pika::find_end(iterator(std::begin(c)),
        iterator(std::end(c)), std::begin(h), std::end(h),
        [](std::size_t v1, std::size_t v2) { return !(v1 != v2); });

    iterator test_index = std::find_end(iterator(std::begin(c)),
        iterator(std::end(c)), std::begin(h), std::end(h),
        [](std::size_t v1, std::size_t v2) { return !(v1 != v2); });

    PIKA_TEST(index == test_index);
}

template <typename ExPolicy, typename IteratorTag>
void test_find_end4(ExPolicy&& policy, IteratorTag)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    // fill vector with random values above 2
    std::fill(std::begin(c), std::end(c), dis(gen));
    // create subsequence in middle of vector
    c[c.size() / 2] = 1;
    c[c.size() / 2 + 1] = 2;

    std::size_t h[] = {1, 2};

    iterator index = pika::find_end(policy, iterator(std::begin(c)),
        iterator(std::end(c)), std::begin(h), std::end(h),
        [](std::size_t v1, std::size_t v2) { return !(v1 != v2); });

    iterator test_index = std::find_end(iterator(std::begin(c)),
        iterator(std::end(c)), std::begin(h), std::end(h),
        [](std::size_t v1, std::size_t v2) { return !(v1 != v2); });

    PIKA_TEST(index == test_index);
}

template <typename ExPolicy, typename IteratorTag>
void test_find_end4_async(ExPolicy&& p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    // fill vector with random values above 2
    std::vector<std::size_t> c(10007);
    std::fill(std::begin(c), std::end(c), dis(gen));

    // create subsequence in middle of vector
    c[c.size() / 2] = 1;
    c[c.size() / 2 + 1] = 2;

    std::size_t h[] = {1, 2};

    pika::future<iterator> f = pika::find_end(p, iterator(std::begin(c)),
        iterator(std::end(c)), std::begin(h), std::end(h),
        [](std::size_t v1, std::size_t v2) { return !(v1 != v2); });
    f.wait();

    iterator test_index = std::find_end(iterator(std::begin(c)),
        iterator(std::end(c)), std::begin(h), std::end(h),
        [](std::size_t v1, std::size_t v2) { return !(v1 != v2); });

    PIKA_TEST(f.get() == test_index);
}

template <typename IteratorTag>
void test_find_end4()
{
    using namespace pika::execution;

    test_find_end4(IteratorTag());

    test_find_end4(seq, IteratorTag());
    test_find_end4(par, IteratorTag());
    test_find_end4(par_unseq, IteratorTag());

    test_find_end4_async(seq(task), IteratorTag());
    test_find_end4_async(par(task), IteratorTag());
}

void find_end_test4()
{
    test_find_end4<std::random_access_iterator_tag>();
    test_find_end4<std::forward_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_find_end_exception(ExPolicy&& policy, IteratorTag)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;
    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), gen() + 1);
    c[c.size() / 2] = 1;
    c[c.size() / 2 + 1] = 2;

    std::vector<std::size_t> h;
    h.push_back(1);
    h.push_back(2);

    bool caught_exception = false;
    try
    {
        pika::find_end(policy,
            decorated_iterator(
                std::begin(c), []() { throw std::runtime_error("test"); }),
            decorated_iterator(
                std::end(c), []() { throw std::runtime_error("test"); }),
            std::begin(h), std::end(h));
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

template <typename IteratorTag>
void test_find_end_exception(IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), gen() + 1);
    c[c.size() / 2] = 1;
    c[c.size() / 2 + 1] = 2;

    std::vector<std::size_t> h;
    h.push_back(1);
    h.push_back(2);

    bool caught_exception = false;
    try
    {
        pika::find_end(decorated_iterator(std::begin(c),
                          []() { throw std::runtime_error("test"); }),
            decorated_iterator(
                std::end(c), []() { throw std::runtime_error("test"); }),
            std::begin(h), std::end(h));
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
}

template <typename ExPolicy, typename IteratorTag>
void test_find_end_exception_async(ExPolicy&& p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), gen() + 1);
    c[c.size() / 2] = 1;
    c[c.size() / 2 + 1] = 2;

    std::size_t h[] = {1, 2};

    bool caught_exception = false;
    bool returned_from_algorithm = false;
    try
    {
        pika::future<decorated_iterator> f = pika::find_end(p,
            decorated_iterator(
                std::begin(c), []() { throw std::runtime_error("test"); }),
            decorated_iterator(
                std::end(c), []() { throw std::runtime_error("test"); }),
            std::begin(h), std::end(h));
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
void test_find_end_exception()
{
    using namespace pika::execution;

    test_find_end_exception(IteratorTag());

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_find_end_exception(seq, IteratorTag());
    test_find_end_exception(par, IteratorTag());

    test_find_end_exception_async(seq(task), IteratorTag());
    test_find_end_exception_async(par(task), IteratorTag());
}

void find_end_exception_test()
{
    test_find_end_exception<std::random_access_iterator_tag>();
    test_find_end_exception<std::forward_iterator_tag>();
}

//////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_find_end_bad_alloc(ExPolicy&& policy, IteratorTag)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(100007);
    std::iota(std::begin(c), std::end(c), gen() + 1);
    c[c.size() / 2] = 0;

    std::size_t h[] = {1, 2};

    bool caught_bad_alloc = false;
    try
    {
        pika::find_end(policy,
            decorated_iterator(std::begin(c), []() { throw std::bad_alloc(); }),
            decorated_iterator(std::end(c), []() { throw std::bad_alloc(); }),
            std::begin(h), std::end(h));
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
void test_find_end_bad_alloc_async(ExPolicy&& p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), gen() + 1);
    c[c.size() / 2] = 0;

    std::size_t h[] = {1, 2};

    bool caught_bad_alloc = false;
    bool returned_from_algorithm = false;
    try
    {
        pika::future<decorated_iterator> f = pika::find_end(p,
            decorated_iterator(std::begin(c), []() { throw std::bad_alloc(); }),
            decorated_iterator(std::end(c), []() { throw std::bad_alloc(); }),
            std::begin(h), std::end(h));
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
void test_find_end_bad_alloc()
{
    using namespace pika::execution;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_find_end_bad_alloc(seq, IteratorTag());
    test_find_end_bad_alloc(par, IteratorTag());

    test_find_end_bad_alloc_async(seq(task), IteratorTag());
    test_find_end_bad_alloc_async(par(task), IteratorTag());
}

void find_end_bad_alloc_test()
{
    test_find_end_bad_alloc<std::random_access_iterator_tag>();
    test_find_end_bad_alloc<std::forward_iterator_tag>();
}

int pika_main(pika::program_options::variables_map& vm)
{
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    gen.seed(seed);

    find_end_test1();
    find_end_test2();
    find_end_test3();
    find_end_test4();
    find_end_exception_test();
    find_end_bad_alloc_test();
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
