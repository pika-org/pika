//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c)      2021 Giannis Gonidelis
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/iterator_support/iterator_range.hpp>
#include <pika/iterator_support/tests/iter_sent.hpp>
#include <pika/local/init.hpp>
#include <pika/modules/testing.hpp>
#include <pika/parallel/container_algorithms/reverse.hpp>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <numeric>
#include <string>
#include <vector>

#include "test_utils.hpp"

///////////////////////////////////////////////////////////////////////////////
void test_reverse_sent()
{
    std::size_t const size = 100;
    std::vector<std::int16_t> c(size);
    std::iota(std::begin(c), std::end(c), 0);

    auto first = c.begin();
    PIKA_TEST(*first == 0);

    pika::ranges::reverse(std::begin(c), sentinel<std::int16_t>{50});
    auto first_reversed = c.begin();

    PIKA_TEST(*first_reversed == 49);
}

template <typename ExPolicy>
void test_reverse_sent(ExPolicy policy)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    std::size_t const size = 100;
    std::vector<std::int16_t> c(size);
    std::iota(std::begin(c), std::end(c), 0);

    auto first = c.begin();
    PIKA_TEST(*first == 0);

    pika::ranges::reverse(policy, std::begin(c), sentinel<std::int16_t>{50});
    auto first_reversed = c.begin();

    PIKA_TEST(*first_reversed == 49);
}

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_reverse(IteratorTag)
{
    typedef test::test_container<std::vector<std::size_t>, IteratorTag>
        test_vector;

    test_vector c(10007);
    std::vector<std::size_t> d1;

    std::iota(std::begin(c.base()), std::end(c.base()), std::rand());
    std::copy(std::begin(c.base()), std::end(c.base()), std::back_inserter(d1));

    pika::ranges::reverse(c);

    std::reverse(std::begin(d1), std::end(d1));

    std::size_t count = 0;
    PIKA_TEST(std::equal(std::begin(c.base()), std::end(c.base()),
        std::begin(d1), [&count](std::size_t v1, std::size_t v2) -> bool {
            PIKA_TEST_EQ(v1, v2);
            ++count;
            return v1 == v2;
        }));
    PIKA_TEST_EQ(count, d1.size());
}

template <typename ExPolicy, typename IteratorTag>
void test_reverse(ExPolicy policy, IteratorTag)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    typedef test::test_container<std::vector<std::size_t>, IteratorTag>
        test_vector;

    test_vector c(10007);
    std::vector<std::size_t> d1;

    std::iota(std::begin(c.base()), std::end(c.base()), std::rand());
    std::copy(std::begin(c.base()), std::end(c.base()), std::back_inserter(d1));

    pika::ranges::reverse(policy, c);

    std::reverse(std::begin(d1), std::end(d1));

    std::size_t count = 0;
    PIKA_TEST(std::equal(std::begin(c.base()), std::end(c.base()),
        std::begin(d1), [&count](std::size_t v1, std::size_t v2) -> bool {
            PIKA_TEST_EQ(v1, v2);
            ++count;
            return v1 == v2;
        }));
    PIKA_TEST_EQ(count, d1.size());
}

template <typename ExPolicy, typename IteratorTag>
void test_reverse_async(ExPolicy p, IteratorTag)
{
    typedef test::test_container<std::vector<std::size_t>, IteratorTag>
        test_vector;

    test_vector c(10007);
    std::vector<std::size_t> d1;

    std::iota(std::begin(c.base()), std::end(c.base()), std::rand());
    std::copy(std::begin(c.base()), std::end(c.base()), std::back_inserter(d1));

    auto f = pika::ranges::reverse(p, c);
    f.wait();

    std::reverse(std::begin(d1), std::end(d1));

    std::size_t count = 0;
    PIKA_TEST(std::equal(std::begin(c.base()), std::end(c.base()),
        std::begin(d1), [&count](std::size_t v1, std::size_t v2) -> bool {
            PIKA_TEST_EQ(v1, v2);
            ++count;
            return v1 == v2;
        }));
    PIKA_TEST_EQ(count, d1.size());
}

template <typename IteratorTag>
void test_reverse()
{
    using namespace pika::execution;
    test_reverse(IteratorTag());
    test_reverse(seq, IteratorTag());
    test_reverse(par, IteratorTag());
    test_reverse(par_unseq, IteratorTag());

    test_reverse_async(seq(task), IteratorTag());
    test_reverse_async(par(task), IteratorTag());

    test_reverse_sent();
    test_reverse_sent(seq);
    test_reverse_sent(par);
    test_reverse_sent(par_unseq);
}

void reverse_test()
{
    test_reverse<std::random_access_iterator_tag>();
    test_reverse<std::bidirectional_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_reverse_exception(IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), std::rand());

    bool caught_exception = false;
    try
    {
        pika::ranges::reverse(
            pika::util::make_iterator_range(decorated_iterator(std::begin(c)),
                decorated_iterator(
                    std::end(c), []() { throw std::runtime_error("test"); })));
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
void test_reverse_exception(ExPolicy policy, IteratorTag)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), std::rand());

    bool caught_exception = false;
    try
    {
        pika::ranges::reverse(policy,
            pika::util::make_iterator_range(decorated_iterator(std::begin(c)),
                decorated_iterator(
                    std::end(c), []() { throw std::runtime_error("test"); })));
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
void test_reverse_exception_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), std::rand());

    bool caught_exception = false;
    bool returned_from_algorithm = false;
    try
    {
        auto f = pika::ranges::reverse(p,
            pika::util::make_iterator_range(decorated_iterator(std::begin(c)),
                decorated_iterator(
                    std::end(c), []() { throw std::runtime_error("test"); })));
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
void test_reverse_exception()
{
    using namespace pika::execution;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_reverse_exception(IteratorTag());
    test_reverse_exception(seq, IteratorTag());
    test_reverse_exception(par, IteratorTag());

    test_reverse_exception_async(seq(task), IteratorTag());
    test_reverse_exception_async(par(task), IteratorTag());
}

void reverse_exception_test()
{
    test_reverse_exception<std::random_access_iterator_tag>();
    test_reverse_exception<std::bidirectional_iterator_tag>();
}

//////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_reverse_bad_alloc(IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), std::rand());

    bool caught_bad_alloc = false;
    try
    {
        pika::ranges::reverse(pika::util::make_iterator_range(
            decorated_iterator(std::begin(c)),
            decorated_iterator(std::end(c), []() { throw std::bad_alloc(); })));
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
void test_reverse_bad_alloc(ExPolicy policy, IteratorTag)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), std::rand());

    bool caught_bad_alloc = false;
    try
    {
        pika::ranges::reverse(policy,
            pika::util::make_iterator_range(decorated_iterator(std::begin(c)),
                decorated_iterator(
                    std::end(c), []() { throw std::bad_alloc(); })));
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
void test_reverse_bad_alloc_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), std::rand());

    bool caught_bad_alloc = false;
    bool returned_from_algorithm = false;
    try
    {
        auto f = pika::ranges::reverse(p,
            pika::util::make_iterator_range(decorated_iterator(std::begin(c)),
                decorated_iterator(
                    std::end(c), []() { throw std::bad_alloc(); })));
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
void test_reverse_bad_alloc()
{
    using namespace pika::execution;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_reverse_bad_alloc(IteratorTag());
    test_reverse_bad_alloc(seq, IteratorTag());
    test_reverse_bad_alloc(par, IteratorTag());

    test_reverse_bad_alloc_async(seq(task), IteratorTag());
    test_reverse_bad_alloc_async(par(task), IteratorTag());
}

void reverse_bad_alloc_test()
{
    test_reverse_bad_alloc<std::random_access_iterator_tag>();
    test_reverse_bad_alloc<std::bidirectional_iterator_tag>();
}

int pika_main(pika::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int) std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    reverse_test();
    reverse_exception_test();
    reverse_bad_alloc_test();
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
