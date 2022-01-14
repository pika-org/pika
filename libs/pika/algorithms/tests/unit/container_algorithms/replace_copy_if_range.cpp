//  Copyright (c) 2014-2017 Hartmut Kaiser
//  Copyright (c)      2021 Giannis Gonidelis
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/iterator_support/iterator_range.hpp>
#include <pika/iterator_support/tests/iter_sent.hpp>
#include <pika/local/init.hpp>
#include <pika/modules/testing.hpp>
#include <pika/parallel/container_algorithms/replace.hpp>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <numeric>
#include <string>
#include <vector>

#include "test_utils.hpp"

////////////////////////////////////////////////////////////////////////////
struct equal_f
{
    equal_f(std::size_t val)
      : val_(val)
    {
    }

    bool operator()(std::size_t lhs) const
    {
        return lhs == val_;
    }

    std::size_t val_;
};

////////////////////////////////////////////////////////////////////////////
void test_replace_copy_if_sent()
{
    using pika::get;

    std::size_t const size = 100;
    std::vector<std::int16_t> c(size), d(size);
    std::iota(std::begin(c), std::end(c), 0);
    c[99] = 42;

    int new_value = 1;

    auto pred = [](std::int16_t const& a) -> bool { return a == 42; };

    pika::ranges::replace_copy_if(std::begin(c), sentinel<std::int16_t>{50},
        std::begin(d), pred, new_value);
    auto result1 = std::count_if(std::begin(d), std::end(d), pred);

    PIKA_TEST(result1 == 0);
}

template <typename ExPolicy>
void test_replace_copy_if_sent(ExPolicy policy)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    using pika::get;

    std::size_t const size = 100;
    std::vector<std::int16_t> c(size), d(size);
    std::iota(std::begin(c), std::end(c), 0);
    c[99] = 42;

    int new_value = 1;

    auto pred = [](std::int16_t const& a) -> bool { return a == 42; };

    pika::ranges::replace_copy_if(policy, std::begin(c),
        sentinel<std::int16_t>{50}, std::begin(d), pred, new_value);
    auto result1 = std::count_if(std::begin(d), std::end(d), pred);

    PIKA_TEST(result1 == 0);
}

////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_replace_copy_if(IteratorTag)
{
    typedef test::test_container<std::vector<std::size_t>, IteratorTag>
        test_vector;

    test_vector c(10007);
    std::vector<std::size_t> d1(c.size());
    std::vector<std::size_t> d2(c.size());    //-V656

    std::iota(std::begin(c.base()), std::end(c.base()), std::rand());

    std::size_t idx = std::rand() % c.size();    //-V104

    pika::ranges::replace_copy_if(
        c, std::begin(d1), equal_f(c[idx]), c[idx] + 1);

    std::replace_copy_if(std::begin(c.base()), std::end(c.base()),
        std::begin(d2), equal_f(c[idx]), c[idx] + 1);

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
void test_replace_copy_if(ExPolicy policy, IteratorTag)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    typedef test::test_container<std::vector<std::size_t>, IteratorTag>
        test_vector;

    test_vector c(10007);
    std::vector<std::size_t> d1(c.size());
    std::vector<std::size_t> d2(c.size());    //-V656

    std::iota(std::begin(c.base()), std::end(c.base()), std::rand());

    std::size_t idx = std::rand() % c.size();    //-V104

    pika::ranges::replace_copy_if(
        policy, c, std::begin(d1), equal_f(c[idx]), c[idx] + 1);

    std::replace_copy_if(std::begin(c.base()), std::end(c.base()),
        std::begin(d2), equal_f(c[idx]), c[idx] + 1);

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
void test_replace_copy_if_async(ExPolicy p, IteratorTag)
{
    typedef test::test_container<std::vector<std::size_t>, IteratorTag>
        test_vector;

    test_vector c(10007);
    std::vector<std::size_t> d1(c.size());
    std::vector<std::size_t> d2(c.size());    //-V656

    std::iota(std::begin(c.base()), std::end(c.base()), std::rand());

    std::size_t idx = std::rand() % c.size();    //-V104

    auto f = pika::ranges::replace_copy_if(
        p, c, std::begin(d1), equal_f(c[idx]), c[idx] + 1);
    f.wait();

    std::replace_copy_if(std::begin(c.base()), std::end(c.base()),
        std::begin(d2), equal_f(c[idx]), c[idx] + 1);

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
void test_replace_copy_if()
{
    using namespace pika::execution;
    test_replace_copy_if(IteratorTag());
    test_replace_copy_if(seq, IteratorTag());
    test_replace_copy_if(par, IteratorTag());
    test_replace_copy_if(par_unseq, IteratorTag());

    test_replace_copy_if_async(seq(task), IteratorTag());
    test_replace_copy_if_async(par(task), IteratorTag());

    test_replace_copy_if_sent();
    test_replace_copy_if_sent(seq);
    test_replace_copy_if_sent(par);
    test_replace_copy_if_sent(par_unseq);
}

void replace_copy_if_test()
{
    test_replace_copy_if<std::random_access_iterator_tag>();
    test_replace_copy_if<std::forward_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_replace_copy_if_exception(IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::iota(std::begin(c), std::end(c), std::rand());

    bool caught_exception = false;
    try
    {
        pika::ranges::replace_copy_if(
            pika::util::make_iterator_range(
                decorated_iterator(
                    std::begin(c), []() { throw std::runtime_error("test"); }),
                decorated_iterator(std::end(c))),
            std::begin(d), equal_f(42), std::size_t(43));
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
void test_replace_copy_if_exception(ExPolicy policy, IteratorTag)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::iota(std::begin(c), std::end(c), std::rand());

    bool caught_exception = false;
    try
    {
        pika::ranges::replace_copy_if(policy,
            pika::util::make_iterator_range(
                decorated_iterator(
                    std::begin(c), []() { throw std::runtime_error("test"); }),
                decorated_iterator(std::end(c))),
            std::begin(d), equal_f(42), std::size_t(43));
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
void test_replace_copy_if_exception_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::iota(std::begin(c), std::end(c), std::rand());

    bool caught_exception = false;
    bool returned_from_algorithm = false;
    try
    {
        auto f = pika::ranges::replace_copy_if(p,
            pika::util::make_iterator_range(
                decorated_iterator(
                    std::begin(c), []() { throw std::runtime_error("test"); }),
                decorated_iterator(std::end(c))),
            std::begin(d), equal_f(42), std::size_t(43));
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
void test_replace_copy_if_exception()
{
    using namespace pika::execution;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_replace_copy_if_exception(IteratorTag());
    test_replace_copy_if_exception(seq, IteratorTag());
    test_replace_copy_if_exception(par, IteratorTag());

    test_replace_copy_if_exception_async(seq(task), IteratorTag());
    test_replace_copy_if_exception_async(par(task), IteratorTag());
}

void replace_copy_if_exception_test()
{
    test_replace_copy_if_exception<std::random_access_iterator_tag>();
    test_replace_copy_if_exception<std::forward_iterator_tag>();
}

//////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_replace_copy_if_bad_alloc(IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::iota(std::begin(c), std::end(c), std::rand());

    bool caught_bad_alloc = false;
    try
    {
        pika::ranges::replace_copy_if(pika::util::make_iterator_range(
                                         decorated_iterator(std::begin(c),
                                             []() { throw std::bad_alloc(); }),
                                         decorated_iterator(std::end(c))),
            std::begin(d), equal_f(42), std::size_t(43));
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
void test_replace_copy_if_bad_alloc(ExPolicy policy, IteratorTag)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::iota(std::begin(c), std::end(c), std::rand());

    bool caught_bad_alloc = false;
    try
    {
        pika::ranges::replace_copy_if(policy,
            pika::util::make_iterator_range(
                decorated_iterator(
                    std::begin(c), []() { throw std::bad_alloc(); }),
                decorated_iterator(std::end(c))),
            std::begin(d), equal_f(42), std::size_t(43));
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
void test_replace_copy_if_bad_alloc_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::iota(std::begin(c), std::end(c), std::rand());

    bool caught_bad_alloc = false;
    bool returned_from_algorithm = false;
    try
    {
        auto f = pika::ranges::replace_copy_if(p,
            pika::util::make_iterator_range(
                decorated_iterator(
                    std::begin(c), []() { throw std::bad_alloc(); }),
                decorated_iterator(std::end(c))),
            std::begin(d), equal_f(42), std::size_t(43));
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
void test_replace_copy_if_bad_alloc()
{
    using namespace pika::execution;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_replace_copy_if_bad_alloc(IteratorTag());
    test_replace_copy_if_bad_alloc(seq, IteratorTag());
    test_replace_copy_if_bad_alloc(par, IteratorTag());

    test_replace_copy_if_bad_alloc_async(seq(task), IteratorTag());
    test_replace_copy_if_bad_alloc_async(par(task), IteratorTag());
}

void replace_copy_if_bad_alloc_test()
{
    test_replace_copy_if_bad_alloc<std::random_access_iterator_tag>();
    test_replace_copy_if_bad_alloc<std::forward_iterator_tag>();
}

int pika_main(pika::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int) std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    replace_copy_if_test();
    replace_copy_if_exception_test();
    replace_copy_if_bad_alloc_test();
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
