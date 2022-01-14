//  Copyright (c) 2014-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/init.hpp>
#include <pika/modules/testing.hpp>
#include <pika/parallel/algorithms/transform_reduce.hpp>

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <numeric>
#include <string>
#include <vector>

#include "test_utils.hpp"

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_transform_reduce(IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), std::rand());

    typedef pika::tuple<std::size_t, std::size_t> result_type;

    using pika::get;
    using pika::make_tuple;

    auto reduce_op = [](result_type v1, result_type v2) -> result_type {
        return make_tuple(get<0>(v1) * get<0>(v2), get<1>(v1) * get<1>(v2));
    };

    auto convert_op = [](std::size_t val) -> result_type {
        return make_tuple(val, val);
    };

    result_type const init = make_tuple(std::size_t(1), std::size_t(1));

    result_type r1 = pika::transform_reduce(iterator(std::begin(c)),
        iterator(std::end(c)), init, reduce_op, convert_op);

    // verify values
    result_type r2 = std::accumulate(std::begin(c), std::end(c), init,
        [&reduce_op, &convert_op](result_type res, std::size_t val) {
            return reduce_op(res, convert_op(val));
        });

    PIKA_TEST_EQ(get<0>(r1), get<0>(r2));
    PIKA_TEST_EQ(get<1>(r1), get<1>(r2));
}

template <typename ExPolicy, typename IteratorTag>
void test_transform_reduce(ExPolicy&& policy, IteratorTag)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), std::rand());

    typedef pika::tuple<std::size_t, std::size_t> result_type;

    using pika::get;
    using pika::make_tuple;

    auto reduce_op = [](result_type v1, result_type v2) -> result_type {
        return make_tuple(get<0>(v1) * get<0>(v2), get<1>(v1) * get<1>(v2));
    };

    auto convert_op = [](std::size_t val) -> result_type {
        return make_tuple(val, val);
    };

    result_type const init = make_tuple(std::size_t(1), std::size_t(1));

    result_type r1 = pika::transform_reduce(policy, iterator(std::begin(c)),
        iterator(std::end(c)), init, reduce_op, convert_op);

    // verify values
    result_type r2 = std::accumulate(std::begin(c), std::end(c), init,
        [&reduce_op, &convert_op](result_type res, std::size_t val) {
            return reduce_op(res, convert_op(val));
        });

    PIKA_TEST_EQ(get<0>(r1), get<0>(r2));
    PIKA_TEST_EQ(get<1>(r1), get<1>(r2));
}

template <typename ExPolicy, typename IteratorTag>
void test_transform_reduce_async(ExPolicy&& p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), std::rand());

    std::size_t val(42);
    auto op = [val](std::size_t v1, std::size_t v2) { return v1 + v2 + val; };

    pika::future<std::size_t> f =
        pika::transform_reduce(p, iterator(std::begin(c)), iterator(std::end(c)),
            val, op, [](std::size_t v) { return v; });
    f.wait();

    // verify values
    std::size_t r2 = std::accumulate(std::begin(c), std::end(c), val, op);
    PIKA_TEST_EQ(f.get(), r2);
}

template <typename IteratorTag>
void test_transform_reduce()
{
    using namespace pika::execution;

    test_transform_reduce(IteratorTag());

    test_transform_reduce(seq, IteratorTag());
    test_transform_reduce(par, IteratorTag());
    test_transform_reduce(par_unseq, IteratorTag());

    test_transform_reduce_async(seq(task), IteratorTag());
    test_transform_reduce_async(par(task), IteratorTag());
}

void transform_reduce_test()
{
    test_transform_reduce<std::random_access_iterator_tag>();
    test_transform_reduce<std::forward_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_transform_reduce_exception(IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), std::rand());

    bool caught_exception = false;
    try
    {
        pika::transform_reduce(
            iterator(std::begin(c)), iterator(std::end(c)), std::size_t(42),
            [](std::size_t v1, std::size_t v2) {
                return throw std::runtime_error("test"), v1 + v2;
            },
            [](std::size_t v) { return v; });

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
void test_transform_reduce_exception(ExPolicy&& policy, IteratorTag)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), std::rand());

    bool caught_exception = false;
    try
    {
        pika::transform_reduce(
            policy, iterator(std::begin(c)), iterator(std::end(c)),
            std::size_t(42),
            [](std::size_t v1, std::size_t v2) {
                return throw std::runtime_error("test"), v1 + v2;
            },
            [](std::size_t v) { return v; });

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
void test_transform_reduce_exception_async(ExPolicy&& p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), std::rand());

    bool caught_exception = false;
    bool returned_from_algorithm = false;
    try
    {
        pika::future<void> f = pika::transform_reduce(
            p, iterator(std::begin(c)), iterator(std::end(c)), std::size_t(42),
            [](std::size_t v1, std::size_t v2) {
                return throw std::runtime_error("test"), v1 + v2;
            },
            [](std::size_t v) { return v; });
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
void test_transform_reduce_exception()
{
    using namespace pika::execution;

    test_transform_reduce_exception(IteratorTag());

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_transform_reduce_exception(seq, IteratorTag());
    test_transform_reduce_exception(par, IteratorTag());

    test_transform_reduce_exception_async(seq(task), IteratorTag());
    test_transform_reduce_exception_async(par(task), IteratorTag());
}

void transform_reduce_exception_test()
{
    test_transform_reduce_exception<std::random_access_iterator_tag>();
    test_transform_reduce_exception<std::forward_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_transform_reduce_bad_alloc(IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), std::rand());

    bool caught_exception = false;
    try
    {
        pika::transform_reduce(
            iterator(std::begin(c)), iterator(std::end(c)), std::size_t(42),
            [](std::size_t v1, std::size_t v2) {
                return throw std::bad_alloc(), v1 + v2;
            },
            [](std::size_t v) { return v; });

        PIKA_TEST(false);
    }
    catch (std::bad_alloc const&)
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
void test_transform_reduce_bad_alloc(ExPolicy&& policy, IteratorTag)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), std::rand());

    bool caught_exception = false;
    try
    {
        pika::transform_reduce(
            policy, iterator(std::begin(c)), iterator(std::end(c)),
            std::size_t(42),
            [](std::size_t v1, std::size_t v2) {
                return throw std::bad_alloc(), v1 + v2;
            },
            [](std::size_t v) { return v; });

        PIKA_TEST(false);
    }
    catch (std::bad_alloc const&)
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
void test_transform_reduce_bad_alloc_async(ExPolicy&& p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), std::rand());

    bool caught_exception = false;
    bool returned_from_algorithm = false;
    try
    {
        pika::future<void> f = pika::transform_reduce(
            p, iterator(std::begin(c)), iterator(std::end(c)), std::size_t(42),
            [](std::size_t v1, std::size_t v2) {
                return throw std::bad_alloc(), v1 + v2;
            },
            [](std::size_t v) { return v; });
        returned_from_algorithm = true;
        f.get();

        PIKA_TEST(false);
    }
    catch (std::bad_alloc const&)
    {
        caught_exception = true;
    }
    catch (...)
    {
        PIKA_TEST(false);
    }

    PIKA_TEST(caught_exception);
    PIKA_TEST(returned_from_algorithm);
}

template <typename IteratorTag>
void test_transform_reduce_bad_alloc()
{
    using namespace pika::execution;

    test_transform_reduce_bad_alloc(IteratorTag());

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_transform_reduce_bad_alloc(seq, IteratorTag());
    test_transform_reduce_bad_alloc(par, IteratorTag());

    test_transform_reduce_bad_alloc_async(seq(task), IteratorTag());
    test_transform_reduce_bad_alloc_async(par(task), IteratorTag());
}

void transform_reduce_bad_alloc_test()
{
    test_transform_reduce_bad_alloc<std::random_access_iterator_tag>();
    test_transform_reduce_bad_alloc<std::forward_iterator_tag>();
}

int pika_main(pika::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int) std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    transform_reduce_test();
    transform_reduce_bad_alloc_test();
    transform_reduce_exception_test();

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

    //By default run on all available cores
    std::vector<std::string> const cfg = {"pika.os_threads=all"};

    // Initialize and run pika
    pika::local::init_params init_args;
    init_args.desc_cmdline = desc_commandline;
    init_args.cfg = cfg;

    PIKA_TEST_EQ_MSG(pika::local::init(pika_main, argc, argv, init_args), 0,
        "pika main exited with non-zero status");

    return pika::util::report_errors();
}
