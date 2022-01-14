//  Copyright (c) 2012 Bryce Adelstein-Lelbach
//  Copyright (c) 2014 Agustin Berge
//  Copyright (c) 2017 Denis Blank
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/future.hpp>
#include <pika/local/init.hpp>
#include <pika/local/thread.hpp>
#include <pika/local/tuple.hpp>
#include <pika/modules/testing.hpp>
#include <pika/pack_traversal/unwrap.hpp>

#include <array>
#include <atomic>
#include <cstddef>
#include <numeric>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using pika::get;
using pika::make_tuple;
using pika::tuple;
using pika::unwrap;
using pika::unwrap_all;
using pika::unwrap_n;
using pika::unwrapping;
using pika::unwrapping_all;
using pika::unwrapping_n;

/// Since the mapping functionality is provided by the `map_pack`
/// API, we only need to test here for the specific behaviour of `unwrap`
/// and its variations.
/// All the test functions are instantiated for testing against future
/// and shared_future (which differ in their return type -> `T` vs `T const&`).

template <template <typename> class FutureType, typename FutureProvider>
void test_unwrap(FutureProvider&& futurize)
{
    // Single values are unwrapped
    {
        int res = unwrap(futurize(0xDD));
        PIKA_TEST_EQ(res, 0xDD);
    }

    // Futures with tuples may be unwrapped
    {
        tuple<int, int> res = unwrap(futurize(make_tuple(0xDD, 0xDF)));
        PIKA_TEST(res == make_tuple(0xDD, 0xDF));
    }

    // The value of multiple futures is returned inside a tuple
    {
        tuple<int, int> res = unwrap(futurize(0xDD), futurize(0xDF));
        PIKA_TEST(res == make_tuple(0xDD, 0xDF));
    }
}

template <template <typename> class FutureType, typename FutureProvider>
void test_unwrap_n(FutureProvider&& futurize)
{
    // Single values are unwrapped
    {
        int res = unwrap_n<2>(futurize(futurize(0xDD)));
        PIKA_TEST_EQ(res, 0xDD);
    }

    // Futures with tuples may be unwrapped
    {
        tuple<int, int> res =
            unwrap_n<2>(futurize(futurize(make_tuple(0xDD, 0xDF))));
        PIKA_TEST(res == make_tuple(0xDD, 0xDF));
    }

    // The value of multiple futures is returned inside a tuple
    {
        tuple<int, int> res =
            unwrap_n<2>(futurize(futurize(0xDD)), futurize(futurize(0xDF)));
        PIKA_TEST(res == make_tuple(0xDD, 0xDF));
    }

    // Futures are not unwrapped beyond the given depth
    {
        FutureType<int> res =
            unwrap_n<3>(futurize(futurize(futurize(futurize(0xDD)))));
        PIKA_TEST(res.get() == 0xDD);
    }
}

template <template <typename> class FutureType, typename FutureProvider>
void test_unwrap_all(FutureProvider&& futurize)
{
    // Single values are unwrapped
    {
        int res =
            unwrap_all(futurize(futurize(futurize(futurize(futurize(0xDD))))));
        PIKA_TEST_EQ(res, 0xDD);
    }

    // Futures with tuples may be unwrapped
    {
        tuple<int, int> res = unwrap_all(futurize(
            futurize(futurize(futurize(make_tuple(futurize(futurize(0xDD)),
                futurize(futurize(futurize(0xDF)))))))));
        PIKA_TEST(res == make_tuple(0xDD, 0xDF));
    }

    // The value of multiple futures is returned inside a tuple
    {
        tuple<int, int> res = unwrap_all(
            futurize(futurize(futurize(futurize(0xDD)))), futurize(0xDF));
        PIKA_TEST(res == make_tuple(0xDD, 0xDF));
    }
}

template <template <typename> class FutureType, typename FutureProvider>
void test_unwrapping(FutureProvider&& futurize)
{
    // One argument is passed without a tuple unwrap
    {
        auto unwrapper = unwrapping([](int a) { return a; });

        int res = unwrapper(futurize(3));

        PIKA_TEST_EQ(res, 3);
    }

    /// Don't unpack single tuples which were passed to the functional unwrap
    {
        auto unwrapper = unwrapping([](tuple<int, int> arg) {
            // ...
            return get<0>(arg) + get<1>(arg);
        });

        int res = unwrapper(futurize(make_tuple(1, 2)));

        PIKA_TEST_EQ(res, 3);
    }

    // Multiple arguments are spread across the callable
    {
        auto unwrapper = unwrapping([](int a, int b) { return a + b; });

        int res = unwrapper(futurize(1), futurize(2));

        PIKA_TEST_EQ(res, 3);
    }
}

/// A callable object which helps us to test deferred unwrapping like the
/// the immediate unwrap, because we materialize the arguments back.
struct back_materializer
{
    void operator()() const {}

    template <typename First>
    First operator()(First&& first) const
    {
        return std::forward<First>(first);
    }

    template <typename First, typename Second, typename... Rest>
    tuple<First, Second, Rest...> operator()(
        First&& first, Second&& second, Rest&&... rest) const
    {
        return tuple<First, Second, Rest...>{std::forward<First>(first),
            std::forward<Second>(second), std::forward<Rest>(rest)...};
    }
};

template <template <typename> class FutureType, typename FutureProvider>
void test_unwrapping_n(FutureProvider&& futurize)
{
    // Single values are unwrapped
    {
        int res =
            unwrapping_n<2>(back_materializer{})(futurize(futurize(0xDD)));
        PIKA_TEST_EQ(res, 0xDD);
    }

    // Futures with tuples may be unwrapped
    {
        tuple<int, int> res = unwrapping_n<2>(back_materializer{})(
            futurize(futurize(make_tuple(0xDD, 0xDF))));
        PIKA_TEST(res == make_tuple(0xDD, 0xDF));
    }

    // The value of multiple futures is returned inside a tuple
    {
        tuple<int, int> res = unwrapping_n<2>(back_materializer{})(
            futurize(futurize(0xDD)), futurize(futurize(0xDF)));
        PIKA_TEST(res == make_tuple(0xDD, 0xDF));
    }

    // Futures are not unwrapped beyond the given depth
    {
        FutureType<int> res = unwrapping_n<3>(back_materializer{})(
            futurize(futurize(futurize(futurize(0xDD)))));
        PIKA_TEST_EQ(res.get(), 0xDD);
    }
}

template <template <typename> class FutureType, typename FutureProvider>
void test_unwrapping_all(FutureProvider&& futurize)
{
    // Single values are unwrapped
    {
        int res = unwrapping_all(back_materializer{})(
            futurize(futurize(futurize(futurize(futurize(0xDD))))));
        PIKA_TEST_EQ(res, 0xDD);
    }

    // Futures with tuples may be unwrapped
    {
        tuple<int, int> res = unwrapping_all(
            back_materializer{})(futurize(futurize(futurize(futurize(make_tuple(
            futurize(futurize(0xDD)), futurize(futurize(futurize(0xDF)))))))));
        PIKA_TEST(res == make_tuple(0xDD, 0xDF));
    }

    // The value of multiple futures is returned inside a tuple
    {
        tuple<int, int> res = unwrapping_all(back_materializer{})(
            futurize(futurize(futurize(futurize(0xDD)))), futurize(0xDF));
        PIKA_TEST(res == make_tuple(0xDD, 0xDF));
    }
}

/// This section declare some unit tests which ensure that issues
/// that occurred while developing the implementation were fixed.
template <template <typename> class FutureType, typename FutureProvider>
void test_development_regressions(FutureProvider&& futurize)
{
    // A regression originally taken from the unwrapped tests
    {
        auto increment = static_cast<int (*)(int)>([](int c) -> int {
            // ...
            return c + 1;
        });

        FutureType<int> future = futurize(42);
        PIKA_TEST_EQ(unwrapping(increment)(future), 42 + 1);
    }

    // A future is mapped to its value
    {
        std::vector<FutureType<int>> f;
        std::vector<int> res = unwrap(f);

        PIKA_TEST(res.empty());
    }

    // A single void future is mapped empty
    {
        FutureType<void> f = futurize();
        using Result = decltype(unwrap(f));
        static_assert(std::is_void<Result>::value, "Failed...");
    }

    // Multiple void futures are mapped empty
    {
        FutureType<void> f1 = futurize();
        FutureType<void> f2 = futurize();
        FutureType<void> f3 = futurize();
        using Result = decltype(unwrap(f1, f2, f3));
        static_assert(std::is_void<Result>::value, "Failed...");
    }

    // Delete empty futures out variadic packs
    {
        FutureType<void> f = futurize();

        auto callable = unwrapping([](int a, int b) {
            // ...
            return a + b;
        });

        PIKA_TEST_EQ(callable(1, f, 2), 3);
    }

    // Call callables with no arguments if the pack was mapped empty.
    // Based on a build failure in local_dataflow_executor_v1.cpp.
    {
        FutureType<void> f = futurize();

        auto callable = unwrapping([]() {
            // ...
            return true;
        });

        PIKA_TEST(callable(f));
    }

    // Map empty mappings back to void, if an empty mapping was propagated back.
    // Based on a build failure in global_spmd_block.cpp.
    {
        std::vector<FutureType<void>> vec;
        (void) vec;
        using Result = decltype(unwrap(vec));
        static_assert(std::is_void<Result>::value, "Failed...");
    }

    // Unwrap single tuple like types back.
    {
        std::array<FutureType<int>, 2> in{{futurize(1), futurize(2)}};

        std::array<int, 2> result = unwrap(in);

        PIKA_TEST((result == std::array<int, 2>{{1, 2}}));
    }

    // Don't unwrap single tuple like types which were passed to the
    // unwrapping callable object.
    {
        auto unwrapper = unwrapping([](std::array<int, 2> in) {
            // ...
            return in[0] + in[1];
        });

        std::array<FutureType<int>, 2> in{{futurize(1), futurize(2)}};

        PIKA_TEST_EQ(unwrapper(in), 3);
    }
}

/// The unit test for the original impleementation `unwrapped`
/// which was overtaken by `unwrap` and `unwrapping`.
/// Most of this functionality was taken and adapted from
/// the file: `tests/unit/util/unwrapped.cpp`.
namespace legacy_tests {
    std::atomic<std::size_t> void_counter;

    static void null_thread()
    {
        ++void_counter;
    }

    std::atomic<std::size_t> result_counter;

    static bool null_result_thread()
    {
        ++result_counter;
        return true;
    }

    static int increment(int c)
    {
        return c + 1;
    }

    static int accumulate(std::vector<int> cs)
    {
        return std::accumulate(cs.begin(), cs.end(), 0);
    }

    static int add(tuple<int, int> result)
    {
        return get<0>(result) + get<1>(result);
    }

    template <template <typename> class FutureType, typename FutureProvider>
    void test_legacy_requirements(FutureProvider&& futurize)
    {
        using pika::async;

        {
            // Sync wait, single future, void return.
            {
                unwrap(async(null_thread));

                PIKA_TEST_EQ(1U, void_counter.load());

                void_counter.store(0);
            }

            // Sync wait, single future, non-void return.
            {
                PIKA_TEST_EQ(true, unwrap(async(null_result_thread)));
                PIKA_TEST_EQ(1U, result_counter.load());

                result_counter.store(0);
            }

            // Sync wait, multiple futures, void return.
            {
                unwrap(
                    async(null_thread), async(null_thread), async(null_thread));

                PIKA_TEST_EQ(3U, void_counter.load());

                void_counter.store(0);
            }

            // Sync wait, multiple futures, non-void return.
            {
                tuple<bool, bool, bool> r = unwrap(async(null_result_thread),
                    async(null_result_thread), async(null_result_thread));

                PIKA_TEST_EQ(true, get<0>(r));
                PIKA_TEST_EQ(true, get<1>(r));
                PIKA_TEST_EQ(true, get<2>(r));
                PIKA_TEST_EQ(3U, result_counter.load());

                result_counter.store(0);
            }

            // Sync wait, vector of futures, void return.
            {
                std::vector<FutureType<void>> futures;
                futures.reserve(64);

                for (std::size_t i = 0; i < 64; ++i)
                    futures.push_back(async(null_thread));

                unwrap(futures);

                PIKA_TEST_EQ(64U, void_counter.load());

                void_counter.store(0);
            }

            // Sync wait, array of futures, void return.
            {
                std::array<FutureType<void>, 64> futures;

                for (std::size_t i = 0; i < 64; ++i)
                    futures[i] = async(null_thread);

                unwrap(futures);

                PIKA_TEST_EQ(64U, void_counter.load());

                void_counter.store(0);
            }

            // Sync wait, vector of futures, non-void return.
            {
                std::vector<FutureType<bool>> futures;
                futures.reserve(64);

                std::vector<bool> values;
                values.reserve(64);

                for (std::size_t i = 0; i < 64; ++i)
                    futures.push_back(async(null_result_thread));

                values = unwrap(futures);

                PIKA_TEST_EQ(64U, result_counter.load());

                for (std::size_t i = 0; i < 64; ++i)
                    PIKA_TEST_EQ(true, values[i]);

                result_counter.store(0);
            }

            // Sync wait, array of futures, non-void return.
            {
                std::array<FutureType<bool>, 64> futures;

                for (std::size_t i = 0; i < 64; ++i)
                    futures[i] = async(null_result_thread);

                std::array<bool, 64> values = unwrap(futures);

                PIKA_TEST_EQ(64U, result_counter.load());

                for (std::size_t i = 0; i < 64; ++i)
                    PIKA_TEST_EQ(true, values[i]);

                result_counter.store(0);
            }

            // Sync wait, vector of futures, non-void return ignored.
            {
                std::vector<FutureType<bool>> futures;
                futures.reserve(64);

                for (std::size_t i = 0; i < 64; ++i)
                    futures.push_back(async(null_result_thread));

                unwrap(futures);

                PIKA_TEST_EQ(64U, result_counter.load());

                result_counter.store(0);
            }

            // Sync wait, array of futures, non-void return ignored.
            {
                std::array<FutureType<bool>, 64> futures;

                for (std::size_t i = 0; i < 64; ++i)
                    futures[i] = async(null_result_thread);

                unwrap(futures);

                PIKA_TEST_EQ(64U, result_counter.load());

                result_counter.store(0);
            }

            // Functional wrapper, single future
            {
                FutureType<int> future = futurize(42);

                PIKA_TEST_EQ(unwrapping(&increment)(future), 42 + 1);
            }

            // Functional wrapper, vector of future
            {
                std::vector<FutureType<int>> futures;
                futures.reserve(64);

                for (std::size_t i = 0; i < 64; ++i)
                    futures.push_back(futurize(42));

                PIKA_TEST_EQ(unwrapping(&accumulate)(futures), 42 * 64);
            }

            // Functional wrapper, tuple of future
            {
                tuple<FutureType<int>, FutureType<int>> tuple =
                    make_tuple(futurize(42), futurize(42));

                PIKA_TEST_EQ(unwrapping(&add)(tuple), 42 + 42);
            }

            // Functional wrapper, future of tuple of future
            {
                FutureType<tuple<FutureType<int>, FutureType<int>>>
                    tuple_future =
                        futurize(make_tuple(futurize(42), futurize(42)));

                PIKA_TEST_EQ(unwrapping_n<2>(&add)(tuple_future), 42 + 42);
            }
        }
    }
}    // end namespace legacy_tests

/// A callable object which provides a specific future type
template <template <typename> class FutureType>
struct future_factory
{
    FutureType<void> operator()() const
    {
        return pika::make_ready_future();
    }

    template <typename T>
    FutureType<typename std::decay<T>::type> operator()(T&& value) const
    {
        return pika::make_ready_future(std::forward<T>(value));
    }
};

template <template <typename> class FutureType>
void test_all()
{
    future_factory<FutureType> provider;

    test_unwrap<FutureType>(provider);
    test_unwrap_n<FutureType>(provider);
    test_unwrap_all<FutureType>(provider);

    test_unwrapping<FutureType>(provider);
    test_unwrapping_n<FutureType>(provider);
    test_unwrapping_all<FutureType>(provider);

    test_development_regressions<FutureType>(provider);

    legacy_tests::test_legacy_requirements<FutureType>(provider);
}

int pika_main()
{
    // Test everything using default futures
    test_all<pika::future>();
    // Test everything using shared futures
    test_all<pika::shared_future>();

    return pika::local::finalize();
}

int main(int argc, char* argv[])
{
    // Configure application-specific options.
    pika::program_options::options_description cmdline(
        "usage: " PIKA_APPLICATION_STRING " [options]");

    // We force this test to use several threads by default.
    std::vector<std::string> const cfg = {"pika.os_threads=all"};

    // Initialize and run pika
    pika::local::init_params init_args;
    init_args.desc_cmdline = cmdline;
    init_args.cfg = cfg;

    if (int result = pika::local::init(pika_main, argc, argv, init_args))
    {
        return result;
    }
    // Report errors after pika was finished
    return pika::util::report_errors();
}
