//  Taken from the Boost.Function library

//  Copyright Douglas Gregor 2001-2003.
//  Copyright 2013 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Use, modification and
//  distribution is subject to the Boost Software License, Version
//  1.0. (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

// For more information, see http://www.boost.org

// NOTE: Warning caused by assignment of pika::util::detail::function<float()> to
// pika::util::detail::function<double()> in test_emptiness. Triggered in
// pika/functional/function.hpp which is included latest by pika/include/util.hpp.
#if defined(__clang__)
# pragma clang diagnostic push
# pragma clang diagnostic ignored "-Wdouble-promotion"
#elif defined(__GNUC__)
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wdouble-promotion"
#endif
#include <pika/functional/function.hpp>
#if defined(__clang__)
# pragma clang diagnostic pop
#elif defined(__GNUC__)
# pragma GCC diagnostic pop
#endif

#include <pika/testing.hpp>

#include <functional>
#include <string>
#include <utility>

using std::string;

int global_int;

struct write_five_obj
{
    void operator()() const { global_int = 5; }
};
struct write_three_obj
{
    int operator()() const
    {
        global_int = 3;
        return 7;
    }
};
static void write_five() { global_int = 5; }
static void write_three() { global_int = 3; }
struct generate_five_obj
{
    int operator()() const { return 5; }
};
struct generate_three_obj
{
    int operator()() const { return 3; }
};
static int generate_five() { return 5; }
static int generate_three() { return 3; }
static string identity_str(string const& s) { return s; }
static string string_cat(string const& s1, string const& s2) { return s1 + s2; }
static int sum_ints(int x, int y) { return x + y; }

struct write_const_1_nonconst_2
{
    void operator()() { global_int = 2; }
    void operator()() const { global_int = 1; }
};

struct add_to_obj
{
    add_to_obj(int v)
      : value(v)
    {
    }

    int operator()(int x) const { return value + x; }

    int value;
};

static void test_zero_args()
{
    using func_void_type = pika::util::detail::function<void()>;

    write_five_obj five;
    write_three_obj three;

    // Default construction
    func_void_type v1;
    PIKA_TEST(v1.empty());

    // Assignment to an empty function
    v1 = five;
    PIKA_TEST(!v1.empty());

    // Invocation of a function
    global_int = 0;
    v1();
    PIKA_TEST_EQ(global_int, 5);

    // reset() method
    v1.reset();
    PIKA_TEST(v1.empty());

    // Assignment to an empty function
    v1 = three;
    PIKA_TEST(!v1.empty());

    // Invocation and self-assignment
    global_int = 0;

#if defined(__clang__)
# pragma clang diagnostic push
# pragma clang diagnostic ignored "-Wself-assign-overloaded"
#endif
    v1 = v1;
#if defined(__clang__)
# pragma clang diagnostic pop
#endif

    v1();
    PIKA_TEST_EQ(global_int, 3);

    // Assignment to a non-empty function
    v1 = five;

    // Invocation and self-assignment
    global_int = 0;

#if defined(__clang__)
# pragma clang diagnostic push
# pragma clang diagnostic ignored "-Wself-assign-overloaded"
#endif
    v1 = (v1);
#if defined(__clang__)
# pragma clang diagnostic pop
#endif

    v1();
    PIKA_TEST_EQ(global_int, 5);

    // clear
    void (*fpv1)() = 0;    // NOLINT
    v1 = fpv1;
    PIKA_TEST(v1.empty());

    v1 = write_five;
    v1 = nullptr;
    PIKA_TEST(v1.empty());

    // Assignment to an empty function from a free function
    v1 = write_five;
    PIKA_TEST(!v1.empty());

    // Invocation
    global_int = 0;
    v1();
    PIKA_TEST_EQ(global_int, 5);

    // Assignment to a non-empty function from a free function
    v1 = write_three;
    PIKA_TEST(!v1.empty());

    // Invocation
    global_int = 0;
    v1();
    PIKA_TEST_EQ(global_int, 3);

    // Assignment
    v1 = five;
    PIKA_TEST(!v1.empty());

    // Invocation
    global_int = 0;
    v1();
    PIKA_TEST_EQ(global_int, 5);

    // Assignment to a non-empty function from a free function
    v1 = &write_three;
    PIKA_TEST(!v1.empty());

    // Invocation
    global_int = 0;
    v1();
    PIKA_TEST_EQ(global_int, 3);

    // Construction from another function (that is empty)
    v1.reset();
    func_void_type v2(v1);
    PIKA_TEST(v2.empty());

    // Assignment to an empty function
    v2 = three;
    PIKA_TEST(!v2.empty());

    // Invocation
    global_int = 0;
    v2();
    PIKA_TEST_EQ(global_int, 3);

    // Assignment to a non-empty function
    v2 = (five);

    // Invocation
    global_int = 0;
    v2();
    PIKA_TEST_EQ(global_int, 5);

    v2.reset();
    PIKA_TEST(v2.empty());

    // Assignment to an empty function from a free function
    v2 = (write_five);
    PIKA_TEST(!v2.empty());

    // Invocation
    global_int = 0;
    v2();
    PIKA_TEST_EQ(global_int, 5);

    // Assignment to a non-empty function from a free function
    v2 = write_three;
    PIKA_TEST(!v2.empty());

    // Invocation
    global_int = 0;
    v2();
    PIKA_TEST_EQ(global_int, 3);

    // Swapping
    v1 = five;
    std::swap(v1, v2);
    v2();
    PIKA_TEST_EQ(global_int, 5);
    v1();
    PIKA_TEST_EQ(global_int, 3);
    std::swap(v1, v2);
    v1.reset();

    // Assignment
    v2 = five;
    PIKA_TEST(!v2.empty());

    // Invocation
    global_int = 0;
    v2();
    PIKA_TEST_EQ(global_int, 5);

    // Assignment to a non-empty function from a free function
    v2 = &write_three;
    PIKA_TEST(!v2.empty());

    // Invocation
    global_int = 0;
    v2();
    PIKA_TEST_EQ(global_int, 3);

    // Assignment to a function from an empty function
    v2 = v1;
    PIKA_TEST(v2.empty());

    // Assignment to a function from a function with a functor
    v1 = three;
    v2 = v1;
    PIKA_TEST(!v1.empty());
    PIKA_TEST(!v2.empty());

    // Invocation
    global_int = 0;
    v1();
    PIKA_TEST_EQ(global_int, 3);
    global_int = 0;
    v2();
    PIKA_TEST_EQ(global_int, 3);

    // Assign to a function from a function with a function
    v2 = write_five;
    v1 = v2;
    PIKA_TEST(!v1.empty());
    PIKA_TEST(!v2.empty());
    global_int = 0;
    v1();
    PIKA_TEST_EQ(global_int, 5);
    global_int = 0;
    v2();
    PIKA_TEST_EQ(global_int, 5);

    // Construct a function given another function containing a function
    func_void_type v3(v1);

    // Invocation of a function
    global_int = 0;
    v3();
    PIKA_TEST_EQ(global_int, 5);

    // reset() method
    v3.reset();
    PIKA_TEST(!v3);

    // Assignment to an empty function
    v3 = three;
    PIKA_TEST(!v3.empty());

    // Invocation
    global_int = 0;
    v3();
    PIKA_TEST_EQ(global_int, 3);

    // Assignment to a non-empty function
    v3 = five;

    // Invocation
    global_int = 0;
    v3();
    PIKA_TEST_EQ(global_int, 5);

    // reset()
    v3.reset();
    PIKA_TEST(v3.empty());

    // Assignment to an empty function from a free function
    v3 = &write_five;
    PIKA_TEST(!v3.empty());

    // Invocation
    global_int = 0;
    v3();
    PIKA_TEST_EQ(global_int, 5);

    // Assignment to a non-empty function from a free function
    v3 = &write_three;
    PIKA_TEST(!v3.empty());

    // Invocation
    global_int = 0;
    v3();
    PIKA_TEST_EQ(global_int, 3);

    // Assignment
    v3 = five;
    PIKA_TEST(!v3.empty());

    // Invocation
    global_int = 0;
    v3();
    PIKA_TEST_EQ(global_int, 5);

    // Construction of a function from a function containing a functor
    func_void_type v4(v3);

    // Invocation of a function
    global_int = 0;
    v4();
    PIKA_TEST_EQ(global_int, 5);

    // reset() method
    v4.reset();
    PIKA_TEST(v4.empty());

    // Assignment to an empty function
    v4 = three;
    PIKA_TEST(!v4.empty());

    // Invocation
    global_int = 0;
    v4();
    PIKA_TEST_EQ(global_int, 3);

    // Assignment to a non-empty function
    v4 = five;

    // Invocation
    global_int = 0;
    v4();
    PIKA_TEST_EQ(global_int, 5);

    // reset()
    v4.reset();
    PIKA_TEST(v4.empty());

    // Assignment to an empty function from a free function
    v4 = &write_five;
    PIKA_TEST(!v4.empty());

    // Invocation
    global_int = 0;
    v4();
    PIKA_TEST_EQ(global_int, 5);

    // Assignment to a non-empty function from a free function
    v4 = &write_three;
    PIKA_TEST(!v4.empty());

    // Invocation
    global_int = 0;
    v4();
    PIKA_TEST_EQ(global_int, 3);

    // Assignment
    v4 = five;
    PIKA_TEST(!v4.empty());

    // Invocation
    global_int = 0;
    v4();
    PIKA_TEST_EQ(global_int, 5);

    // Construction of a function from a functor
    func_void_type v5(five);

    // Invocation of a function
    global_int = 0;
    v5();
    PIKA_TEST_EQ(global_int, 5);

    // reset() method
    v5.reset();
    PIKA_TEST(v5.empty());

    // Assignment to an empty function
    v5 = three;
    PIKA_TEST(!v5.empty());

    // Invocation
    global_int = 0;
    v5();
    PIKA_TEST_EQ(global_int, 3);

    // Assignment to a non-empty function
    v5 = five;

    // Invocation
    global_int = 0;
    v5();
    PIKA_TEST_EQ(global_int, 5);

    // reset()
    v5.reset();
    PIKA_TEST(v5.empty());

    // Assignment to an empty function from a free function
    v5 = &write_five;
    PIKA_TEST(!v5.empty());

    // Invocation
    global_int = 0;
    v5();
    PIKA_TEST_EQ(global_int, 5);

    // Assignment to a non-empty function from a free function
    v5 = &write_three;
    PIKA_TEST(!v5.empty());

    // Invocation
    global_int = 0;
    v5();
    PIKA_TEST_EQ(global_int, 3);

    // Assignment
    v5 = five;
    PIKA_TEST(!v5.empty());

    // Invocation
    global_int = 0;
    v5();
    PIKA_TEST_EQ(global_int, 5);

    // Construction of a function from a function
    func_void_type v6(&write_five);

    // Invocation of a function
    global_int = 0;
    v6();
    PIKA_TEST_EQ(global_int, 5);

    // reset() method
    v6.reset();
    PIKA_TEST(v6.empty());

    // Assignment to an empty function
    v6 = three;
    PIKA_TEST(!v6.empty());

    // Invocation
    global_int = 0;
    v6();
    PIKA_TEST_EQ(global_int, 3);

    // Assignment to a non-empty function
    v6 = five;

    // Invocation
    global_int = 0;
    v6();
    PIKA_TEST_EQ(global_int, 5);

    // reset()
    v6.reset();
    PIKA_TEST(v6.empty());

    // Assignment to an empty function from a free function
    v6 = &write_five;
    PIKA_TEST(!v6.empty());

    // Invocation
    global_int = 0;
    v6();
    PIKA_TEST_EQ(global_int, 5);

    // Assignment to a non-empty function from a free function
    v6 = &write_three;
    PIKA_TEST(!v6.empty());

    // Invocation
    global_int = 0;
    v6();
    PIKA_TEST_EQ(global_int, 3);

    // Assignment
    v6 = five;
    PIKA_TEST(!v6.empty());

    // Invocation
    global_int = 0;
    v6();
    PIKA_TEST_EQ(global_int, 5);

    // Const vs. non-const
    write_const_1_nonconst_2 one_or_two;
    pika::util::detail::function<void()> const v7(one_or_two);
    pika::util::detail::function<void()> v8(one_or_two);

    global_int = 0;
    v7();
    PIKA_TEST_EQ(global_int, 2);

    global_int = 0;
    v8();
    PIKA_TEST_EQ(global_int, 2);

    // Test construction from 0
    void (*fpv9)() = 0;    // NOLINT
    func_void_type v9(fpv9);
    PIKA_TEST(v9.empty());

    // Test construction from nullptr
    func_void_type v9np(nullptr);
    PIKA_TEST(v9np.empty());

    // Test return values
    using func_int_type = pika::util::detail::function<int()>;
    generate_five_obj gen_five;
    generate_three_obj gen_three;

    func_int_type i0(gen_five);

    PIKA_TEST_EQ(i0(), 5);
    i0 = gen_three;
    PIKA_TEST_EQ(i0(), 3);
    i0 = &generate_five;
    PIKA_TEST_EQ(i0(), 5);
    i0 = &generate_three;
    PIKA_TEST_EQ(i0(), 3);
    PIKA_TEST(!i0.empty());
    i0.reset();
    PIKA_TEST(!i0);

    // Test return values with compatible types
    using func_long_type = pika::util::detail::function<long()>;
    func_long_type i1(gen_five);

    PIKA_TEST_EQ(i1(), 5);
    i1 = gen_three;
    PIKA_TEST_EQ(i1(), 3);
    i1 = &generate_five;
    PIKA_TEST_EQ(i1(), 5);
    i1 = &generate_three;
    PIKA_TEST_EQ(i1(), 3);
    PIKA_TEST(!i1.empty());
    i1.reset();
    PIKA_TEST(!i1);
}

static void test_one_arg()
{
    std::negate<int> neg;

    pika::util::detail::function<int(int)> f1(neg);
    PIKA_TEST_EQ(f1(5), -5);

    pika::util::detail::function<string(string)> id(&identity_str);
    PIKA_TEST_EQ(id("str"), "str");

    pika::util::detail::function<string(char const*)> id2(&identity_str);
    PIKA_TEST_EQ(id2("foo"), "foo");

    add_to_obj add_to(5);
    pika::util::detail::function<int(int)> f2(add_to);
    PIKA_TEST_EQ(f2(3), 8);

    pika::util::detail::function<int(int)> const cf2(add_to);
    PIKA_TEST_EQ(cf2(3), 8);
}

static void test_two_args()
{
    pika::util::detail::function<string(string const&, string const&)> cat(&string_cat);
    PIKA_TEST_EQ(cat("str", "ing"), "string");

    pika::util::detail::function<int(short, short)> sum(&sum_ints);
    PIKA_TEST_EQ(sum(2, 3), 5);
}

static void test_emptiness()
{
    pika::util::detail::function<float()> f1;
    PIKA_TEST(f1.empty());

    pika::util::detail::function<float()> f2;
    f2 = f1;
    PIKA_TEST(f2.empty());

    pika::util::detail::function<double()> f3;
    f3 = f2;
    PIKA_TEST(f3.empty());
}

struct X
{
    X(int v)
      : value(v)
    {
    }

    int twice() const { return 2 * value; }
    int plus(int v) { return value + v; }

    int value;
};

static void test_member_functions()
{
    pika::util::detail::function<int(X*)> f1(&X::twice);

    X one(1);
    X five(5);

    PIKA_TEST_EQ(f1(&one), 2);
    PIKA_TEST_EQ(f1(&five), 10);

    pika::util::detail::function<int(X*)> f1_2;
    f1_2 = &X::twice;

    PIKA_TEST_EQ(f1_2(&one), 2);
    PIKA_TEST_EQ(f1_2(&five), 10);

    pika::util::detail::function<int(X&, int)> f2(&X::plus);
    PIKA_TEST_EQ(f2(one, 3), 4);
    PIKA_TEST_EQ(f2(five, 4), 9);
}

struct add_with_throw_on_copy
{
    int operator()(int x, int y) const { return x + y; }

    add_with_throw_on_copy() {}

    add_with_throw_on_copy(add_with_throw_on_copy const&)
    {
        throw std::runtime_error("But this CAN'T throw");
    }

    add_with_throw_on_copy& operator=(add_with_throw_on_copy const&)
    {
        throw std::runtime_error("But this CAN'T throw");
    }
};

static void test_ref()
{
    add_with_throw_on_copy atc;
    try
    {
        pika::util::detail::function<int(int, int)> f(std::ref(atc));
        PIKA_TEST_EQ(f(1, 3), 4);
    }
    catch (std::runtime_error const& /*e*/)
    {
        PIKA_TEST_MSG(false, "Nonthrowing constructor threw an exception");
    }
}

static void dummy() {}

static void test_empty_ref()
{
    pika::util::detail::function<void()> f1;
    pika::util::detail::function<void()> f2(std::ref(f1));

    try
    {
        f2();
        PIKA_TEST_MSG(false, "Exception didn't throw for reference to empty function.");
    }
    // NOLINTNEXTLINE(bugprone-empty-catch)
    catch (std::runtime_error const& /*e*/)
    {
    }

    f1 = dummy;

    try
    {
        f2();
    }
    catch (std::runtime_error const&)
    {
        PIKA_TEST_MSG(false, "Error calling referenced function.");
    }
}

static void test_exception()
{
    pika::util::detail::function<int(int, int)> f;
    try
    {
        f(5, 4);
        PIKA_TEST(false);
    }
    // NOLINTNEXTLINE(bugprone-empty-catch)
    catch (std::runtime_error const&)
    {
        // okay
    }
}

using reader_type = pika::util::detail::function<void*(void* reader)>;
using mapped_type = std::pair<int, reader_type>;

static void test_implicit()
{
    mapped_type m;
    m = mapped_type();
}

static void test_call_obj(pika::util::detail::function<int(int, int)> f) { PIKA_TEST(!f.empty()); }

static void test_call_cref(pika::util::detail::function<int(int, int)> const& f)
{
    PIKA_TEST(!f.empty());
}

static void test_call()
{
    test_call_obj(std::plus<int>());
    test_call_cref(std::plus<int>());
}

struct big_aggregating_structure
{
    int disable_small_objects_optimizations[32];

    big_aggregating_structure() { ++global_int; }

    big_aggregating_structure(big_aggregating_structure const&) { ++global_int; }

    ~big_aggregating_structure() { --global_int; }

    void operator()() { ++global_int; }

    void operator()(int) { ++global_int; }
};

template <class FunctionT>
static void test_move_semantics()
{
    using f1_type = FunctionT;

    big_aggregating_structure obj;

    f1_type f1 = obj;
    global_int = 0;
    f1();

    PIKA_TEST(!f1.empty());
    PIKA_TEST_EQ(global_int, 1);

    // Testing rvalue constructors
    f1_type f2(static_cast<f1_type&&>(f1));
    PIKA_TEST(f1.empty());
    PIKA_TEST(!f2.empty());
    PIKA_TEST_EQ(global_int, 1);
    f2();
    PIKA_TEST_EQ(global_int, 2);

    f1_type f3(static_cast<f1_type&&>(f2));
    PIKA_TEST(f1.empty());
    PIKA_TEST(f2.empty());
    PIKA_TEST(!f3.empty());
    PIKA_TEST_EQ(global_int, 2);
    f3();
    PIKA_TEST_EQ(global_int, 3);

    // Testing move assignment
    f1_type f4;
    PIKA_TEST(f4.empty());
    f4 = static_cast<f1_type&&>(f3);
    PIKA_TEST(f1.empty());
    PIKA_TEST(f2.empty());
    PIKA_TEST(f3.empty());
    PIKA_TEST(!f4.empty());
    PIKA_TEST_EQ(global_int, 3);
    f4();
    PIKA_TEST_EQ(global_int, 4);

    // Testing self move assignment
    f4 = static_cast<f1_type&&>(f4);
    PIKA_TEST(!f4.empty());
    PIKA_TEST_EQ(global_int, 4);

    // Testing, that no memory leaked when assigning to nonempty function
    f4 = obj;
    PIKA_TEST(!f4.empty());
    PIKA_TEST_EQ(global_int, 4);
    f1_type f5 = obj;
    PIKA_TEST_EQ(global_int, 5);
    f4 = static_cast<f1_type&&>(f5);
    PIKA_TEST_EQ(global_int, 4);
}

int main(int, char*[])
{
    test_zero_args();
    test_one_arg();
    test_two_args();
    test_emptiness();
    test_member_functions();
    test_ref();
    test_empty_ref();
    test_exception();
    test_implicit();
    test_call();
    test_move_semantics<pika::util::detail::function<void()>>();

    return 0;
}
