//  Copyright (c) 2013-2015 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/config.hpp>
#include <pika/functional/traits/is_invocable.hpp>
#include <pika/modules/testing.hpp>

struct X
{
    void operator()(int);
};
struct Xc
{
    void operator()(int) const;
};

template <typename T>
struct smart_ptr
{
    T* p;
    T& operator*() const
    {
        return *p;
    }
};

void nullary_function()
{
    typedef void (*f)();
    PIKA_TEST_MSG((pika::is_invocable_v<f> == true), "nullary function");
}

void lambdas()
{
    auto lambda = []() {};

    typedef decltype(lambda) f;
    PIKA_TEST_MSG((pika::is_invocable_v<f> == true), "lambda");
}

void functions_byval_params()
{
    typedef void (*f)(int);
    PIKA_TEST_MSG((pika::is_invocable_v<f, int> == true), "fun-value/value");
    PIKA_TEST_MSG((pika::is_invocable_v<f, int&> == true), "fun-value/lvref");
    PIKA_TEST_MSG(
        (pika::is_invocable_v<f, int const&> == true), "fun-value/const-lvref");
    PIKA_TEST_MSG((pika::is_invocable_v<f, int&&> == true), "fun-value/rvref");
    PIKA_TEST_MSG(
        (pika::is_invocable_v<f, int const&&> == true), "fun-value/const-rvref");

    typedef void (*fc)(int const);
    PIKA_TEST_MSG(
        (pika::is_invocable_v<fc, int> == true), "fun-const-value/value");
    PIKA_TEST_MSG(
        (pika::is_invocable_v<fc, int&> == true), "fun-const-value/lvref");
    PIKA_TEST_MSG((pika::is_invocable_v<fc, int const&> == true),
        "fun-const-value/const-lvref");
    PIKA_TEST_MSG(
        (pika::is_invocable_v<fc, int&&> == true), "fun-const-value/rvref");
    PIKA_TEST_MSG((pika::is_invocable_v<fc, int const&&> == true),
        "fun-const-value/const-rvref");
}

void functions_bylvref_params()
{
    typedef void (*f)(int&);
    PIKA_TEST_MSG((pika::is_invocable_v<f, int> == false), "fun-lvref/value");
    PIKA_TEST_MSG((pika::is_invocable_v<f, int&> == true), "fun-lvref/lvref");
    PIKA_TEST_MSG(
        (pika::is_invocable_v<f, int const&> == false), "fun-lvref/const-lvref");
    PIKA_TEST_MSG((pika::is_invocable_v<f, int&&> == false), "fun-lvref/rvref");
    PIKA_TEST_MSG((pika::is_invocable_v<f, int const&&> == false),
        "fun-lvref/const-rvref");

    typedef void (*fc)(int const&);
    PIKA_TEST_MSG(
        (pika::is_invocable_v<fc, int> == true), "fun-const-lvref/value");
    PIKA_TEST_MSG(
        (pika::is_invocable_v<fc, int&> == true), "fun-const-lvref/lvref");
    PIKA_TEST_MSG((pika::is_invocable_v<fc, int const&> == true),
        "fun-const-lvref/const-lvref");
    PIKA_TEST_MSG(
        (pika::is_invocable_v<fc, int&&> == true), "fun-const-lvref/rvref");
    PIKA_TEST_MSG((pika::is_invocable_v<fc, int const&&> == true),
        "fun-const-lvref/const-rvref");
}

void functions_byrvref_params()
{
    typedef void (*f)(int&&);
    PIKA_TEST_MSG((pika::is_invocable_v<f, int> == true), "fun-rvref/value");
    PIKA_TEST_MSG((pika::is_invocable_v<f, int&> == false), "fun-rvref/lvref");
    PIKA_TEST_MSG(
        (pika::is_invocable_v<f, int const&> == false), "fun-rvref/const-lvref");
    PIKA_TEST_MSG((pika::is_invocable_v<f, int&&> == true), "fun-rvref/rvref");
#if !defined(BOOST_INTEL)
    PIKA_TEST_MSG((pika::is_invocable_v<f, int const&&> == false),
        "fun-rvref/const-rvref");
#endif

    typedef void (*fc)(int const&&);
    PIKA_TEST_MSG(
        (pika::is_invocable_v<fc, int> == true), "fun-const-rvref/value");
    PIKA_TEST_MSG(
        (pika::is_invocable_v<fc, int&> == false), "fun-const-rvref/lvref");
    PIKA_TEST_MSG((pika::is_invocable_v<fc, int const&> == false),
        "fun-const-rvref/const-lvref");
    PIKA_TEST_MSG(
        (pika::is_invocable_v<fc, int&&> == true), "fun-const-rvref/rvref");
    PIKA_TEST_MSG((pika::is_invocable_v<fc, int const&&> == true),
        "fun-const-rvref/const-rvref");
}

void member_function_pointers()
{
    typedef int (X::*f)(double);
    PIKA_TEST_MSG(
        (pika::is_invocable_v<f, X*, float> == true), "mem-fun-ptr/ptr");
    PIKA_TEST_MSG((pika::is_invocable_v<f, X const*, float> == false),
        "mem-fun-ptr/const-ptr");
    PIKA_TEST_MSG(
        (pika::is_invocable_v<f, X&, float> == true), "mem-fun-ptr/lvref");
    PIKA_TEST_MSG((pika::is_invocable_v<f, X const&, float> == false),
        "mem-fun-ptr/const-lvref");
    PIKA_TEST_MSG(
        (pika::is_invocable_v<f, X&&, float> == true), "mem-fun-ptr/rvref");
    PIKA_TEST_MSG((pika::is_invocable_v<f, X const&&, float> == false),
        "mem-fun-ptr/const-rvref");
    PIKA_TEST_MSG((pika::is_invocable_v<f, smart_ptr<X>, float> == true),
        "mem-fun-ptr/smart-ptr");
    PIKA_TEST_MSG((pika::is_invocable_v<f, smart_ptr<X const>, float> == false),
        "mem-fun-ptr/smart-const-ptr");

    typedef int (X::*fc)(double) const;
    PIKA_TEST_MSG(
        (pika::is_invocable_v<fc, X*, float> == true), "const-mem-fun-ptr/ptr");
    PIKA_TEST_MSG((pika::is_invocable_v<fc, X const*, float> == true),
        "const-mem-fun-ptr/const-ptr");
    PIKA_TEST_MSG((pika::is_invocable_v<fc, X&, float> == true),
        "const-mem-fun-ptr/lvref");
    PIKA_TEST_MSG((pika::is_invocable_v<fc, X const&, float> == true),
        "const-mem-fun-ptr/const-lvref");
    PIKA_TEST_MSG((pika::is_invocable_v<fc, X&&, float> == true),
        "const-mem-fun-ptr/rvref");
    PIKA_TEST_MSG((pika::is_invocable_v<fc, X const&&, float> == true),
        "const-mem-fun-ptr/const-rvref");
    PIKA_TEST_MSG((pika::is_invocable_v<fc, smart_ptr<X>, float> == true),
        "const-mem-fun-ptr/smart-ptr");
    PIKA_TEST_MSG((pika::is_invocable_v<fc, smart_ptr<X const>, float> == true),
        "const-mem-fun-ptr/smart-const-ptr");
}

void member_object_pointers()
{
    typedef int(X::*f);
    PIKA_TEST_MSG((pika::is_invocable_v<f, X*> == true), "mem-obj-ptr/ptr");
    PIKA_TEST_MSG(
        (pika::is_invocable_v<f, X const*> == true), "mem-obj-ptr/const-ptr");
    PIKA_TEST_MSG((pika::is_invocable_v<f, X&> == true), "mem-obj-ptr/lvref");
    PIKA_TEST_MSG(
        (pika::is_invocable_v<f, X const&> == true), "mem-obj-ptr/const-lvref");
    PIKA_TEST_MSG((pika::is_invocable_v<f, X&&> == true), "mem-obj-ptr/rvref");
    PIKA_TEST_MSG(
        (pika::is_invocable_v<f, X const&&> == true), "mem-obj-ptr/const-rvref");
    PIKA_TEST_MSG((pika::is_invocable_v<f, smart_ptr<X>> == true),
        "mem-obj-ptr/smart-ptr");
    PIKA_TEST_MSG((pika::is_invocable_v<f, smart_ptr<X const>> == true),
        "mem-obj-ptr/smart-const-ptr");
}

void function_objects()
{
    PIKA_TEST_MSG((pika::is_invocable_v<X, int> == true), "fun-obj/value");
    PIKA_TEST_MSG(
        (pika::is_invocable_v<X const, int> == false), "fun-obj/const-value");
    PIKA_TEST_MSG((pika::is_invocable_v<X*, int> == false), "fun-obj/ptr");
    PIKA_TEST_MSG(
        (pika::is_invocable_v<X const*, int> == false), "fun-obj/const-ptr");
    PIKA_TEST_MSG((pika::is_invocable_v<X&, int> == true), "fun-obj/lvref");
    PIKA_TEST_MSG(
        (pika::is_invocable_v<X const&, int> == false), "fun-obj/const-lvref");
    PIKA_TEST_MSG((pika::is_invocable_v<X&&, int> == true), "fun-obj/rvref");
    PIKA_TEST_MSG(
        (pika::is_invocable_v<X const&&, int> == false), "fun-obj/const-rvref");

    PIKA_TEST_MSG((pika::is_invocable_v<Xc, int> == true), "const-fun-obj/value");
    PIKA_TEST_MSG((pika::is_invocable_v<Xc const, int> == true),
        "const-fun-obj/const-value");
    PIKA_TEST_MSG((pika::is_invocable_v<Xc*, int> == false), "const-fun-obj/ptr");
    PIKA_TEST_MSG((pika::is_invocable_v<Xc const*, int> == false),
        "const-fun-obj/const-ptr");
    PIKA_TEST_MSG(
        (pika::is_invocable_v<Xc&, int> == true), "const-fun-obj/lvref");
    PIKA_TEST_MSG((pika::is_invocable_v<Xc const&, int> == true),
        "const-fun-obj/const-lvref");
    PIKA_TEST_MSG(
        (pika::is_invocable_v<Xc&&, int> == true), "const-fun-obj/rvref");
    PIKA_TEST_MSG((pika::is_invocable_v<Xc const&&, int> == true),
        "const-fun-obj/const-rvref");
}

///////////////////////////////////////////////////////////////////////////////
int main()
{
    {
        nullary_function();
        lambdas();
        functions_byval_params();
        functions_bylvref_params();
        functions_byrvref_params();
        member_function_pointers();
        member_object_pointers();
        function_objects();
    }

    return pika::util::report_errors();
}
