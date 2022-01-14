//
//  intrusive_ptr_test.cpp
//
//  Copyright (c) 2002-2005 Peter Dimov
//
// SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//

#include <pika/local/config.hpp>

#if defined(PIKA_MSVC)

#pragma warning(disable : 4786)    // identifier truncated in debug info
#pragma warning(disable : 4710)    // function not inlined
#pragma warning(                                                               \
    disable : 4711)    // function selected for automatic inline expansion
#pragma warning(disable : 4514)    // unreferenced inline removed
#pragma warning(                                                               \
    disable : 4355)    // 'this' : used in base member initializer list
#pragma warning(disable : 4511)    // copy constructor could not be generated
#pragma warning(disable : 4512)    // assignment operator could not be generated
#pragma warning(disable : 4675)    // resolved overload found with Koenig lookup

#endif

#include <pika/modules/memory.hpp>
#include <pika/modules/testing.hpp>
#include <pika/thread_support/atomic_count.hpp>

#include <algorithm>
#include <functional>
#include <utility>

//
namespace N {

    class base
    {
    private:
        mutable pika::util::atomic_count use_count_;

        base(base const&) = delete;
        base& operator=(base const&) = delete;

    protected:
        base()
          : use_count_(0)
        {
            ++instances;
        }

        virtual ~base()
        {
            --instances;
        }

    public:
        static long instances;

        long use_count() const
        {
            return use_count_;
        }

        inline friend void intrusive_ptr_add_ref(base const* p)
        {
            ++p->use_count_;
        }

        inline friend void intrusive_ptr_release(base const* p)
        {
            if (--p->use_count_ == 0)
                delete p;
        }
    };

    long base::instances = 0;

}    // namespace N

//
struct X : public virtual N::base
{
};

struct Y : public X
{
};

//
namespace n_element_type {

    void f(X&) {}

    void test()
    {
        using T = pika::intrusive_ptr<X>::element_type;

        T t;
        f(t);
    }

}    // namespace n_element_type

namespace n_constructors {

    void default_constructor()
    {
        pika::intrusive_ptr<X> px;
        PIKA_TEST(px.get() == nullptr);
    }

    void pointer_constructor()
    {
        {
            pika::intrusive_ptr<X> px(nullptr);
            PIKA_TEST(px.get() == nullptr);
        }

        {
            pika::intrusive_ptr<X> px(nullptr, false);
            PIKA_TEST(px.get() == nullptr);
        }

        PIKA_TEST_EQ(N::base::instances, 0);

        {
            X* p = new X;
            PIKA_TEST_EQ(p->use_count(), 0);

            PIKA_TEST_EQ(N::base::instances, 1);

            pika::intrusive_ptr<X> px(p);
            PIKA_TEST_EQ(px.get(), p);
            PIKA_TEST_EQ(px->use_count(), 1);
        }

        PIKA_TEST_EQ(N::base::instances, 0);

        {
            X* p = new X;
            PIKA_TEST_EQ(p->use_count(), 0);

            PIKA_TEST_EQ(N::base::instances, 1);

            intrusive_ptr_add_ref(p);
            PIKA_TEST_EQ(p->use_count(), 1);

            pika::intrusive_ptr<X> px(p, false);
            PIKA_TEST_EQ(px.get(), p);
            PIKA_TEST_EQ(px->use_count(), 1);
        }

        PIKA_TEST_EQ(N::base::instances, 0);
    }

    void copy_constructor()
    {
        {
            pika::intrusive_ptr<X> px;
            pika::intrusive_ptr<X> px2(px);
            PIKA_TEST_EQ(px2.get(), px.get());
        }

        {
            pika::intrusive_ptr<Y> py;
            pika::intrusive_ptr<X> px(py);
            PIKA_TEST_EQ(px.get(), py.get());
        }

        {
            pika::intrusive_ptr<X> px(nullptr);
            pika::intrusive_ptr<X> px2(px);
            PIKA_TEST_EQ(px2.get(), px.get());
        }

        {
            pika::intrusive_ptr<Y> py(nullptr);
            pika::intrusive_ptr<X> px(py);
            PIKA_TEST_EQ(px.get(), py.get());
        }

        {
            pika::intrusive_ptr<X> px(nullptr, false);
            pika::intrusive_ptr<X> px2(px);
            PIKA_TEST_EQ(px2.get(), px.get());
        }

        {
            pika::intrusive_ptr<Y> py(nullptr, false);
            pika::intrusive_ptr<X> px(py);
            PIKA_TEST_EQ(px.get(), py.get());
        }

        PIKA_TEST_EQ(N::base::instances, 0);

        {
            pika::intrusive_ptr<X> px(new X);
            pika::intrusive_ptr<X> px2(px);
            PIKA_TEST_EQ(px2.get(), px.get());

            PIKA_TEST_EQ(N::base::instances, 1);
        }

        PIKA_TEST_EQ(N::base::instances, 0);

        {
            pika::intrusive_ptr<Y> py(new Y);
            pika::intrusive_ptr<X> px(py);
            PIKA_TEST_EQ(px.get(), py.get());

            PIKA_TEST_EQ(N::base::instances, 1);
        }

        PIKA_TEST_EQ(N::base::instances, 0);
    }

    void test()
    {
        default_constructor();
        pointer_constructor();
        copy_constructor();
    }

}    // namespace n_constructors

namespace n_destructor {

    void test()
    {
        PIKA_TEST_EQ(N::base::instances, 0);

        {
            pika::intrusive_ptr<X> px(new X);
            PIKA_TEST_EQ(px->use_count(), 1);

            PIKA_TEST_EQ(N::base::instances, 1);

            {
                pika::intrusive_ptr<X> px2(px);
                PIKA_TEST_EQ(px->use_count(), 2);
            }

            PIKA_TEST_EQ(px->use_count(), 1);
        }

        PIKA_TEST_EQ(N::base::instances, 0);
    }

}    // namespace n_destructor

namespace n_assignment {

    void copy_assignment()
    {
        PIKA_TEST_EQ(N::base::instances, 0);

        {
            pika::intrusive_ptr<X> p1;

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wself-assign-overloaded"
#endif
            p1 = p1;
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

            PIKA_TEST_EQ(p1, p1);
            PIKA_TEST(p1 ? false : true);
            PIKA_TEST(!p1);
            PIKA_TEST(p1.get() == nullptr);

            pika::intrusive_ptr<X> p2;

            p1 = p2;

            PIKA_TEST_EQ(p1, p2);
            PIKA_TEST(p1 ? false : true);
            PIKA_TEST(!p1);
            PIKA_TEST(p1.get() == nullptr);

            pika::intrusive_ptr<X> p3(p1);

            p1 = p3;

            PIKA_TEST_EQ(p1, p3);
            PIKA_TEST(p1 ? false : true);
            PIKA_TEST(!p1);
            PIKA_TEST(p1.get() == nullptr);

            PIKA_TEST_EQ(N::base::instances, 0);

            pika::intrusive_ptr<X> p4(new X);

            PIKA_TEST_EQ(N::base::instances, 1);

            p1 = p4;

            PIKA_TEST_EQ(N::base::instances, 1);

            PIKA_TEST_EQ(p1, p4);

            PIKA_TEST_EQ(p1->use_count(), 2);

            p1 = p2;

            PIKA_TEST_EQ(p1, p2);
            PIKA_TEST_EQ(N::base::instances, 1);

            p4 = p3;

            PIKA_TEST_EQ(p4, p3);
            PIKA_TEST_EQ(N::base::instances, 0);
        }
    }

    void conversion_assignment()
    {
        PIKA_TEST_EQ(N::base::instances, 0);

        {
            pika::intrusive_ptr<X> p1;

            pika::intrusive_ptr<Y> p2;

            p1 = p2;

            PIKA_TEST_EQ(p1, p2);
            PIKA_TEST(p1 ? false : true);
            PIKA_TEST(!p1);
            PIKA_TEST(p1.get() == nullptr);

            PIKA_TEST_EQ(N::base::instances, 0);

            pika::intrusive_ptr<Y> p4(new Y);

            PIKA_TEST_EQ(N::base::instances, 1);
            PIKA_TEST_EQ(p4->use_count(), 1);

            pika::intrusive_ptr<X> p5(p4);
            PIKA_TEST_EQ(p4->use_count(), 2);

            p1 = p4;

            PIKA_TEST_EQ(N::base::instances, 1);

            PIKA_TEST_EQ(p1, p4);

            PIKA_TEST_EQ(p1->use_count(), 3);
            PIKA_TEST_EQ(p4->use_count(), 3);

            p1 = p2;

            PIKA_TEST_EQ(p1, p2);
            PIKA_TEST_EQ(N::base::instances, 1);
            PIKA_TEST_EQ(p4->use_count(), 2);

            p4 = p2;
            p5 = p2;

            PIKA_TEST_EQ(p4, p2);
            PIKA_TEST_EQ(N::base::instances, 0);
        }
    }

    void pointer_assignment()
    {
        PIKA_TEST_EQ(N::base::instances, 0);

        {
            pika::intrusive_ptr<X> p1;

            p1 = p1.get();

            PIKA_TEST_EQ(p1, p1);
            PIKA_TEST(p1 ? false : true);
            PIKA_TEST(!p1);
            PIKA_TEST(p1.get() == nullptr);

            pika::intrusive_ptr<X> p2;

            p1 = p2.get();

            PIKA_TEST_EQ(p1, p2);
            PIKA_TEST(p1 ? false : true);
            PIKA_TEST(!p1);
            PIKA_TEST(p1.get() == nullptr);

            pika::intrusive_ptr<X> p3(p1);

            p1 = p3.get();

            PIKA_TEST_EQ(p1, p3);
            PIKA_TEST(p1 ? false : true);
            PIKA_TEST(!p1);
            PIKA_TEST(p1.get() == nullptr);

            PIKA_TEST_EQ(N::base::instances, 0);

            pika::intrusive_ptr<X> p4(new X);

            PIKA_TEST_EQ(N::base::instances, 1);

            p1 = p4.get();

            PIKA_TEST_EQ(N::base::instances, 1);

            PIKA_TEST_EQ(p1, p4);

            PIKA_TEST_EQ(p1->use_count(), 2);

            p1 = p2.get();

            PIKA_TEST_EQ(p1, p2);
            PIKA_TEST_EQ(N::base::instances, 1);

            p4 = p3.get();

            PIKA_TEST_EQ(p4, p3);
            PIKA_TEST_EQ(N::base::instances, 0);
        }

        {
            pika::intrusive_ptr<X> p1;

            pika::intrusive_ptr<Y> p2;

            p1 = p2.get();

            PIKA_TEST_EQ(p1, p2);
            PIKA_TEST(p1 ? false : true);
            PIKA_TEST(!p1);
            PIKA_TEST(p1.get() == nullptr);

            PIKA_TEST_EQ(N::base::instances, 0);

            pika::intrusive_ptr<Y> p4(new Y);

            PIKA_TEST_EQ(N::base::instances, 1);
            PIKA_TEST_EQ(p4->use_count(), 1);

            pika::intrusive_ptr<X> p5(p4);
            PIKA_TEST_EQ(p4->use_count(), 2);

            p1 = p4.get();

            PIKA_TEST_EQ(N::base::instances, 1);

            PIKA_TEST_EQ(p1, p4);

            PIKA_TEST_EQ(p1->use_count(), 3);
            PIKA_TEST_EQ(p4->use_count(), 3);

            p1 = p2.get();

            PIKA_TEST_EQ(p1, p2);
            PIKA_TEST_EQ(N::base::instances, 1);
            PIKA_TEST_EQ(p4->use_count(), 2);

            p4 = p2.get();
            p5 = p2.get();

            PIKA_TEST_EQ(p4, p2);
            PIKA_TEST_EQ(N::base::instances, 0);
        }
    }

    void test()
    {
        copy_assignment();
        conversion_assignment();
        pointer_assignment();
    }

}    // namespace n_assignment

namespace n_reset {

    void test()
    {
        PIKA_TEST_EQ(N::base::instances, 0);

        {
            pika::intrusive_ptr<X> px;
            PIKA_TEST(px.get() == nullptr);

            px.reset();
            PIKA_TEST(px.get() == nullptr);

            X* p = new X;
            PIKA_TEST_EQ(p->use_count(), 0);
            PIKA_TEST_EQ(N::base::instances, 1);

            px.reset(p);
            PIKA_TEST_EQ(px.get(), p);
            PIKA_TEST_EQ(px->use_count(), 1);

            px.reset();
            PIKA_TEST(px.get() == nullptr);
        }

        PIKA_TEST_EQ(N::base::instances, 0);

        {
            pika::intrusive_ptr<X> px(new X);
            PIKA_TEST_EQ(N::base::instances, 1);

            px.reset(nullptr);
            PIKA_TEST(px.get() == nullptr);
        }

        PIKA_TEST_EQ(N::base::instances, 0);

        {
            pika::intrusive_ptr<X> px(new X);
            PIKA_TEST_EQ(N::base::instances, 1);

            px.reset(nullptr, false);
            PIKA_TEST(px.get() == nullptr);
        }

        PIKA_TEST_EQ(N::base::instances, 0);

        {
            pika::intrusive_ptr<X> px(new X);
            PIKA_TEST_EQ(N::base::instances, 1);

            px.reset(nullptr, true);
            PIKA_TEST(px.get() == nullptr);
        }

        PIKA_TEST_EQ(N::base::instances, 0);

        {
            X* p = new X;
            PIKA_TEST_EQ(p->use_count(), 0);

            PIKA_TEST_EQ(N::base::instances, 1);

            pika::intrusive_ptr<X> px;
            PIKA_TEST(px.get() == nullptr);

            px.reset(p, true);
            PIKA_TEST_EQ(px.get(), p);
            PIKA_TEST_EQ(px->use_count(), 1);
        }

        PIKA_TEST_EQ(N::base::instances, 0);

        {
            X* p = new X;
            PIKA_TEST_EQ(p->use_count(), 0);

            PIKA_TEST_EQ(N::base::instances, 1);

#if defined(BOOST_NO_ARGUMENT_DEPENDENT_LOOKUP)
            using pika::intrusive_ptr_add_ref;
#endif
            intrusive_ptr_add_ref(p);
            PIKA_TEST_EQ(p->use_count(), 1);

            pika::intrusive_ptr<X> px;
            PIKA_TEST(px.get() == nullptr);

            px.reset(p, false);
            PIKA_TEST_EQ(px.get(), p);
            PIKA_TEST_EQ(px->use_count(), 1);
        }

        PIKA_TEST_EQ(N::base::instances, 0);

        {
            pika::intrusive_ptr<X> px(new X);
            PIKA_TEST(px.get() != nullptr);
            PIKA_TEST_EQ(px->use_count(), 1);

            PIKA_TEST_EQ(N::base::instances, 1);

            X* p = new X;
            PIKA_TEST_EQ(p->use_count(), 0);

            PIKA_TEST_EQ(N::base::instances, 2);

            px.reset(p);
            PIKA_TEST_EQ(px.get(), p);
            PIKA_TEST_EQ(px->use_count(), 1);

            PIKA_TEST_EQ(N::base::instances, 1);
        }

        PIKA_TEST_EQ(N::base::instances, 0);

        {
            pika::intrusive_ptr<X> px(new X);
            PIKA_TEST(px.get() != nullptr);
            PIKA_TEST_EQ(px->use_count(), 1);

            PIKA_TEST_EQ(N::base::instances, 1);

            X* p = new X;
            PIKA_TEST_EQ(p->use_count(), 0);

            PIKA_TEST_EQ(N::base::instances, 2);

            px.reset(p, true);
            PIKA_TEST_EQ(px.get(), p);
            PIKA_TEST_EQ(px->use_count(), 1);

            PIKA_TEST_EQ(N::base::instances, 1);
        }

        PIKA_TEST_EQ(N::base::instances, 0);

        {
            pika::intrusive_ptr<X> px(new X);
            PIKA_TEST(px.get() != nullptr);
            PIKA_TEST_EQ(px->use_count(), 1);

            PIKA_TEST_EQ(N::base::instances, 1);

            X* p = new X;
            PIKA_TEST_EQ(p->use_count(), 0);

#if defined(BOOST_NO_ARGUMENT_DEPENDENT_LOOKUP)
            using pika::intrusive_ptr_add_ref;
#endif
            intrusive_ptr_add_ref(p);
            PIKA_TEST_EQ(p->use_count(), 1);

            PIKA_TEST_EQ(N::base::instances, 2);

            px.reset(p, false);
            PIKA_TEST_EQ(px.get(), p);
            PIKA_TEST_EQ(px->use_count(), 1);

            PIKA_TEST_EQ(N::base::instances, 1);
        }

        PIKA_TEST_EQ(N::base::instances, 0);
    }

}    // namespace n_reset

namespace n_access {

    void test()
    {
        {
            pika::intrusive_ptr<X> px;
            PIKA_TEST(px ? false : true);
            PIKA_TEST(!px);

#if defined(BOOST_NO_ARGUMENT_DEPENDENT_LOOKUP)
            using boost::get_pointer;
#endif

            PIKA_TEST_EQ(get_pointer(px), px.get());
        }

        {
            pika::intrusive_ptr<X> px(nullptr);
            PIKA_TEST(px ? false : true);
            PIKA_TEST(!px);

#if defined(BOOST_NO_ARGUMENT_DEPENDENT_LOOKUP)
            using boost::get_pointer;
#endif

            PIKA_TEST_EQ(get_pointer(px), px.get());
        }

        {
            pika::intrusive_ptr<X> px(new X);
            PIKA_TEST(px ? true : false);
            PIKA_TEST(!!px);
            PIKA_TEST_EQ(&*px, px.get());
            PIKA_TEST_EQ(px.operator->(), px.get());

#if defined(BOOST_NO_ARGUMENT_DEPENDENT_LOOKUP)
            using boost::get_pointer;
#endif

            PIKA_TEST_EQ(get_pointer(px), px.get());
        }

        {
            pika::intrusive_ptr<X> px;
            X* detached = px.detach();
            PIKA_TEST(px.get() == nullptr);
            PIKA_TEST(detached == nullptr);
        }

        {
            X* p = new X;
            PIKA_TEST_EQ(p->use_count(), 0);

            pika::intrusive_ptr<X> px(p);
            PIKA_TEST_EQ(px.get(), p);
            PIKA_TEST_EQ(px->use_count(), 1);

            X* detached = px.detach();
            PIKA_TEST(px.get() == nullptr);

            PIKA_TEST_EQ(detached, p);
            PIKA_TEST_EQ(detached->use_count(), 1);

            delete detached;
        }
    }

}    // namespace n_access

namespace n_swap {

    void test()
    {
        {
            pika::intrusive_ptr<X> px;
            pika::intrusive_ptr<X> px2;

            px.swap(px2);

            PIKA_TEST(px.get() == nullptr);
            PIKA_TEST(px2.get() == nullptr);

            using std::swap;
            swap(px, px2);

            PIKA_TEST(px.get() == nullptr);
            PIKA_TEST(px2.get() == nullptr);
        }

        {
            X* p = new X;
            pika::intrusive_ptr<X> px;
            pika::intrusive_ptr<X> px2(p);
            pika::intrusive_ptr<X> px3(px2);

            px.swap(px2);

            PIKA_TEST_EQ(px.get(), p);
            PIKA_TEST_EQ(px->use_count(), 2);
            PIKA_TEST(px2.get() == nullptr);
            PIKA_TEST_EQ(px3.get(), p);
            PIKA_TEST_EQ(px3->use_count(), 2);

            using std::swap;
            swap(px, px2);

            PIKA_TEST(px.get() == nullptr);
            PIKA_TEST_EQ(px2.get(), p);
            PIKA_TEST_EQ(px2->use_count(), 2);
            PIKA_TEST_EQ(px3.get(), p);
            PIKA_TEST_EQ(px3->use_count(), 2);
        }

        {
            X* p1 = new X;
            X* p2 = new X;
            pika::intrusive_ptr<X> px(p1);
            pika::intrusive_ptr<X> px2(p2);
            pika::intrusive_ptr<X> px3(px2);

            px.swap(px2);

            PIKA_TEST_EQ(px.get(), p2);
            PIKA_TEST_EQ(px->use_count(), 2);
            PIKA_TEST_EQ(px2.get(), p1);
            PIKA_TEST_EQ(px2->use_count(), 1);
            PIKA_TEST_EQ(px3.get(), p2);
            PIKA_TEST_EQ(px3->use_count(), 2);

            using std::swap;
            swap(px, px2);

            PIKA_TEST_EQ(px.get(), p1);
            PIKA_TEST_EQ(px->use_count(), 1);
            PIKA_TEST_EQ(px2.get(), p2);
            PIKA_TEST_EQ(px2->use_count(), 2);
            PIKA_TEST_EQ(px3.get(), p2);
            PIKA_TEST_EQ(px3->use_count(), 2);
        }
    }

}    // namespace n_swap

namespace n_comparison {

    template <class T, class U>
    void test2(pika::intrusive_ptr<T> const& p, pika::intrusive_ptr<U> const& q)
    {
        PIKA_TEST((p == q) == (p.get() == q.get()));
        PIKA_TEST((p != q) == (p.get() != q.get()));
    }

    template <class T>
    void test3(pika::intrusive_ptr<T> const& p, pika::intrusive_ptr<T> const& q)
    {
        PIKA_TEST((p == q) == (p.get() == q.get()));
        PIKA_TEST((p.get() == q) == (p.get() == q.get()));
        PIKA_TEST((p == q.get()) == (p.get() == q.get()));
        PIKA_TEST((p != q) == (p.get() != q.get()));
        PIKA_TEST((p.get() != q) == (p.get() != q.get()));
        PIKA_TEST((p != q.get()) == (p.get() != q.get()));

        // 'less' moved here as a g++ 2.9x parse error workaround
        std::less<T*> less;
        PIKA_TEST((p < q) == less(p.get(), q.get()));
    }

    void test()
    {
        {
            pika::intrusive_ptr<X> px;
            test3(px, px);

            pika::intrusive_ptr<X> px2;
            test3(px, px2);

            pika::intrusive_ptr<X> px3(px);
            test3(px3, px3);
            test3(px, px3);
        }

        {
            pika::intrusive_ptr<X> px;

            pika::intrusive_ptr<X> px2(new X);
            test3(px, px2);
            test3(px2, px2);

            pika::intrusive_ptr<X> px3(new X);
            test3(px2, px3);

            pika::intrusive_ptr<X> px4(px2);
            test3(px2, px4);
            test3(px4, px4);
        }

        {
            pika::intrusive_ptr<X> px(new X);

            pika::intrusive_ptr<Y> py(new Y);
            test2(px, py);

            pika::intrusive_ptr<X> px2(py);
            test2(px2, py);
            test3(px, px2);
            test3(px2, px2);
        }
    }

}    // namespace n_comparison

namespace n_static_cast {

    void test()
    {
        {
            pika::intrusive_ptr<X> px(new Y);

            pika::intrusive_ptr<Y> py = pika::static_pointer_cast<Y>(px);
            PIKA_TEST_EQ(px.get(), py.get());
            PIKA_TEST_EQ(px->use_count(), 2);
            PIKA_TEST_EQ(py->use_count(), 2);

            pika::intrusive_ptr<X> px2(py);
            PIKA_TEST_EQ(px2.get(), px.get());
        }

        PIKA_TEST_EQ(N::base::instances, 0);

        {
            pika::intrusive_ptr<Y> py =
                pika::static_pointer_cast<Y>(pika::intrusive_ptr<X>(new Y));
            PIKA_TEST(py.get() != nullptr);
            PIKA_TEST_EQ(py->use_count(), 1);
        }

        PIKA_TEST_EQ(N::base::instances, 0);
    }

}    // namespace n_static_cast

namespace n_const_cast {

    void test()
    {
        {
            pika::intrusive_ptr<X const> px;

            pika::intrusive_ptr<X> px2 = pika::const_pointer_cast<X>(px);
            PIKA_TEST(px2.get() == nullptr);
        }

        {
            pika::intrusive_ptr<X> px2 =
                pika::const_pointer_cast<X>(pika::intrusive_ptr<X const>());
            PIKA_TEST(px2.get() == nullptr);
        }

        PIKA_TEST_EQ(N::base::instances, 0);

        {
            pika::intrusive_ptr<X const> px(new X);

            pika::intrusive_ptr<X> px2 = pika::const_pointer_cast<X>(px);
            PIKA_TEST_EQ(px2.get(), px.get());
            PIKA_TEST_EQ(px2->use_count(), 2);
            PIKA_TEST_EQ(px->use_count(), 2);
        }

        PIKA_TEST_EQ(N::base::instances, 0);

        {
            pika::intrusive_ptr<X> px =
                pika::const_pointer_cast<X>(pika::intrusive_ptr<X const>(new X));
            PIKA_TEST(px.get() != nullptr);
            PIKA_TEST_EQ(px->use_count(), 1);
        }

        PIKA_TEST_EQ(N::base::instances, 0);
    }

}    // namespace n_const_cast

namespace n_dynamic_cast {

    void test()
    {
        {
            pika::intrusive_ptr<X> px;

            pika::intrusive_ptr<Y> py = pika::dynamic_pointer_cast<Y>(px);
            PIKA_TEST(py.get() == nullptr);
        }

        {
            pika::intrusive_ptr<Y> py =
                pika::dynamic_pointer_cast<Y>(pika::intrusive_ptr<X>());
            PIKA_TEST(py.get() == nullptr);
        }

        {
            pika::intrusive_ptr<X> px(static_cast<X*>(nullptr));

            pika::intrusive_ptr<Y> py = pika::dynamic_pointer_cast<Y>(px);
            PIKA_TEST(py.get() == nullptr);
        }

        {
            pika::intrusive_ptr<Y> py = pika::dynamic_pointer_cast<Y>(
                pika::intrusive_ptr<X>(static_cast<X*>(nullptr)));
            PIKA_TEST(py.get() == nullptr);
        }

        {
            pika::intrusive_ptr<X> px(new X);

            pika::intrusive_ptr<Y> py = pika::dynamic_pointer_cast<Y>(px);
            PIKA_TEST(py.get() == nullptr);
        }

        PIKA_TEST_EQ(N::base::instances, 0);

        {
            pika::intrusive_ptr<Y> py =
                pika::dynamic_pointer_cast<Y>(pika::intrusive_ptr<X>(new X));
            PIKA_TEST(py.get() == nullptr);
        }

        PIKA_TEST_EQ(N::base::instances, 0);

        {
            pika::intrusive_ptr<X> px(new Y);

            pika::intrusive_ptr<Y> py = pika::dynamic_pointer_cast<Y>(px);
            PIKA_TEST_EQ(py.get(), px.get());
            PIKA_TEST_EQ(py->use_count(), 2);
            PIKA_TEST_EQ(px->use_count(), 2);
        }

        PIKA_TEST_EQ(N::base::instances, 0);

        {
            pika::intrusive_ptr<X> px(new Y);

            pika::intrusive_ptr<Y> py =
                pika::dynamic_pointer_cast<Y>(pika::intrusive_ptr<X>(new Y));
            PIKA_TEST(py.get() != nullptr);
            PIKA_TEST_EQ(py->use_count(), 1);
        }

        PIKA_TEST_EQ(N::base::instances, 0);
    }

}    // namespace n_dynamic_cast

namespace n_transitive {

    struct X : public N::base
    {
        pika::intrusive_ptr<X> next;
    };

    void test()
    {
        pika::intrusive_ptr<X> p(new X);
        p->next = pika::intrusive_ptr<X>(new X);
        PIKA_TEST(!p->next->next);
        p = p->next;
        PIKA_TEST(!p->next);
    }

}    // namespace n_transitive

namespace n_report_1 {

    class foo : public N::base
    {
    public:
        foo()
          : m_self(this)
        {
        }

        void suicide()
        {
            m_self = nullptr;
        }

    private:
        pika::intrusive_ptr<foo> m_self;
    };

    void test()
    {
        foo* foo_ptr = new foo;
        foo_ptr->suicide();
    }

}    // namespace n_report_1

int main()
{
    n_element_type::test();
    n_constructors::test();
    n_destructor::test();
    n_assignment::test();
    n_reset::test();
    n_access::test();
    n_swap::test();
    n_comparison::test();
    n_static_cast::test();
    n_const_cast::test();
    n_dynamic_cast::test();

    n_transitive::test();
    n_report_1::test();

    return pika::util::report_errors();
}
