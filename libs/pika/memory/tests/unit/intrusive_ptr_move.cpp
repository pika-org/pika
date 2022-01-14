//
//  intrusive_ptr_move_test.cpp
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

#include <utility>

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

int main()
{
    PIKA_TEST_EQ(N::base::instances, 0);

    {
        pika::intrusive_ptr<X> p(new X);
        PIKA_TEST_EQ(N::base::instances, 1);

        pika::intrusive_ptr<X> p2(std::move(p));
        PIKA_TEST_EQ(N::base::instances, 1);
        // NOLINTNEXTLINE(bugprone-use-after-move)
        PIKA_TEST(p.get() == nullptr);

        p2.reset();
        PIKA_TEST_EQ(N::base::instances, 0);
    }

    {
        pika::intrusive_ptr<Y> p(new Y);
        PIKA_TEST_EQ(N::base::instances, 1);

        pika::intrusive_ptr<X> p2(std::move(p));
        PIKA_TEST_EQ(N::base::instances, 1);
        // NOLINTNEXTLINE(bugprone-use-after-move)
        PIKA_TEST(p.get() == nullptr);

        p2.reset();
        PIKA_TEST_EQ(N::base::instances, 0);
    }

    {
        pika::intrusive_ptr<X> p(new X);
        PIKA_TEST_EQ(N::base::instances, 1);

        pika::intrusive_ptr<X> p2;
        p2 = std::move(p);
        PIKA_TEST_EQ(N::base::instances, 1);
        // NOLINTNEXTLINE(bugprone-use-after-move)
        PIKA_TEST(p.get() == nullptr);

        p2.reset();
        PIKA_TEST_EQ(N::base::instances, 0);
    }

    {
        pika::intrusive_ptr<X> p(new X);
        PIKA_TEST_EQ(N::base::instances, 1);

        pika::intrusive_ptr<X> p2(new X);
        PIKA_TEST_EQ(N::base::instances, 2);
        p2 = std::move(p);
        PIKA_TEST_EQ(N::base::instances, 1);
        // NOLINTNEXTLINE(bugprone-use-after-move)
        PIKA_TEST(p.get() == nullptr);

        p2.reset();
        PIKA_TEST_EQ(N::base::instances, 0);
    }

    {
        pika::intrusive_ptr<Y> p(new Y);
        PIKA_TEST_EQ(N::base::instances, 1);

        pika::intrusive_ptr<X> p2;
        p2 = std::move(p);
        PIKA_TEST_EQ(N::base::instances, 1);
        // NOLINTNEXTLINE(bugprone-use-after-move)
        PIKA_TEST(p.get() == nullptr);

        p2.reset();
        PIKA_TEST_EQ(N::base::instances, 0);
    }

    {
        pika::intrusive_ptr<Y> p(new Y);
        PIKA_TEST_EQ(N::base::instances, 1);

        pika::intrusive_ptr<X> p2(new X);
        PIKA_TEST_EQ(N::base::instances, 2);
        p2 = std::move(p);
        PIKA_TEST_EQ(N::base::instances, 1);
        // NOLINTNEXTLINE(bugprone-use-after-move)
        PIKA_TEST(p.get() == nullptr);

        p2.reset();
        PIKA_TEST_EQ(N::base::instances, 0);
    }

    {
        pika::intrusive_ptr<X> px(new Y);

        X* px2 = px.get();

        pika::intrusive_ptr<Y> py = pika::static_pointer_cast<Y>(std::move(px));
        PIKA_TEST_EQ(py.get(), px2);
        // NOLINTNEXTLINE(bugprone-use-after-move)
        PIKA_TEST(px.get() == nullptr);
        PIKA_TEST_EQ(py->use_count(), 1);
    }

    PIKA_TEST_EQ(N::base::instances, 0);

    {
        pika::intrusive_ptr<X const> px(new X);

        X const* px2 = px.get();

        pika::intrusive_ptr<X> px3 = pika::const_pointer_cast<X>(std::move(px));
        PIKA_TEST_EQ(px3.get(), px2);
        // NOLINTNEXTLINE(bugprone-use-after-move)
        PIKA_TEST(px.get() == nullptr);
        PIKA_TEST_EQ(px3->use_count(), 1);
    }

    PIKA_TEST_EQ(N::base::instances, 0);

    {
        pika::intrusive_ptr<X> px(new Y);

        X* px2 = px.get();

        pika::intrusive_ptr<Y> py = pika::dynamic_pointer_cast<Y>(std::move(px));
        PIKA_TEST_EQ(py.get(), px2);
        // NOLINTNEXTLINE(bugprone-use-after-move)
        PIKA_TEST(px.get() == nullptr);
        PIKA_TEST_EQ(py->use_count(), 1);
    }

    PIKA_TEST_EQ(N::base::instances, 0);

    {
        pika::intrusive_ptr<X> px(new X);

        X* px2 = px.get();

        pika::intrusive_ptr<Y> py = pika::dynamic_pointer_cast<Y>(std::move(px));
        PIKA_TEST(py.get() == nullptr);
        // NOLINTNEXTLINE(bugprone-use-after-move)
        PIKA_TEST_EQ(px.get(), px2);
        PIKA_TEST_EQ(px->use_count(), 1);
    }

    PIKA_TEST_EQ(N::base::instances, 0);

    return pika::util::report_errors();
}
