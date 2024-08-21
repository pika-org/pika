//  Copyright (c) 2019 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

PIKA_GLOBAL_MODULE_FRAGMENT

#include <pika/config.hpp>
#if !defined(PIKA_HAVE_MODULE)
#include <pika/threading_base/print.hpp>
#include <pika/threading_base/scheduler_base.hpp>
#include <pika/threading_base/thread_data.hpp>

#include <cstdint>
#include <thread>

#if defined(__linux) || defined(linux) || defined(__linux__)
# include <linux/unistd.h>
# include <sys/mman.h>
# define PIKA_DEBUGGING_PRINT_LINUX
#endif
#endif

#if defined(PIKA_HAVE_MODULE)
module pika.threading_base;
#endif

// ------------------------------------------------------------
/// \cond NODETAIL
namespace pika::debug::detail {

    std::ostream& operator<<(std::ostream& os, threadinfo<threads::detail::thread_data*> const& d)
    {
        os << ptr(d.data) << " \"" << ((d.data != nullptr) ? d.data->get_description() : "nullptr")
           << "\"";
        return os;
    }

    std::ostream& operator<<(
        std::ostream& os, threadinfo<threads::detail::thread_id_type*> const& d)
    {
        if (d.data == nullptr) { os << "nullptr"; }
        else { os << threadinfo<threads::detail::thread_data*>(get_thread_id_data(*d.data)); }
        return os;
    }

    std::ostream& operator<<(
        std::ostream& os, threadinfo<threads::detail::thread_id_ref_type*> const& d)
    {
        if (d.data == nullptr) { os << "nullptr"; }
        else { os << threadinfo<threads::detail::thread_data*>(get_thread_id_data(*d.data)); }
        return os;
    }

    std::ostream& operator<<(
        std::ostream& os, threadinfo<pika::threads::detail::thread_init_data> const& d)
    {
#if defined(PIKA_HAVE_THREAD_DESCRIPTION)
        os << std::left << " \"" << d.data.description.get_description() << "\"";
#else
        os << "??? " << /*hex<8,uintptr_t>*/ (std::uintptr_t(&d.data));
#endif
        return os;
    }

    // ------------------------------------------------------------------
    // helper class for printing thread ID, either std:: or pika::
    // ------------------------------------------------------------------
    namespace detail {

        void print_thread_info(std::ostream& os)
        {
            if (pika::threads::detail::get_self_id() == pika::threads::detail::invalid_thread_id)
            {
                os << "-------------- ";
            }
            else
            {
                pika::threads::detail::thread_data* dummy =
                    pika::threads::detail::get_self_id_data();
                os << hex<12, std::uintptr_t>(reinterpret_cast<std::uintptr_t>(dummy)) << " ";
            }
            const char* pool = "--------";
            auto tid = pika::threads::detail::get_self_id();
            if (tid != threads::detail::invalid_thread_id)
            {
                auto* p = get_thread_id_data(tid)->get_scheduler_base()->get_parent_pool();
                pool = p->get_pool_name().c_str();
            }
            os << hex<12, std::thread::id>(std::this_thread::get_id()) << " "
               << debug::detail::str<8>(pool)

#ifdef PIKA_DEBUGGING_PRINT_LINUX
               << " cpu " << debug::detail::dec<3, int>(sched_getcpu()) << " ";
#else
               << " cpu "
               << "--- ";
#endif
        }

        struct current_thread_print_helper
        {
            current_thread_print_helper()
            {
                debug::detail::register_print_info(&detail::print_thread_info);
            }

            static current_thread_print_helper helper_;
        };

        current_thread_print_helper current_thread_print_helper::helper_{};
    }    // namespace detail
}    // namespace pika::debug::detail
/// \endcond
