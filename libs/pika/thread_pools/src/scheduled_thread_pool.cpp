//  Copyright (c) 2017 Shoshana Jakobovits
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

PIKA_GLOBAL_MODULE_FRAGMENT

#include <pika/config.hpp>

#if !defined(PIKA_HAVE_MODULE)
#include <pika/schedulers/local_priority_queue_scheduler.hpp>
#include <pika/schedulers/local_queue_scheduler.hpp>
#include <pika/schedulers/shared_priority_queue_scheduler.hpp>
#include <pika/schedulers/static_priority_queue_scheduler.hpp>
#include <pika/schedulers/static_queue_scheduler.hpp>
#include <pika/thread_pools/scheduled_thread_pool.hpp>
#include <pika/thread_pools/scheduled_thread_pool_impl.hpp>

#include <mutex>
#endif

#if defined(PIKA_HAVE_MODULE)
module pika.thread_pools;
#endif

///////////////////////////////////////////////////////////////////////////////
/// explicit template instantiation for the thread pools of our choice
// template class PIKA_EXPORT pika::threads::detail::local_queue_scheduler<>;
template class PIKA_EXPORT
    pika::threads::detail::scheduled_thread_pool<pika::threads::detail::local_queue_scheduler<>>;

// template class PIKA_EXPORT pika::threads::detail::static_queue_scheduler<>;
template class PIKA_EXPORT
    pika::threads::detail::scheduled_thread_pool<pika::threads::detail::static_queue_scheduler<>>;

// template class PIKA_EXPORT pika::threads::detail::local_priority_queue_scheduler<>;
template class PIKA_EXPORT pika::threads::detail::scheduled_thread_pool<pika::threads::detail::
        local_priority_queue_scheduler<std::mutex, pika::threads::detail::lockfree_fifo>>;

// template class PIKA_EXPORT pika::threads::detail::static_priority_queue_scheduler<>;
template class PIKA_EXPORT pika::threads::detail::scheduled_thread_pool<
    pika::threads::detail::static_priority_queue_scheduler<>>;
#if defined(PIKA_HAVE_CXX11_STD_ATOMIC_128BIT)
// template class PIKA_EXPORT pika::threads::detail::local_priority_queue_scheduler<std::mutex,
//     pika::threads::detail::lockfree_lifo>;
template class PIKA_EXPORT pika::threads::detail::scheduled_thread_pool<pika::threads::detail::
        local_priority_queue_scheduler<std::mutex, pika::threads::detail::lockfree_lifo>>;
#endif

#if defined(PIKA_HAVE_CXX11_STD_ATOMIC_128BIT)
// template class PIKA_EXPORT pika::threads::detail::local_priority_queue_scheduler<std::mutex,
//     pika::threads::detail::lockfree_abp_fifo>;
template class PIKA_EXPORT pika::threads::detail::scheduled_thread_pool<pika::threads::detail::
        local_priority_queue_scheduler<std::mutex, pika::threads::detail::lockfree_abp_fifo>>;
// template class PIKA_EXPORT pika::threads::detail::local_priority_queue_scheduler<std::mutex,
//     pika::threads::detail::lockfree_abp_lifo>;
template class PIKA_EXPORT pika::threads::detail::scheduled_thread_pool<pika::threads::detail::
        local_priority_queue_scheduler<std::mutex, pika::threads::detail::lockfree_abp_lifo>>;
#endif

// template class PIKA_EXPORT pika::threads::detail::shared_priority_queue_scheduler<>;
template class PIKA_EXPORT pika::threads::detail::scheduled_thread_pool<
    pika::threads::detail::shared_priority_queue_scheduler<>>;
