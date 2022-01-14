..
    Copyright (c) 2019 The STE||AR-Group

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _modules_synchronization:

===============
synchronization
===============

This module provides synchronization primitives which should be used rather than
the C++ standard ones in |pika| threads:

* :cpp:class:`pika::lcos::local::barrier`
* :cpp:class:`pika::lcos::local::condition_variable`
* :cpp:class:`pika::lcos::local::counting_semaphore`
* :cpp:class:`pika::lcos::local::event`
* :cpp:class:`pika::lcos::local::latch`
* :cpp:class:`pika::lcos::local::mutex`
* :cpp:class:`pika::lcos::local::no_mutex`
* :cpp:class:`pika::lcos::local::once_flag`
* :cpp:class:`pika::lcos::local::recursive_mutex`
* :cpp:class:`pika::lcos::local::shared_mutex`
* :cpp:class:`pika::lcos::local::sliding_semaphore`
* :cpp:class:`pika::lcos::local::spinlock` (`std::mutex` compatible spinlock)
* :cpp:class:`pika::lcos::local::spinlock_no_backoff` (`boost::mutex` compatible spinlock)
* :cpp:class:`pika::lcos::local::spinlock_pool`

See :ref:`modules_lcos_local`, :ref:`modules_async_combinators`, and :ref:`modules_async`
for higher level synchronization facilities.

See the :ref:`API reference <modules_synchronization_api>` of this module for more
details.

