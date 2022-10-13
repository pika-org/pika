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

* :cpp:class:`pika::barrier`
* :cpp:class:`pika::condition_variable`
* :cpp:class:`pika::lcos::local::counting_semaphore`
* :cpp:class:`pika::experimental::event`
* :cpp:class:`pika::latch`
* :cpp:class:`pika::mutex`
* :cpp:class:`pika::no_mutex`
* :cpp:class:`pika::once_flag`
* :cpp:class:`pika::recursive_mutex`
* :cpp:class:`pika::shared_mutex`
* :cpp:class:`pika::sliding_semaphore`
* :cpp:class:`pika::spinlock` (`std::mutex` compatible spinlock)
* :cpp:class:`pika::detail::spinlock_pool`

See :ref:`modules_lcos`, :ref:`modules_async_combinators`, and :ref:`modules_async`
for higher level synchronization facilities.

See the :ref:`API reference <modules_synchronization_api>` of this module for more
details.
