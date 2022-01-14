..
    Copyright (c) 2020 The STE||AR-Group

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _modules_executors:

=========
executors
=========

The executors module exposes executors and execution policies. Most importantly,
it exposes the following classes and constants:

* :cpp:class:`pika::execution::sequenced_executor`
* :cpp:class:`pika::execution::parallel_executor`
* :cpp:class:`pika::execution::sequenced_policy`
* :cpp:class:`pika::execution::parallel_policy`
* :cpp:class:`pika::execution::parallel_unsequenced_policy`
* :cpp:class:`pika::execution::sequenced_task_policy`
* :cpp:class:`pika::execution::parallel_task_policy`
* :cpp:var:`pika::execution::seq`
* :cpp:var:`pika::execution::par`
* :cpp:var:`pika::execution::par_unseq`
* :cpp:var:`pika::execution::task`

See the :ref:`API reference <modules_executors_api>` of this module for more
details.

