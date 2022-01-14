..
    Copyright (c) 2019 The STE||AR-Group

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _modules_lcos_local:

==========
lcos_local
==========

This module provides the following local :term:`LCO`\ s:

* :cpp:class:`pika::lcos::local::and_gate`
* :cpp:class:`pika::lcos::local::channel`
* :cpp:class:`pika::lcos::local::one_element_channel`
* :cpp:class:`pika::lcos::local::receive_channel`
* :cpp:class:`pika::lcos::local::send_channel`
* :cpp:class:`pika::lcos::local::guard`
* :cpp:class:`pika::lcos::local::guard_set`
* :cpp:func:`pika::lcos::local::run_guarded`
* :cpp:class:`pika::lcos::local::conditional_trigger`
* :cpp:class:`pika::lcos::local::packaged_task`
* :cpp:class:`pika::lcos::local::promise`
* :cpp:class:`pika::lcos::local::receive_buffer`
* :cpp:class:`pika::lcos::local::trigger`

See :ref:`modules_lcos_distributed` for distributed LCOs. Basic synchronization
primitives for use in |pika| threads can be found in :ref:`modules_synchronization`.
:ref:`async_combinators` contains useful utility functions for combining
futures.

See the :ref:`API reference <modules_lcos_local_api>` of this module for more
details.

