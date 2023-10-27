..
    Copyright (c) 2023 ETH Zurich

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _api:

=============
API reference
=============

pika follows `semver <https://semver.org>`_. pika is currently at a 0.X version which means that
minor versions may break the API. pika gives no guarantees about ABI stability. The ABI may change
even in patch versions.

The API reference is a work in progress. The following headers are part of the public API. Any other
headers are internal implementation details.

- ``pika/async_rw_mutex.hpp``
- ``pika/barrier.hpp``
- ``pika/condition_variable.hpp``
- ``pika/cuda.hpp``
- ``pika/execution.hpp``
- ``pika/latch.hpp``
- ``pika/mpi.hpp``
- ``pika/mutex.hpp``
- ``pika/runtime.hpp``
- ``pika/semaphore.hpp``
- ``pika/shared_mutex.hpp``
- ``pika/thread.hpp``

All functionality in a namespace containing ``detail`` and all macros prefixed
with ``PIKA_DETAIL`` are implementation details and may change without warning
at any time.
