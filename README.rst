..
    Copyright (c) 2022-2023 ETH Zurich

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

|bors_enabled|
|circleci_status|
|github_actions_linux_debug_status|
|github_actions_linux_hip_status|
|github_actions_linux_sanitizers_status|
|github_actions_macos_debug_status|
|codacy|
|codacy_coverage|

====
pika
====

pika is a C++ library for concurrency and parallelism. It implements
senders/receivers (as proposed in `P2300 <https://wg21.link/p2300>`_) for CPU
thread pools, MPI, and CUDA.

To get started using pika see the `documentation <https://pikacpp.org>`_.

Documentation
=============

Documentation is a work in progress. The following headers are part of the
public API. Any other headers are internal implementation details.

- ``pika/async_rw_mutex.hpp``
- ``pika/barrier.hpp``
- ``pika/channel.hpp``
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

.. |bors_enabled| image:: https://bors.tech/images/badge_small.svg
     :target: https://app.bors.tech/repositories/41470
     :alt: Bors enabled

.. |circleci_status| image:: https://circleci.com/gh/pika-org/pika/tree/main.svg?style=svg
     :target: https://circleci.com/gh/pika-org/pika/tree/main
     :alt: CircleCI

.. |github_actions_linux_debug_status| image:: https://github.com/pika-org/pika/actions/workflows/linux_debug.yml/badge.svg
     :target: https://github.com/pika-org/pika/actions/workflows/linux_debug.yml
     :alt: Linux CI (Debug)

.. |github_actions_linux_hip_status| image:: https://github.com/pika-org/pika/actions/workflows/linux_hip.yml/badge.svg
     :target: https://github.com/pika-org/pika/actions/workflows/linux_hip.yml
     :alt: Linux CI (HIP, Debug)

.. |github_actions_linux_sanitizers_status| image:: https://github.com/pika-org/pika/actions/workflows/linux_sanitizers.yml/badge.svg
     :target: https://github.com/pika-org/pika/actions/workflows/linux_sanitizers.yml
     :alt: Linux CI (asan/ubsan)

.. |github_actions_macos_debug_status| image:: https://github.com/pika-org/pika/actions/workflows/macos_debug.yml/badge.svg
     :target: https://github.com/pika-org/pika/actions/workflows/macos_debug.yml
     :alt: macOS CI (Debug)

.. |codacy| image:: https://api.codacy.com/project/badge/Grade/e03f57f1c4cd40e7b514e552a723c125
     :target: https://www.codacy.com/gh/pika-org/pika
     :alt: Codacy

.. |codacy_coverage| image:: https://api.codacy.com/project/badge/Coverage/e03f57f1c4cd40e7b514e552a723c125
     :target: https://www.codacy.com/gh/pika-org/pika
     :alt: Codacy coverage
