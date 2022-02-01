..
    Copyright (c) 2022 ETH Zurich

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

====
pika
====

pika is a C++ library for concurrency and parallelism. It implements
senders/receivers (as proposed in `P2300 <https://wg21.link/p2300>`_) for CPU
thread pools, MPI, and CUDA.

Dependencies
============

pika requires:

* a C++17-capable compiler:

  * `GCC <https://gcc.gnu.org>`_ 7 or greater
  * `clang <https://clang.llvm.org>`_ 7 or greater
  * MSVC is likely to work but is not regularly tested

* `CMake <https://cmake.org>`_ 3.18.0 or greater
* `header-only Boost <https://boost.org>`_ 1.71.0 or greater
* `hwloc <https://www-lb.open-mpi.org/projects/hwloc/>`_ 1.11.5 or greater

pika optionally requires:

* `gperftools/tcmalloc <https://github.com/gperftools/gperftools>`_, `jemalloc
  <http://jemalloc.net/>`_, or `mimalloc
  <https://github.com/microsoft/mimalloc>`_
* `CUDA <https://docs.nvidia.com/cuda/>`_ 11.0 or greater
* `HIP <https://rocmdocs.amd.com/en/latest/index.html>`_ 4.3.0 or greater
* `MPI <https://www.mpi-forum.org/>`_
* `Boost.Context, Thread, and Chrono <https://boost.org>`_ on macOS
* `Boost.Regex <https://boost.org>`_ when building pika tools

Building
========

pika is built using CMake. Please see the documentation of
CMake for help on how to use it. Dependencies are usually available in
distribution repositories. Alternatively, pika can be built using `spack
<https://spack.readthedocs.io>`_ (`pika spack package
<https://spack.readthedocs.io/en/latest/package_list.html#pika>`_). The pika
repository also includes a ``shell.nix`` file for use with `nix
<https://nixos.org/download.html#download-nix>`_. The file includes dependencies
for regular development. It is provided for convenience only and is not
comprehensive or guaranteed to be up to date. It may require the nixos unstable
channel.

pika is configured using CMake variables. The most important variables are:

* ``PIKA_WITH_MALLOC``: This defaults to ``tcmalloc`` which requires gperftools.
  Can be set to ``tcmalloc``, ``jemalloc``, ``mimalloc``, or ``system``. Setting
  it to ``system`` can be useful in debug builds.
* ``PIKA_WITH_CUDA``: Enable CUDA support.
* ``PIKA_WITH_HIP``: Enable HIP support. To enabled this set the compiler to
  ``hipcc`` instead of setting the variable explicitly.
* ``PIKA_WITH_MPI``: Enable MPI support.
* ``PIKA_WITH_GENERIC_CONTEXT_COROUTINES``: Enables the use of Boost.Context for
  fiber context switching. This has to be enabled on non-Linux and non-x86
  platforms.

Tests and examples are disabled by default and can be enabled with
``PIKA_WITH_TESTS``, ``PIKA_WITH_TESTS_*``, and ``PIKA_WITH_EXAMPLES``.

Acknowledgements
================

pika is a fork of `HPX <https://hpx.stellar-group.org>`_ focusing on the
single-node use case complemented by minimal MPI support.

Name
====

Pick your favourite meaning from the following:

* `pika the animal <https://en.wikipedia.org/wiki/Pika>`_
* `pika- as a prefix for fast in Finnish
  <https://en.wiktionary.org/wiki/pika->`_
* `pika in various other languages <https://en.wiktionary.org/wiki/pika>`_
