..
    Copyright (c) 2022-2023 ETH Zurich

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _usage:

=====
Usage
=====

.. _getting_started:

Getting started
===============

The recommended way to install pika is through `spack <https://spack.readthedocs.io>`__:

.. code-block:: bash

   spack install pika

See

.. code-block:: bash

   spack info pika

for available options.

pika is currently available in the following repositories.

.. |repology| image:: https://repology.org/badge/vertical-allrepos/pika-concurrency-library.svg
     :target: https://repology.org/project/pika-concurrency-library/versions
     :alt: Packaging status

|repology|

Manual installation
-------------------

If you'd like to build pika manually you will need CMake 3.22.0 or greater and a recent C++ compiler
supporting C++17:

- `GCC <https://gcc.gnu.org>`__ 9 or greater
- `clang <https://clang.llvm.org>`__ 11 or greater

Additionally, pika depends on:

- `header-only Boost <https://boost.org>`__ 1.71.0 or greater
- `hwloc <https://www-lb.open-mpi.org/projects/hwloc/>`__ 1.11.5 or greater
- `fmt <https://fmt.dev/latest/index.html>`__ 9.0.0 or greater

pika optionally depends on:

* `gperftools/tcmalloc <https://github.com/gperftools/gperftools>`__, `jemalloc
  <http://jemalloc.net/>`__, or `mimalloc <https://github.com/microsoft/mimalloc>`__. It is *highly*
  recommended to use one of these allocators as they perform significantly better than the system
  allocators. You can set the allocator through the CMake variable ``PIKA_WITH_MALLOC``. If you want
  to use the system allocator (e.g. for debugging) you can do so by setting
  ``PIKA_WITH_MALLOC=system``.
* `CUDA <https://docs.nvidia.com/cuda/>`__ 11.0 or greater. CUDA support can be enabled with
  ``PIKA_WITH_CUDA=ON``. pika can also be built with nvc++ from the `NVIDIA HPC SDK
  <https://developer.nvidia.com/hpc-sdk>`__. In the latter case, set ``CMAKE_CXX_COMPILER`` to
  ``nvc++``.
* `HIP <https://rocmdocs.amd.com/en/latest/index.html>`__ 5.2.0 or greater. HIP support can be
  enabled with ``PIKA_WITH_HIP=ON``.
* `MPI <https://www.mpi-forum.org/>`__. MPI support can be enabled with ``PIKA_WITH_MPI=ON``.
* `Boost.Context <https://boost.org>`__ on macOS or exotic platforms which are not supported by the
  default user-level thread implementations in pika. This can be enabled with
  ``PIKA_WITH_BOOST_CONTEXT=ON``.
* `stdexec <https://github.com/NVIDIA/stdexec>`__. stdexec support can be enabled with
  ``PIKA_WITH_STDEXEC=ON`` (currently tested with commit `8bc7c7f06fe39831dea6852407ebe7f6be8fa9fd
  <https://github.com/NVIDIA/stdexec/tree/8bc7c7f06fe39831dea6852407ebe7f6be8fa9fd>`__).  The
  integration is experimental. See :ref:`pika_stdexec` for more information about the integration.

If you are using `nix <https://nixos.org>`__ you can also use the ``shell.nix`` file provided at the
root of the repository to quickly enter a development environment:

.. code-block:: bash

   nix-shell <pika-root>/shell.nix

The ``nixpkgs`` version is not pinned and may break occasionally.

Including in CMake projects
---------------------------

Once installed, pika can be used in a CMake project through the ``pika::pika`` target:

.. code-block:: cmake

   find_package(pika REQUIRED)
   add_executable(app main.cpp)
   target_link_libraries(app PRIVATE pika::pika)

Other ways of depending on pika are likely to work but are not officially supported.

.. _cmake_configuration:

Customizing the pika installation
=================================

The most important CMake options are listed in :ref:`getting_started`. Below is a more complete list
of CMake options you can use to customize the installation.

- ``PIKA_WITH_MALLOC``: This defaults to ``mimalloc`` which requires mimalloc to be installed.  Can
  be set to ``tcmalloc``, ``jemalloc``, ``mimalloc``, or ``system``. Setting it to ``system`` can be
  useful in debug builds.
- ``PIKA_WITH_CUDA``: Enable CUDA support.
- ``PIKA_WITH_HIP``: Enable HIP support.
- ``PIKA_WITH_MPI``: Enable MPI support.
- ``PIKA_WITH_STDEXEC``: Enable `stdexec <https://github.com/NVIDIA/stdexec>`__ support.
- ``PIKA_WITH_APEX``: Enable `APEX <https://uo-oaciss.github.io/apex>`__ support.
- ``PIKA_WITH_TRACY``: Enable `Tracy <https://github.com/wolfpld/tracy>`__ support.
- ``PIKA_WITH_BOOST_CONTEXT``: Use Boost.Context for user-level thread context switching.
- ``PIKA_WITH_TESTS``: Enable tests. Tests can be built with ``cmake --build . --target tests`` and
  run with ``ctest --output-on-failure``.
- ``PIKA_WITH_EXAMPLES``: Enable examples. Binaries will be placed under ``bin`` in the build
  directory.

Testing
-------

Tests and examples are disabled by default and can be enabled with ``PIKA_WITH_TESTS``,
``PIKA_WITH_TESTS_{BENCHMARKS,REGRESSIONS,UNIT}``, and ``PIKA_WITH_EXAMPLES``. The tests must be
explicitly built before running them, e.g.  with ``cmake --build . --target tests && ctest
--output-on-failure``.

.. _thread_bindings:

Controlling the number of threads and thread bindings
=====================================================

The thread pool created by the pika runtime will by default be created with a number of threads
equal to the number of cores on the system. The number of threads can explicitly be controlled by a
few command line options. The most straightforward way of changing the number of threads is with the
``--pika:threads`` command line option. It takes an explicit number of threads. Alternatively it can
also be passed the special values ``cores`` (the default, use one thread per core) or ``all`` (use
one thread per hyperthread).

Process masks
-------------

Many batch systems and e.g. MPI can set a process mask on the application to restrict on what cores
an application can run. pika will by default take this process mask into account when determining
how many threads to use for the runtime. ``hwloc-bind`` can also be used to manually set a process
mask on the application. When a process mask is set, the default behaviour is to use only one thread
per core in the process mask. Setting ``--pika:threads`` to a number higher than the number of cores
available in the mask is not allowed. Using ``--pika:threads=all`` will use all the hyperthreads in
the process mask.

The process mask can explicitly be ignored with the option ``--pika:ignore-process-mask`` or
overridden with ``--pika:process-mask``. With ``--pika:ignore-process-mask`` pika behaves as if no
process mask is set. ``--pika:process-mask`` takes an explicit hexadecimal string (beginning with
``0x``) representing the process mask to use. The mask can also be set with the environment variable
``PIKA_PROCESS_MASK``. ``--pika:process-mask`` takes precedence over ``PIKA_PROCESS_MASK``.
``--pika:print-bind`` can be used to verify that the bindings used by pika are correct. Exporting
the environment variable ``PIKA_PRINT_BIND`` (any value) is equivalent to using the
``--pika:print-bind`` option.

Interaction with OpenMP
-----------------------

When pika is used together with OpenMP extra care may be needed to ensure pika uses the correct
process mask. This is because with OpenMP the main thread participates in parallel regions and if
OpenMP binds threads to cores, the main thread may have a mask set to a single core before pika can
read the mask. Typically, OpenMP will bind threads to cores if the ``OMP_PROC_BIND`` or
``OMP_PLACES`` environment variables are set. Some implementations of OpenMP (e.g. LLVM) set the
binding of the main thread only at the first parallel region which means that if pika is initialized
before the first parallel region, the mask will most likely be read correctly. Other implementations
(e.g. GNU) set the binding of the main thread in global constructors which may run before pika can
read the process mask. In that case you may need to either use ``--pika:ignore-process-mask`` to use
all cores on the system or explicitly set a mask with ``--pika:process-mask``. If there is a process
mask already set in the environment that is launching the application (e.g. in a SLURM job) you can
read the mask before the application runs with hwloc:

.. code-block:: bash

   ./app --pika:process-mask=$(hwloc-bind --get --taskset)

``pika-bind`` helper script
---------------------------

Since version ``0.20.0``, the ``pika-bind`` helper script is bundled with pika. ``pika-bind`` sets the
``PIKA_PROCESS_MASK`` environment variable based on process mask information found before the pika runtime is started,
and then runs the given command. ``pika-bind`` is a more convenient alternative to manually setting ``PIKA_PROCESS_MASK``
when pika is used together with a runtime that may reset the process mask of the main thread, like OpenMP.

.. _logging:

Logging
=======

The pika runtime uses `spdlog <https://github.com/gabime/spdlog>`__ for logging. Warnings and more
severe messages are logged by default. To change the logging level, set the ``PIKA_LOG_LEVEL``
environment variable to a value between 0 (trace) and 6 (off) (the values correspond to levels in
spdlog). The log messages are sent to stderr by default. The destination can be changed by setting
the ``PIKA_LOG_DESTINATION`` environment variable. Supported values are:

- ``cerr``
- ``cout``
- any other value is interpreted as a path to a file

pika will by default print messages in the following format:

.. code-block::

   [2024-04-18 13:45:07.095279283] [pika] [info] [host:machine/----] [pid:2786603] [tid:2786607] [pool:0000/0003/0003] [parent:----/----] [task:0x7fa6a4077cf0/pika_main] [set_thread_state.cpp:205:set_thread_state] set_thread_state: thread(0x7fa6a802c8d0), description(<unknown>), new state(pending), old state(suspended)

The fields are as follows:

- ``[2024-04-18 13:45:07.095279283]``: The timestamp of the message.
- ``[pika]``: An identifier present in all pika's logs.
- ``[info]``: The severity level of the message.
- ``[host:machine/----]``: The hostname and the MPI rank of the process (``----`` if MPI is
  disabled).
- ``[pid:2786603]``: The process id as reported by the operating system.
- ``[tid:2786607]``: The thread id as reported by the operating system.
- ``[pool:0000/0003/0003]``: The pika thread pool and worker thread ids: the first component is the
  thread pool id, the second is the global worker thread id (unique across all thread pools), and
  the third is the local worker thread id (unique only within the current thread pool).
- ``[parent:----/----]``: The id and description of the parent task that spawned the current task.
- ``[task:0x7fa6a4077cf0/pika_main]``: The id and description of the current task.
- ``[set_thread_state.cpp:205/set_thread_state]``: The file, line number, and function where the
  message was logged.
- The logged message is printed last.

The pool field is ``[pool:----/----/----]`` when a message is logged from a thread that does not
belong to the pika runtime. The main thread will only have the global thread id set, e.g.
``[pool:----/0004/----]``.

Task ids and descriptions are logged as ``----/----`` when there is no current or parent task. Task
descriptions are only printed when enabled with APEX and Tracy support, or with the CMake option
``PIKA_WITH_THREAD_DEBUG_INFO``.

The log message format can be changed by setting the environment variable ``PIKA_LOG_FORMAT`` to a
format string supported by spdlog. The custom fields defined by pika can be accessed with the
following:

- ``%j``: The hostname and MPI rank.
- ``%w``: The thread pool and worker thread ids.
- ``%q``: The parent task id and description.
- ``%k``: The current task id and description.

.. _malloc:

Using custom allocators with pika
=================================

Typical use of pika can often lead to many small allocations from many different threads,
potentially leading to suboptimal performance with the system allocator. By default, pika uses
`mimalloc <https://github.com/microsoft/mimalloc>`__ as the memory allocator because it usually
performs significantly better than the system allocator. In some cases, the system allocator or
other custom allocators might perform better.

Setting the following environment variables usually further improves performance with mimalloc:

- ``MIMALLOC_EAGER_COMMIT_DELAY=0``
- ``MIMALLOC_ALLOW_LARGE_OS_PAGES=1``

We have observed mimalloc performing worse than the defaults with the above options on some systems,
as well as worse than the system allocator. Always benchmark to find the most suitable allocator for
your workload and system.

To ease testing of different allocators, you may also configure pika with the system allocator and
instead use ``LD_PRELOAD`` to replace the default allocator at runtime. This allows you to choose
the allocator without rebuilding pika. To do so, export the ``LD_PRELOAD`` environment variable to
point to the shared library of the allocator. For example, to use `jemalloc
<https://jemalloc.net>`__, set ``LD_PRELOAD`` to the full path of ``libjemalloc.so``:

.. code-block:: bash

   export LD_PRELOAD=/path/to/libjemalloc.so

.. _pika_stdexec:

Relation to std::execution and stdexec
======================================

When pika was first created as a fork of `HPX <https://github.com/STEllAR-GROUP/hpx>`__ in 2022
stdexec was in its infancy. Because of this, pika contains an implementation of a subset of the
earlier revisions of P2300. The main differences to stdexec and the proposed facilities are:

- The pika implementation uses C++17 and thus does not make use of concepts or coroutines. This
  allows compatibility with slightly older compiler versions and e.g. nvcc.
- The pika implementation uses ``value_types``, ``error_types``, and ``sends_done`` instead of
  ``completion_signatures`` in sender types, as in the `first 3 revisions of P2300
  <https://wg21.link/p2300r3>`__.
- ``pika::this_thread::experimental::sync_wait`` differs from ``std::this_thread::sync_wait``
  in that the former expects the sender to send a single value which is returned directly by
  ``sync_wait``. If no value is sent by the sender, ``sync_wait`` returns ``void``.  Errors in
  ``set_error`` are thrown and ``set_stopped`` is not supported.

pika has an experimental CMake option ``PIKA_WITH_STDEXEC`` which can be enabled to use stdexec for
the P2300 facilities. pika brings the ``stdexec`` namespace into ``pika::execution::experimental``,
but provides otherwise no guarantees of interchangeable functionality. pika only implements a subset
of the proposed sender algorithms which is why we recommend that you enable ``PIKA_WITH_STDEXEC``
whenever possible. We plan to deprecate and remove the P2300 implementation in pika in favour of
stdexec and/or standard library implementations.

.. _std_execution_more_resources:

More resources
==============

.. |stdexec_resources| replace:: list of presentations, blog posts etc. about the ``std::execution`` model
.. _stdexec_resources: https://github.com/NVIDIA/stdexec#resources

.. |cppreference_execution| replace:: documentation about ``std::execution``
.. _cppreference_execution: https://en.cppreference.com/w/cpp/experimental/execution

The `C++ standard <https://eel.is/c++draft/exec>`__ is the source of truth for ``std::execution``.
The `P2300 proposal <https://wg21.link/p2300>`__ also contains both the wording for the majority of
``std::execution`` functionality as well as the motivation for it. The reference implementation of
P2300, stdexec, maintains a |stdexec_resources|_.  In addition to the above, other implementations
of the ``std::execution`` model exist, with useful documentation and examples:

- `HPX <https://hpx-docs.stellar-group.org/latest/html/index.html>`__
- `libunifex <https://github.com/facebookexperimental/libunifex/blob/main/doc/overview.md>`__
- `C++ Baremetal Senders & Receivers <https://intel.github.io/cpp-baremetal-senders-and-receivers/>`__
- `execution26 <https://github.com/beman-project/execution26>`__

Even though the implementations differ, the concepts are transferable between implementations and
useful for learning. cppreference.com also contains early |cppreference_execution|_.

pika has been presented at the following events and slides of the presentations are public:

- `CERN Computing seminar in 2022 <https://indico.cern.ch/category/82/>`__: introduction to pika and
  DLA-Future (`slides <https://indico.cern.ch/event/1194848/>`__)
- `The SOS-25 workshop in 2023 <https://sos-25.highspeedcomputing.org/home>`__: an overview of use of
  ``std::execution`` at the Swiss National Supercomputing Centre, covering uses of pika and HPX in
  DLA-Future, Octo-Tiger, and and Kokkos (`slides
  <https://drive.google.com/file/d/1rs-iosjFZJzBm1nsVwnhr6qjWbzdRmpc/view>`__)
