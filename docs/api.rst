..
    Copyright (c) 2023 ETH Zurich

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

:tocdepth: 3

.. _api:

=============
API reference
=============

pika follows `semver <https://semver.org>`__. pika is currently at a 0.X version which means that
minor versions may break the API. pika gives no guarantees about ABI stability. The ABI may change
even in patch versions.

The API reference is a work in progress. While the reference is being expanded, the
:ref:`std_execution_more_resources` section contains useful links to high level overviews and low
level API descriptions of ``std::execution``.

The following headers are part of the public API. Any other
headers are internal implementation details.

.. contents::
   :local:
   :depth: 1
   :backlinks: none

These headers are part of the public API, but are currently undocumented.

- ``pika/async_rw_mutex.hpp``
- ``pika/barrier.hpp``
- ``pika/condition_variable.hpp``
- ``pika/latch.hpp``
- ``pika/mpi.hpp``
- ``pika/mutex.hpp``
- ``pika/runtime.hpp``
- ``pika/semaphore.hpp``
- ``pika/thread.hpp``

All functionality in a namespace containing ``detail`` and all macros prefixed with ``PIKA_DETAIL``
are implementation details and may change without warning at any time. All functionality in a
namespace containing ``experimental`` may change without warning at any time. However, the intention
is to stabilize those APIs over time.

.. _header_pika_init:

``pika/init.hpp``
=================

The ``pika/init.hpp`` header provides functionality to manage the pika runtime.

.. literalinclude:: ../examples/documentation/init_hpp_documentation.cpp
   :language: c++
   :start-at: #include

.. doxygenfunction:: pika::start(int argc, char const* const* argv, init_params const &params)
.. doxygenfunction:: pika::stop()
.. doxygenfunction:: pika::finalize()
.. doxygenfunction:: pika::wait()
.. doxygenfunction:: pika::resume()
.. doxygenfunction:: pika::suspend()
.. doxygenfunction:: pika::is_runtime_initialized()

.. versionadded:: 0.22.0

.. doxygenstruct:: pika::init_params
   :members:

.. _header_pika_execution:

``pika/execution.hpp``
======================

The ``pika/execution.hpp`` header provides functionality related to ``std::execution``.
``std::execution`` functionality, including extensions provided by pika, is defined in the
``pika::execution::experimental`` namespace. When the CMake option ``PIKA_WITH_STDEXEC`` is enabled,
pika pulls the ``stdexec`` namespace into ``pika::execution::experimental``.

See :ref:`pika_stdexec` and :ref:`std_execution_more_resources` for more details on how pika relates
to ``std::execution`` and for more resources on learning about ``std::execution``. Documentation for
sender functionality added to the C++ standard in the above resources apply to both pika's and
stdexec's implementations of them.

Documented below are sender adaptors not available in stdexec or not proposed for standardization.

All sender adaptors are `customization point objects (CPOs)
<https://eel.is/c++draft/customization.point.object>`__.

.. doxygenvariable:: pika::execution::experimental::drop_value

.. versionadded:: 0.6.0

.. literalinclude:: ../examples/documentation/drop_value_documentation.cpp
   :language: c++
   :start-at: #include

.. doxygenvariable:: pika::execution::experimental::drop_operation_state

.. versionadded:: 0.19.0

.. literalinclude:: ../examples/documentation/drop_operation_state_documentation.cpp
   :language: c++
   :start-at: #include

.. doxygenvariable:: pika::execution::experimental::require_started

.. versionadded:: 0.21.0

.. literalinclude:: ../examples/documentation/require_started_documentation.cpp
   :language: c++
   :start-at: #include

.. doxygenvariable:: pika::execution::experimental::split_tuple

.. versionadded:: 0.12.0

.. literalinclude:: ../examples/documentation/split_tuple_documentation.cpp
   :language: c++
   :start-at: #include

.. doxygenvariable:: pika::execution::experimental::unpack

.. versionadded:: 0.17.0

.. literalinclude:: ../examples/documentation/unpack_documentation.cpp
   :language: c++
   :start-at: #include

.. doxygenvariable:: pika::execution::experimental::when_all_vector

.. versionadded:: 0.2.0

.. literalinclude:: ../examples/documentation/when_all_vector_documentation.cpp
   :language: c++
   :start-at: #include

.. _header_pika_cuda:

``pika/cuda.hpp``
=================

The ``pika/cuda.hpp`` header provides functionality related to CUDA and HIP. All functionality is
under the ``pika::cuda::experimental`` namespace and class and function names contain ``cuda``, even
when HIP support is enabled. CUDA and HIP functionality can be enabled with the CMake options
``PIKA_WITH_CUDA`` and ``PIKA_WITH_HIP``, respectively. In the following, whenever CUDA is
mentioned, it refers to to CUDA and HIP interchangeably.

.. note::
   https://github.com/pika-org/pika/issues/116 tracks a potential renaming of the functionality
   to avoid using ``cuda`` even when HIP is enabled. If you have feedback on a rename or just want
   to follow along, please see that issue.

.. warning::
   At the moment, ``nvcc`` can not compile stdexec headers. Of the CUDA compilers, only ``nvc++`` is
   able to compile stdexec headers. If you have stdexec support enabled in pika, either ensure that
   ``.cu`` files do not include stdexec headers, or use ``nvc++`` to compile your application.
   However, ``nvc++`` does not officially support compiling device code. Use at your own risk.

   For HIP there are no known restrictions.

The CUDA support in pika relies on four major components:

1. A pool of CUDA streams as well as cuBLAS and cuSOLVER handles. These streams and handles are used
   in a round-robin fashion by various sender adaptors.
2. A CUDA scheduler, in the ``std::execution`` sense. This uses the CUDA pool to schedule work on a
   GPU.
3. Sender adaptors. A few special-purpose sender adaptors, as well as customizations of a few
   ``std::execution`` adaptors are provided to help schedule different types of work on a GPU.
4. Polling of CUDA events integrated into the pika scheduling loop. This integration is essential to
   avoid calling e.g. ``cudaStreamSynchronize`` on a pika task, which would block the underlying
   worker thread and thus block progress of other work.

The following example gives an overview of using the above CUDA functionalities in pika:

.. literalinclude:: ../examples/documentation/cuda_overview_documentation.cu
   :language: c++
   :start-at: #include

.. note::
   pika uses `whip <https://github.com/eth-cscs/whip>`__ internally for portability between CUDA and
   HIP. However, users of pika are not forced to use whip as whip only creates aliases for CUDA/HIP
   types and enumerations. whip is thus compatible with directly using the types and enumerations
   provided by CUDA/HIP.

While :cpp:class:`pika::cuda::experimental::cuda_pool` gives direct access to streams and handles,
the recommended way to access them is through the sender adaptors available below.

.. doxygenclass:: pika::cuda::experimental::cuda_scheduler
.. doxygenstruct:: pika::cuda::experimental::then_with_stream_t
.. doxygenvariable:: pika::cuda::experimental::then_with_stream

.. literalinclude:: ../examples/documentation/then_with_stream_documentation.cu
    :language: c++
    :start-at: #include

.. doxygenstruct:: pika::cuda::experimental::then_with_cublas_t
.. doxygenvariable:: pika::cuda::experimental::then_with_cublas

.. literalinclude:: ../examples/documentation/then_with_cublas_documentation.cu
    :language: c++
    :start-at: #include

.. doxygenstruct:: pika::cuda::experimental::then_with_cusolver_t
.. doxygenvariable:: pika::cuda::experimental::then_with_cusolver

See :cpp:var:`pika::cuda::experimental::then_with_cublas` for an example of what can be done with
:cpp:var:`pika::cuda::experimental::then_with_cusolver`. The interfaces are identical except for the
type of handle passed to the callable.

.. doxygenclass:: pika::cuda::experimental::cuda_pool
.. doxygenclass:: pika::cuda::experimental::cuda_stream
.. doxygenclass:: pika::cuda::experimental::cublas_handle
.. doxygenclass:: pika::cuda::experimental::locked_cublas_handle
.. doxygenclass:: pika::cuda::experimental::cusolver_handle
.. doxygenclass:: pika::cuda::experimental::locked_cusolver_handle
