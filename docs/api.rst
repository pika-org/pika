..
    Copyright (c) 2023 ETH Zurich

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

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
- ``pika/cuda.hpp``
- ``pika/execution.hpp``
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

.. literalinclude:: ../examples/documentation/drop_value_documentation.cpp
   :language: c++
   :start-at: #include

.. doxygenvariable:: pika::execution::experimental::drop_operation_state

.. literalinclude:: ../examples/documentation/drop_operation_state_documentation.cpp
   :language: c++
   :start-at: #include

.. doxygenvariable:: pika::execution::experimental::split_tuple

.. literalinclude:: ../examples/documentation/split_tuple_documentation.cpp
   :language: c++
   :start-at: #include

.. doxygenvariable:: pika::execution::experimental::unpack

.. literalinclude:: ../examples/documentation/unpack_documentation.cpp
   :language: c++
   :start-at: #include

.. doxygenvariable:: pika::execution::experimental::when_all_vector

.. literalinclude:: ../examples/documentation/when_all_vector_documentation.cpp
   :language: c++
   :start-at: #include
