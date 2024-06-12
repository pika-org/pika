..
    Copyright (c) 2024 ETH Zurich

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _develop:

=======================
Developer documentation
=======================

Building documentation
======================

Building pika's documentation requires doxygen, sphinx, as well the sphinx extensions and themes
listed in ``docs/conf.py``. Building the documentation is not done through CMake as the rest of
pika. Assuming ``SOURCE_DIR`` is set to the pika source directory, and ``BUILD_DIR`` is set to a
build directory (e.g. ``$SOURCE_DIR/build``), you first need to export the following environment
variables for doxygen:

.. code-block:: bash

   export PIKA_DOCS_DOXYGEN_INPUT_ROOT="$SOURCE_DIR"
   export PIKA_DOCS_DOXYGEN_OUTPUT_DIRECTORY="$BUILD_DIR/doxygen"

Then generate the doxygen XML files using:

.. code-block:: bash

   doxygen "$SOURCE_DIR/docs/Doxyfile"

Finally, build the sphinx documentation using:

.. code-block:: bash

   sphinx-build -W -b html "$SOURCE_DIR/docs" "$BUILD_DIR/docs"

Assuming the build finished without errors, the HTML documentation will now be in
``$BUILD_DIR/docs`` with the entry point being ``$BUILD_DIR/docs/index.html``.

Doxygen only needs to be rerun if the source code documentation has changed. See the `doxygen
<https://www.doxygen.nl>`__
and `sphinx <https://www.sphinx-doc.org>`__ documentation for more information on using the tools.
