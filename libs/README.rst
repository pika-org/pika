..
    Copyright (c) 2018 Thomas Heller

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

This directory holds modularized libraries pika is build upon. Those libraries
can be seen as independent modules, with clear dependencies and no cycles.

The structure of a module should be as follows:

* ``<lib_name>/``

  * ``README.rst``
  * ``CMakeLists.txt``
  * ``cmake``
  * ``docs/``

    * ``index.rst``

  * ``examples/``

    * ``CMakeLists.txt``

  * ``include/``

    * ``pika/``

      * ``<lib_name>``

  * ``src/``

    * ``CMakeLists.txt``

  * ``tests/``

    * ``CMakeLists.txt``
    * ``unit/``

      * ``CMakeLists.txt``

    * ``regressions/``

      * ``CMakeLists.txt``

    * ``performance/``

      * ``CMakeLists.txt``

A ``README.rst`` should be always included which explains the basic purpose of
the library and a link to the generated documentation.

The ``include`` directory should contain only headers that other libraries need.
Private headers should be placed under the ``src`` directory. This allows for
clear separation. The ``cmake`` subdirectory may include additional |cmake|_
scripts needed to generate the respective build configurations.

Documentation is placed in the ``docs`` folder which contains a empty index.rst.
It is picked up by the main build system and will be part of the generated
documentation.

Note: each file should include the copyright and the license as this one.
