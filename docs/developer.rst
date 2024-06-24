..
    Copyright (c) 2024 ETH Zurich

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _develop:

=======================
Developer documentation
=======================

Contributing
============

Contributions are always welcome. Before starting work on larger features it's recommended to
coordinate design ideas through GitHub issues. Existing open issues may sometimes no longer be
useful, have already been solved, or the notes on solving a particular problem may be outdated.
Issues that we think are easier for someone new to the project to get started with are `labeled on
the issue tracker <https://github.com/pika-org/pika/issues/1118>`__.

Building documentation
======================

Building pika's documentation requires doxygen, sphinx, as well the sphinx extensions and themes
listed in ``docs/requirements.txt``. Building the documentation is not done through CMake as the rest of
pika. Assuming ``SOURCE_DIR`` is set to the pika source directory, and ``BUILD_DIR`` is set to a
build directory (e.g. ``$SOURCE_DIR/build``), you first need to export the following environment
variables for doxygen:

.. code-block:: bash

   export PIKA_DOCS_DOXYGEN_INPUT_ROOT="$SOURCE_DIR"
   export PIKA_DOCS_DOXYGEN_OUTPUT_DIRECTORY="$BUILD_DIR/docs/doxygen"

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

Release procedure
=================

The current target is to produce a new (minor) release once a month. See `milestones
<https://github.com/pika-org/pika/milestones>`__ for the planned dates for the next releases.

pika follows `Semantic Versioning <https://semver.org>`__.

#. For minor and major releases: create and check out a new branch at an
   appropriate point on ``main`` with the name ``release-major.minor.X``.
   ``major`` and ``minor`` should be the major and minor versions of the
   release. For patch releases: check out the corresponding
   ``release-major.minor.X`` branch.

#. Write release notes in ``docs/changelog.md``. Check for issues and pull requests
   for the release on the
   `pika planning board <https://github.com/orgs/pika-org/projects/1>`__. Check
   for items that do not have a release associated to them on the `Done` view.
   Assign them to a release if needed.

#. Make sure ``PIKA_VERSION_MAJOR/MINOR/PATCH`` in ``CMakeLists.txt`` contain
   the correct values. Change them if needed.

#. When making a post-1.0.0 major release, remove deprecated functionality if
   appropriate.

#. Update the minimum required versions if necessary.

#. Check that projects dependent on pika are passing CI with pika main branch.
   Check if there is no performance regressions due to the pika upgrade in
   those projects.

#. Repeat the following steps until satisfied with the release.

   #. Change ``PIKA_VERSION_TAG`` in ``CMakeLists.txt`` to ``-rcN``, where ``N``
      is the current iteration of this step. Start with ``-rc1``.

   #. Create a pre-release on GitHub using the script ``tools/roll_release.sh``.
      This script automatically tags with the corresponding release number.

   #. Add patches as needed to the release candidate until the next release
      candidate, or the final release.

#. Change ``PIKA_VERSION_TAG`` in ``CMakeLists.txt`` to an empty string.

#. Add the release date to the caption of the current ``docs/changelog.md`` section
   and change the value of ``PIKA_VERSION_DATE`` in ``CMakeLists.txt``.

#. Create a release on GitHub using the script ``tools/roll_release.sh``. This
   script automatically tags the release with the corresponding release number.
   You'll need to set ``GITHUB_TOKEN`` or both ``GITHUB_USER`` and
   ``GITHUB_PASSWORD`` for the hub release command. When creating a
   ``GITHUB_TOKEN``, the only access necessary is ``public_repo``.

#. Merge release branch into ``main`` (with --no-ff).

#. Modify the release procedure if necessary.

#. Change ``PIKA_VERSION_TAG`` in ``CMakeLists.txt`` back to ``-trunk``.

#. Update spack (``https://github.com/spack/spack``).

#. Clean up the `pika planning board <https://github.com/orgs/pika-org/projects/1>`__:

   - Move the version-specific views one release forward (e.g. change the name
     and filter of `pika 0.X.0` to `pika 0.<X+1>.0`).
   - Change the status of `Done` items to `Archive` in the `Done` view.

#. Delete your ``GITHUB_TOKEN`` if created only for the release.

#. Announce the release on the #pika slack channel.
