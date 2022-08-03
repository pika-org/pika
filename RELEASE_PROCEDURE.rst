..
    Copyright (c)      2022 ETH Zurich
    Copyright (c) 2007-2017 Louisiana State University

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

Release procedure for pika
==========================

The current target is to produce a new (minor) release once a month.

pika follows `Semantic Versioning <https://semver.org>`_.

#. For minor and major releases: create and check out a new branch at an
   appropriate point on ``main`` with the name ``release-major.minor.X``.
   ``major`` and ``minor`` should be the major and minor versions of the
   release. For patch releases: check out the corresponding
   ``release-major.minor.X`` branch.

#. Write release notes in ``CHANGELOG.md``.

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

#. Add the release date to the caption of the current ``CHANGELOG.md`` section
   and change the value of ``PIKA_VERSION_DATE`` in ``CMakeLists.txt``.

#. Create a release on GitHub using the script ``tools/roll_release.sh``. This
   script automatically tags the release with the corresponding release number.
   You'll need to set ``GITHUB_TOKEN`` or both ``GITHUB_USER`` and
   ``GITHUB_PASSWORD`` for the hub release command.

#. Merge release branch into ``main``.

#. Modify the release procedure if necessary.

#. Change ``PIKA_VERSION_TAG`` in ``CMakeLists.txt`` back to ``-trunk``.

#. Update spack (``https://github.com/spack/spack``).
