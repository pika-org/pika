# Copyright (c) 2020 ETH Zurich
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

export CRAYPE_LINK_TYPE=dynamic
export CXX_STD="17"
export BOOST_ROOT=/apps/dom/UES/jenkins/7.0.UP03/21.09/dom-gpu/software/Boost/1.78.0-CrayGNU-21.09
export HWLOC_ROOT=/apps/dom/UES/jenkins/7.0.UP03/21.09/dom-gpu/software/hwloc/2.4.1/

module switch PrgEnv-cray PrgEnv-gnu
module switch gcc gcc/9.3.0
module load cudatoolkit/21.5_11.3
module load CMake/3.22.1

export CXX=`which CC`
export CC=`which cc`

configure_extra_options+=" -DPIKA_WITH_CXX_STANDARD=${CXX_STD}"
configure_extra_options+=" -DPIKA_WITH_MALLOC=system"
configure_extra_options+=" -DPIKA_WITH_CUDA=ON"
configure_extra_options+=" -DPIKA_WITH_EXAMPLES_OPENMP=ON"
configure_extra_options+=" -DPIKA_WITH_COMPILER_WARNINGS=ON"
configure_extra_options+=" -DPIKA_WITH_COMPILER_WARNINGS_AS_ERRORS=ON"
configure_extra_options+=" -DPIKA_WITH_TESTS_HEADERS=ON"
