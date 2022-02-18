# Copyright (c) 2020-2022 ETH Zurich
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cxx_std="17"
gcc_version="10.3.0"
spack_compiler="gcc@${gcc_version}"
spack_arch="cray-cnl7-haswell"

# apex +openmp does not currently build so we disable openmp
spack_spec="pika@main arch=${spack_arch} %${spack_compiler} +apex malloc=system cxxstd=${cxx_std} ^apex ~openmp"

configure_extra_options+=" -DPIKA_WITH_CXX_STANDARD=${cxx_std}"
configure_extra_options+=" -DPIKA_WITH_APEX=ON"
configure_extra_options+=" -DPIKA_WITH_MALLOC=system"
configure_extra_options+=" -DPIKA_WITH_COMPILER_WARNINGS=ON"
configure_extra_options+=" -DPIKA_WITH_COMPILER_WARNINGS_AS_ERRORS=ON"
configure_extra_options+=" -DPIKA_WITH_SPINLOCK_DEADLOCK_DETECTION=ON"
configure_extra_options+=" -DPIKA_WITH_TESTS_HEADERS=ON"

build_extra_options+=" -j10"
