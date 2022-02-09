# Copyright (c) 2020-2022 ETH Zurich
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

export CRAYPE_LINK_TYPE=dynamic
export CXX_STD="17"

# apex +openmp does not currently build so we disable openmp
spack_spec="pika@main arch=cray-cnl7-haswell %gcc@10.3.0 +apex malloc=system cxxstd=${CXX_STD} ^apex ~openmp"

configure_extra_options+=" -DPIKA_WITH_CXX_STANDARD=${CXX_STD}"
configure_extra_options+=" -DPIKA_WITH_APEX=ON"
configure_extra_options+=" -DPIKA_WITH_MALLOC=system"
configure_extra_options+=" -DPIKA_WITH_COMPILER_WARNINGS=ON"
configure_extra_options+=" -DPIKA_WITH_COMPILER_WARNINGS_AS_ERRORS=ON"
configure_extra_options+=" -DPIKA_WITH_SPINLOCK_DEADLOCK_DETECTION=ON"
configure_extra_options+=" -DPIKA_WITH_TESTS_HEADERS=ON"

build_extra_options+=" -j10"
