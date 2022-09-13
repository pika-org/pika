# Copyright (c) 2020 ETH Zurich
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cxx_std="20"
clang_version="14.0.6"
boost_version="1.79.0"
hwloc_version="2.7.0"
cuda_version="11.5.0"
spack_compiler="clang@${clang_version}"
spack_arch="cray-cnl7-haswell"

spack_spec="pika@main arch=${spack_arch} %${spack_compiler} +cuda malloc=system cxxstd=${cxx_std} +p2300 ^boost@${boost_version} ^cuda@${cuda_version} +allow-unsupported-compilers ^hwloc@${hwloc_version}"

configure_extra_options+=" -DPIKA_WITH_CXX_STANDARD=${cxx_std}"
configure_extra_options+=" -DPIKA_WITH_MALLOC=system"
configure_extra_options+=" -DPIKA_WITH_SPINLOCK_DEADLOCK_DETECTION=ON"
configure_extra_options+=" -DPIKA_WITH_P2300_REFERENCE_IMPLEMENTATION=ON"
