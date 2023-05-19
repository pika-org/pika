# Copyright (c) 2020 ETH Zurich
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cxx_std="17"
gcc_version="12.1.0"
boost_version="1.82.0"
hwloc_version="2.9.1"
cuda_version="12.0.0"
spack_compiler="gcc@${gcc_version}"
spack_arch="cray-cnl7-haswell"

spack_spec="pika@main arch=${spack_arch} %${spack_compiler} +cuda malloc=system cxxstd=${cxx_std} ^boost@${boost_version} ^hwloc@${hwloc_version} ^cuda@${cuda_version}"

configure_extra_options+=" -DPIKA_WITH_CXX_STANDARD=${cxx_std}"
configure_extra_options+=" -DPIKA_WITH_MALLOC=system"
configure_extra_options+=" -DPIKA_WITH_CUDA=ON"
configure_extra_options+=" -DCMAKE_CUDA_ARCH=60=ON"

# All async_cuda tests are disabled from running because the driver on Piz Daint
# is too old for CUDA 12.
test_extra_options+=" EXCLUDE async_cuda"
