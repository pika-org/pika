# Copyright (c) 2020 ETH Zurich
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


cxx_std="20"
gcc_version="12.2.0"
boost_version="1.79.0"
hip_version="5.3.3"
hwloc_version="2.6.0"
spack_compiler="gcc@${gcc_version}"
spack_arch="linux-centos8-zen"
stdexec_version="c0fe16f451137b00ceb85d912ec3a288990ce766"

# The xnack- architectures are not supported by rocblas@5.3.3
spack_spec="pika@main+rocm+p2300 amdgpu_target='gfx900,gfx906:xnack-' arch=${spack_arch} %${spack_compiler} malloc=system cxxstd=${cxx_std} ^boost@${boost_version} ^hwloc@${hwloc_version} ^hip@${hip_version} ^whip amdgpu_target='gfx900,gfx906:xnack-' ^stdexec@${stdexec_version}"

configure_extra_options+=" -DCMAKE_BUILD_RPATH=$(spack location -i ${spack_compiler})/lib64"
configure_extra_options+=" -DCMAKE_HIP_ARCHITECTURES=gfx900;gfx906:xnack-"
configure_extra_options+=" -DPIKA_WITH_HIP=ON"
configure_extra_options+=" -DPIKA_WITH_CXX_STANDARD=${cxx_std}"
configure_extra_options+=" -DPIKA_WITH_MALLOC=system"
configure_extra_options+=" -DPIKA_WITH_P2300_REFERENCE_IMPLEMENTATION=ON"
