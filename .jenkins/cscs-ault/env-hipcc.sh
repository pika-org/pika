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
stdexec_version="git.5e378418d9360a1a53392baae8502c4961f5f4c8=main"

# The xnack- architectures are not supported by rocblas@5.3.3
spack_spec="pika@main+rocm+stdexec amdgpu_target='gfx906:xnack-' arch=${spack_arch} %${spack_compiler} malloc=system cxxstd=${cxx_std} ^boost@${boost_version} ^hwloc@${hwloc_version} ^hip@${hip_version} ^whip amdgpu_target='gfx906:xnack-' ^stdexec@${stdexec_version}"

configure_extra_options+=" -DCMAKE_BUILD_RPATH=$(spack location -i ${spack_compiler})/lib64"
configure_extra_options+=" -DCMAKE_HIP_ARCHITECTURES=gfx906:xnack-"
configure_extra_options+=" -DPIKA_WITH_HIP=ON"
configure_extra_options+=" -DPIKA_WITH_CXX_STANDARD=${cxx_std}"
configure_extra_options+=" -DPIKA_WITH_MALLOC=system"
configure_extra_options+=" -DPIKA_WITH_STDEXEC=ON"

build_extra_options+=" -j16"
