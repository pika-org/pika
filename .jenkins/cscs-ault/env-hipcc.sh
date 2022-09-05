# Copyright (c) 2020 ETH Zurich
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


cxx_std="17"
gcc_version="10.2.0"
boost_version="1.79.0"
hwloc_version="2.6.0"
spack_compiler="gcc@${gcc_version}"
spack_arch="linux-centos8-zen"

spack_spec="pika@main+rocm arch=${spack_arch} %${spack_compiler} malloc=system cxxstd=${cxx_std} ^boost@${boost_version} ^hwloc@${hwloc_version} ^hip@5.2.0"

configure_extra_options+=" -DCMAKE_BUILD_RPATH=$(spack location -i ${spack_compiler})/lib64"
configure_extra_options+=" -DCMAKE_HIP_ARCHITECTURES=gfx900;gfx906"
configure_extra_options+=" -DWHIP_TYPE=HIP"
configure_extra_options+=" -DPIKA_WITH_HIP=ON"
configure_extra_options+=" -DPIKA_WITH_CXX_STANDARD=${cxx_std}"
configure_extra_options+=" -DPIKA_WITH_MALLOC=system"
