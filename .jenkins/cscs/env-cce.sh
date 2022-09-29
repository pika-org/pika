# Copyright (c) 2020 ETH Zurich
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cxx_std="17"
cce_version="12.0.3"
boost_version="1.78.0"
hwloc_version="2.0.3"
spack_compiler="cce@${cce_version}"
spack_arch="cray-cnl7-broadwell"

spack_spec="pika@main arch=${spack_arch} %${spack_compiler} malloc=system cxxstd=${cxx_std} ^boost@${boost_version} ^cray-mpich ^hwloc@${hwloc_version}"

configure_extra_options+=" -DPIKA_WITH_CXX_STANDARD=${cxx_std}"
configure_extra_options+=" -DPIKA_WITH_MAX_CPU_COUNT=128"
configure_extra_options+=" -DPIKA_WITH_MALLOC=system"
configure_extra_options+=" -DPIKA_WITH_MPI=ON"
configure_extra_options+=" -DMPIEXEC_EXECUTABLE=$(which srun)"
configure_extra_options+=" -DPIKA_WITH_SPINLOCK_DEADLOCK_DETECTION=ON"

export MPICH_MAX_THREAD_SAFETY=multiple
