# Copyright (c) 2020 ETH Zurich
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cxx_std="17"
clang_version="12.0.1"
boost_version="1.79.0"
hwloc_version="2.2.0"
spack_compiler="clang@${clang_version}"
spack_arch="cray-cnl7-broadwell"

spack_spec="pika@main arch=${spack_arch} %${spack_compiler} malloc=system cxxstd=${cxx_std} ^boost@${boost_version} ^hwloc@${hwloc_version}"

configure_extra_options+=" -DPIKA_WITH_CXX_STANDARD=${cxx_std}"
configure_extra_options+=" -DPIKA_WITH_MALLOC=system"
configure_extra_options+=" -DPIKA_WITH_SPINLOCK_DEADLOCK_DETECTION=ON"
configure_extra_options+=" -DPIKA_WITH_UNITY_BUILD=ON"
configure_extra_options+=" -DPIKA_WITH_COROUTINE_COUNTERS=ON"
configure_extra_options+=" -DPIKA_WITH_THREAD_IDLE_RATES=ON"
configure_extra_options+=" -DPIKA_WITH_THREAD_CREATION_AND_CLEANUP_RATES=ON"
configure_extra_options+=" -DPIKA_WITH_THREAD_CUMULATIVE_COUNTS=ON"
configure_extra_options+=" -DPIKA_WITH_THREAD_QUEUE_WAITTIME=ON"
configure_extra_options+=" -DPIKA_WITH_THREAD_STEALING_COUNTS=ON"
