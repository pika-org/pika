# Copyright (c) 2021 ETH Zurich
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cxx_std="20"
gcc_version="13.1.0"
boost_version="1.82.0"
hwloc_version="2.9.1"
stdexec_version="7a47a4aa411c1ca9adfcb152c28cc3dd7b156b4d"
spack_compiler="gcc@${gcc_version}"
spack_arch="cray-cnl7-broadwell"

spack_spec="pika@main arch=${spack_arch} %${spack_compiler} malloc=system cxxstd=${cxx_std} +stdexec ^boost@${boost_version} ^hwloc@${hwloc_version} ^stdexec@${stdexec_version}"

configure_extra_options+=" -DPIKA_WITH_CXX_STANDARD=${cxx_std}"
configure_extra_options+=" -DPIKA_WITH_MALLOC=system"
configure_extra_options+=" -DPIKA_WITH_STDEXEC=ON"
configure_extra_options+=" -DPIKA_WITH_SPINLOCK_DEADLOCK_DETECTION=ON"

# In release mode GCC 13 emits a false-positive array-bounds warning so we
# disable it. See https://github.com/fmtlib/fmt/issues/3354 and
# https://gcc.gnu.org/bugzilla/show_bug.cgi?id=107852.
if [[ "${build_type}" == "Release" ]]; then
    configure_extra_options+=" -DCMAKE_CXX_FLAGS=-Wno-error=array-bounds"
fi
