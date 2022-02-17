# Copyright (c) 2020 ETH Zurich
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
set -eu

module load daint-gpu spack-config

export SPACK_USER_CONFIG_PATH="${SPACK_ROOT}/../spack-user-config"
export SPACK_USER_CACHE_PATH="/scratch/snx3000/simbergm/spack-user-cache-jenkins"
source "${SPACK_ROOT}/share/spack/setup-env.sh"

spack load ccache@4.5.1 %gcc@10.3.0
spack load cmake@3.18.6 %gcc@10.3.0
spack load ninja@1.10.0 %gcc@10.3.0

export CMAKE_CXX_COMPILER_LAUNCHER=ccache
export CMAKE_GENERATOR=Ninja
export CCACHE_DIR=/scratch/snx3000/simbergm/ccache-jenkins-pika
export CCACHE_MAXSIZE=100G
export CCACHE_MAXFILES=50000

export CRAYPE_LINK_TYPE=dynamic
export APPS_ROOT="/apps/daint/SSL/pika/packages"
export CLANG_VER="11.0.0"
export CXX_STD="17"
export BOOST_VER="1.74.0"
export HWLOC_VER="2.2.0"
export CLANG_ROOT="${APPS_ROOT}/llvm-${CLANG_VER}"
export BOOST_ROOT="${APPS_ROOT}/boost-${BOOST_VER}-clang-${CLANG_VER}-c++${CXX_STD}-release"
export HWLOC_ROOT="${APPS_ROOT}/hwloc-${HWLOC_VER}-gcc-10.2.0"
export CXXFLAGS="-Wno-unused-command-line-argument -stdlib=libc++ -nostdinc++ -I${CLANG_ROOT}/include/c++/v1 -L${CLANG_ROOT}/lib -Wl,-rpath,${CLANG_ROOT}/lib,-lsupc++"
export LDCXXFLAGS="-stdlib=libc++ -L${CLANG_ROOT}/lib -Wl,-rpath,${CLANG_ROOT}/lib,-lsupc++"
export CXX="${CLANG_ROOT}/bin/clang++"
export CC="${CLANG_ROOT}/bin/clang"
export CPP="${CLANG_ROOT}/bin/clang -E"

module load cray-jemalloc/5.1.0.3
