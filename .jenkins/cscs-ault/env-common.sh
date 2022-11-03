# Copyright (c) 2021 ETH Zurich
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


export SPACK_ROOT="/apps/ault/SSD/pika/spack"
set +u
test -z "$SPACK_USER_CONFIG_PATH" && export SPACK_USER_CONFIG_PATH="${SPACK_ROOT}/../spack-user-config"
test -z "$SPACK_USER_CACHE_PATH" && export SPACK_USER_CACHE_PATH="${SCRATCH}/spack-user-cache-jenkins/pika-cache"
set -u
source "${SPACK_ROOT}/share/spack/setup-env.sh"

spack load ccache@4.5.1 %gcc@10.2.0

export CMAKE_CXX_COMPILER_LAUNCHER=ccache
export CMAKE_GENERATOR=Ninja
export CCACHE_DIR="$SCRATCH/ccache/ccache-jenkins-pika"
export CCACHE_MAXSIZE=100G
export CCACHE_MAXFILES=50000
export CCACHE_COMPILERCHECK="%compiler% -v"

configure_extra_options+=" -DCMAKE_BUILD_TYPE=${build_type}"
configure_extra_options+=" -DPIKA_WITH_COMPILER_WARNINGS=ON"
configure_extra_options+=" -DPIKA_WITH_COMPILER_WARNINGS_AS_ERRORS=ON"
configure_extra_options+=" -DPIKA_WITH_CHECK_MODULE_DEPENDENCIES=ON"
configure_extra_options+=" -DPIKA_WITH_EXAMPLES=ON"
configure_extra_options+=" -DPIKA_WITH_TESTS=ON"
