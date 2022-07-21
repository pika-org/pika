#!/usr/bin/env bash

# Copyright (c) 2022 ETH Zurich
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set -euo pipefail

# Set this only to avoid errors from unset variables
export build_type=Debug

# Load the environment required for spack
module purge
source env-common.sh

# Unset variables that we don't want while building dependencies as a
# non-jenkssl user
unset CMAKE_CXX_COMPILER_LAUNCHER
unset CMAKE_GENERATOR
unset SPACK_USER_CACHE_PATH

for env in env-*.sh; do
    source ${env}
    echo "Installing dependencies for spec ${spack_spec}"
    spack install --only dependencies ${spack_spec} &
done

wait
