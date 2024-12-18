#!/bin/bash -l

# Copyright (c) 2020 ETH Zurich
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set -eux

# Computes the status of the job and store the artifacts
status_computation_and_artifacts_storage() {
    ctest_exit_code=$?
    ctest_status=$((ctest_exit_code + configure_build_errors + test_errors + plot_errors))

    # Copy the testing directory for saving as an artifact
    cp -r "${build_dir}/Testing" "${orig_src_dir}/${configuration_name}-Testing"
    cp -r "${build_dir}/reports" "${orig_src_dir}/${configuration_name}-reports"

    echo "${ctest_status}" >"jenkins-pika-${configuration_name}-ctest-status.txt"
    exit $ctest_status
}

trap "status_computation_and_artifacts_storage" EXIT

orig_src_dir="$(pwd)"
src_dir="/dev/shm/pika/src"
build_dir="/dev/shm/pika/build"

mkdir -p ${build_dir}/tools
# Copy source directory to /dev/shm for faster builds and copy the perftest
# utility in the build dir
cp -r "${orig_src_dir}" "${src_dir}" &&
    cp -r ${src_dir}/tools/perftests_ci ${build_dir}/tools &

# Variables
perftests_dir=${build_dir}/tools/perftests_ci
mkdir -p ${build_dir}/reports
logfile=${build_dir}/reports/jenkins-pika-${configuration_name}.log

# Load python packages
source /apps/daint/SSL/pika/virtual_envs/perftests_env_03_01_2023/bin/activate

# Things went alright by default
configure_build_errors=0
test_errors=0
plot_errors=0

# Synchronize after the asynchronous copy from the source dir
wait

export build_type=Release
source "${src_dir}/.jenkins/cscs/env-common.sh"
source "${src_dir}/.jenkins/cscs-perftests/env-${configuration_name}.sh"

# Build and Run the perftests. We source the environment to keep the context for
# variables like configure_build_errors in launch_perftests.sh. Modifications
# written to variables in a sub-shell created with spack build-env would not be
# visible here.
set +e
spack build-env --dump env.txt "${spack_spec}"
source env.txt
set -e
source "${src_dir}/.jenkins/cscs-perftests/launch_perftests.sh"

# Dummy ctest to upload the html report of the perftest
set +e
ctest \
    --verbose \
    -S "${src_dir}/.jenkins/cscs-perftests/ctest.cmake" \
    -DCTEST_BUILD_CONFIGURATION_NAME="${configuration_name}" \
    -DCTEST_SOURCE_DIRECTORY="${src_dir}" \
    -DCTEST_BINARY_DIRECTORY="${build_dir}"
set -e
