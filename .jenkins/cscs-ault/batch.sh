#!/bin/bash -l

# Copyright (c) 2020 ETH Zurich
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set -eux

# Make the name of the directories unique otherwise directories are removed by
# another job
orig_src_dir="$(pwd)"
pika_dir="/dev/shm/pika"
src_dir="${pika_dir}/src_${job_name}"
build_dir="${pika_dir}/build_${job_name}"
install_dir="${pika_dir}/install_${job_name}"

# Clean up directories older than 4 days, find fails if dir does not exist
test -d ${pika_dir} && find ${pika_dir} -depth -mindepth 1 -type d -ctime +4 -exec rm -rf {} \;

rm -rf "${src_dir}" "${build_dir}"
# Copy source directory to /dev/shm for faster builds
mkdir -p "${build_dir}" "${src_dir}"
cp -r "${orig_src_dir}"/. "${src_dir}"

source "${src_dir}/.jenkins/cscs-ault/env-common.sh"
source "${src_dir}/.jenkins/cscs-ault/env-${configuration_name}.sh"

set +e
spack build-env "${spack_spec}" -- ctest \
    --verbose \
    -S "${src_dir}/.jenkins/cscs-ault/ctest.cmake" \
    -DCTEST_BUILD_EXTRA_OPTIONS="${build_extra_options:-}" \
    -DCTEST_CONFIGURE_EXTRA_OPTIONS="${configure_extra_options} -DCMAKE_INSTALL_PREFIX=${install_dir}" \
    -DCTEST_BUILD_CONFIGURATION_NAME="${configuration_name_with_build_type}" \
    -DCTEST_SOURCE_DIRECTORY="${src_dir}" \
    -DCTEST_BINARY_DIRECTORY="${build_dir}" \
    -DCTEST_JOB_NAME="${job_name}"
set -e

ctest_exit_code=$? # /!\ Should be positioned right after the ctest command
# Things went wrong by default
file_errors=1
configure_errors=1
build_errors=1
test_errors=1

# Copy the testing directory for saving as an artifact
cp -r "${build_dir}/Testing" "${orig_src_dir}/${configuration_name_with_build_type}-Testing"

if [[ -f ${build_dir}/Testing/TAG ]]; then
    file_errors=0
    tag=$(head -n 1 "${build_dir}/Testing/TAG")

    if [[ -f "${build_dir}/Testing/${tag}/Configure.xml" ]]; then
        configure_errors=$(grep '<Error>' "${build_dir}/Testing/${tag}/Configure.xml" | wc -l)
    fi

    if [[ -f "${build_dir}/Testing/${tag}/Build.xml" ]]; then
        build_errors=$(grep '<Error>' "${build_dir}/Testing/${tag}/Build.xml" | wc -l)
    fi

    if [[ -f "${build_dir}/Testing/${tag}/Test.xml" ]]; then
        test_errors=$(grep '<Test Status=\"failed\">' "${build_dir}/Testing/${tag}/Test.xml" | wc -l)
    fi
fi
ctest_status=$(( ctest_exit_code + file_errors + configure_errors + build_errors + test_errors ))

echo "${ctest_status}" > "${job_name}-ctest-status.txt"
exit $ctest_status
