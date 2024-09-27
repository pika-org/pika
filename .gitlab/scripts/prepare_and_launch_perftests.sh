#!/bin/bash -l

# Copyright (c) 2020 ETH Zurich
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set -eux

#TODO: check that SOURCE_DIR and BUILD_DIR are defined in the spack build env, otherwise redefine them there

# Computes the status of the job and store the artifacts
status_computation_and_artifacts_storage() {
    ctest_exit_code=$?
    ctest_status=$((ctest_exit_code + configure_build_errors + test_errors + plot_errors))

    # TODO: build_dir and src_dir are not defined yet
    # Copy the testing directory for saving as an artifact
    cp -r "${build_dir}/Testing" "${orig_src_dir}/${configuration_name}-Testing"
    cp -r "${build_dir}/reports" "${orig_src_dir}/${configuration_name}-reports"

    echo "${ctest_status}" >"jenkins-pika-${configuration_name}-ctest-status.txt"
    exit $ctest_status
}

trap "status_computation_and_artifacts_storage" EXIT

# TODO: check that SRC_DIR is defined, define build_dir or take the one from spack
# Variables
perftests_dir=${SRC_DIR}/tools/perftests_ci

# Things went alright by default
configure_build_errors=0
test_errors=0
plot_errors=0

source ~/venv_perftests/bin/activate
# Build and Run the perftests. We source the environment to keep the context for
# variables like configure_build_errors in launch_perftests.sh. Modifications
# written to variables in a sub-shell created with spack build-env would not be
# visible here.
set +e
spack build-env --dump env.txt $SPACK_SPEC && source env.txt    # To keep context of variables
set -e

source "${SRC_DIR}/.gitlab/scripts/launch_perftests.sh"

# Comment on the PR if any failures
if [[ $(cat ${status_file}) != 0 ]]; then
    pushd perftests-reports/reference-comparison

    # In the order of replacement rule in sed:
    # - Remove the image as does not display in github comments (section Details in the report)
    # - Escape double quotes for JSON compatibility
    # - Escape slashes for JSON compatibility
    report=$(cat index.html | \
        sed -e 's:<section class="grid-section"><h2>Details[-a-z0-9<>/"=\ \.]*</section>::Ig' \
            -e 's/"/\\"/g' \
            -e 's/\//\\\//g')

    curl \
      -X POST \
      -H "Authorization: token ${GITHUB_TOKEN}" \
      https://api.github.com/repos/pika-org/pika/issues/${ghprbPullId}/comments \
      -d "{\"body\": \"<details><summary>Performance test report</summary>${report}<\/details>\"}"

    popd
fi
