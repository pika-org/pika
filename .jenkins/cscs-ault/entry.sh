#!/bin/bash -l

# Copyright (c) 2020 ETH Zurich
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Make undefined variables errors, print each command
set -eux

export configuration_name_with_build_type="${configuration_name}-${build_type,,}"

source ".jenkins/cscs-ault/slurm-constraint-${configuration_name}.sh"

if [[ -z "${ghprbPullId:-}" ]]; then
    # Set name of branch if not building a pull request
    export git_local_branch=$(echo "${GIT_BRANCH}" | cut -f2 -d'/')
    # Set first line of commit message to have better build names on staging and
    # trying
    export git_commit_message=$(git log --format=%B -n 1 HEAD | head -n1)
    export job_name="jenkins-pika-${git_local_branch}-${configuration_name_with_build_type}"
else
    export job_name="jenkins-pika-${ghprbPullId}-${configuration_name_with_build_type}"

    # Cancel currently running builds on the same branch, but only for pull
    # requests
    scancel --account="djenkssl" --jobname="${job_name}"
fi

# Clean up old artifacts
rm -rf "./${job_name}*" ./*-Testing

# Start the actual build
set +e
sbatch \
    --job-name="${job_name}" \
    --nodes="1" \
    --partition="${configuration_slurm_partition}" \
    --account="djenkssl" \
    --time="00:30:00" \
    --output="${job_name}.out" \
    --error="${job_name}.err" \
    --exclusive \
    --wait .jenkins/cscs-ault/batch.sh

# Print slurm logs
echo "= stdout =================================================="
cat "${job_name}.out"

echo "= stderr =================================================="
cat "${job_name}.err"

# Get build status
status_file="${job_name}-ctest-status.txt"
if [[ -f "${status_file}" && $(cat "${status_file}") -eq "0" ]]; then
    github_commit_status="success"
else
    github_commit_status="failure"
fi

# Get the CDash dashboard build id
cdash_build_id=$(cat "${job_name}-cdash-build-id.txt")

if [[ -z "${ghprbPullId:-}" ]]; then
    .jenkins/common/set_github_status.sh \
        "${GITHUB_TOKEN}" \
        "pika-org/pika" \
        "${GIT_COMMIT}" \
        "${github_commit_status}" \
        "${configuration_name_with_build_type}" \
        "${cdash_build_id}" \
        "jenkins/cscs-ault"
else
    # Extract just the organization and repo names "org/repo" from the full URL
    github_commit_repo=$(echo "$ghprbPullLink" | sed -n 's/https:\/\/github.com\/\(.*\)\/pull\/[0-9]*/\1/p')

    # Set GitHub status with CDash url
    .jenkins/common/set_github_status.sh \
        "${GITHUB_TOKEN}" \
        "${github_commit_repo}" \
        "${ghprbActualCommit}" \
        "${github_commit_status}" \
        "${configuration_name_with_build_type}" \
        "${cdash_build_id}" \
        "jenkins/cscs-ault"
fi

set -e
exit $(cat "${status_file}")
