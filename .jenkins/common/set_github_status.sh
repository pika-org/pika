#!/bin/bash -l

# Copyright (c) 2020 ETH Zurich
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set -eux

github_token=${1}
commit_repo=${2}
commit_sha=${3}
commit_status=${4}
configuration_name=${5}
build_id=${6}
context=${7}

if [[ "${commit_status}" == "success" ]]; then
    status_text="Passed"
elif [[ "${commit_status}" == "failure" ]]; then
    status_text="Failed"
elif [[ "${commit_status}" == "pending" ]]; then
    status_text="Running"
elif [[ "${commit_status}" == "skipped" ]]; then
    status_text="Skipped"
    commit_status="success"
fi

if [[ "${build_id}" == "0" ]]; then
    url="https://cdash.cscs.ch/index.php?project=pika"
else
    url="https://cdash.cscs.ch/buildSummary.php?buildid=${build_id}"
fi

curl --verbose \
    --request POST \
    --url "https://api.github.com/repos/${commit_repo}/statuses/${commit_sha}" \
    --header 'Content-Type: application/json' \
    --header "authorization: Bearer ${github_token}" \
    --data "{ \"state\": \"${commit_status}\", \"target_url\": \"${url}\", \"description\": \"${status_text}\", \"context\": \"${context}/${configuration_name}\" }"
