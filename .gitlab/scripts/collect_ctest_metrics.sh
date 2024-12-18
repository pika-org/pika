#!/usr/bin/env bash

# Copyright (c) 2023 ETH Zurich
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set -euo pipefail

ctest_xml="${1}"
current_dir=$(dirname -- "${BASH_SOURCE[0]}")

source "${current_dir}/utilities.sh"

metadata_file=$(mktemp --tmpdir metadata.XXXXXXXXXX.json)
create_metadata_file "${metadata_file}"

# Submit individual test data
result_file_all=$(mktemp --tmpdir "ctest.XXXXXXXXXX.json")
num_tests=$(xq . "${ctest_xml}" | jq '.testsuite.testcase | length')
for i in $(seq 1 ${num_tests}); do
    result_file=$(mktemp --tmpdir "ctest_${i}.XXXXXXXXXX.json")
    echo '{}' >"${result_file}"

    # Extract a single test case from the XML output and convert the time from a string to a number.
    ctest_object=$(xq . "${ctest_xml}" | jq ".testsuite.testcase[$((i - 1))]" | jq '."@time" |= tonumber')
    json_add_value_json "${result_file}" "ctest.testcase" "${ctest_object}"

    json_merge "${metadata_file}" "${result_file}" "${result_file}"
    cat "${result_file}" >>"${result_file_all}"
done
submit_logstash "${result_file_all}"

result_file=$(mktemp --tmpdir "ctest.XXXXXXXXXX.json")
echo '{}' >"${result_file}"

# Submit overall ctest data. Convert numeric fields from strings to numbers.
ctest_object=$(
    xq . "${ctest_xml}" |
        jq 'del(.testsuite.testcase)' |
        jq '.testsuite."@disabled" |= tonumber' |
        jq '.testsuite."@failures" |= tonumber' |
        jq '.testsuite."@skipped" |= tonumber' |
        jq '.testsuite."@tests" |= tonumber' |
        jq '.testsuite."@time" |= tonumber'
)
json_add_value_json "${result_file}" "ctest" "${ctest_object}"

json_merge "${metadata_file}" "${result_file}" "${result_file}"
submit_logstash "${result_file}"
