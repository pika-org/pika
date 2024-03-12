#!/usr/bin/env bash
#
# Copyright (c) 2023 ETH Zurich
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set -euo pipefail

source_directory="${1}"
current_dir=$(dirname -- "${BASH_SOURCE[0]}")

source "${current_dir}/utilities.sh"

metadata_file=$(mktemp --tmpdir metadata.XXXXXXXXXX.json)
create_metadata_file "${metadata_file}"

result_file=$(mktemp --tmpdir "sloc.XXXXXXXXXX.json")
echo '{}' >"${result_file}"

# Count lines only for the core library, ignore tests and examples
sloc_object=$(sloc --format json --exclude '(examples|tests)' "${source_directory}/libs/pika" |
    # Remove file-specific line counts, leaving only the overall counts (by file type and total)
    jq 'walk(if type == "object" then del(.files) else . end)')

json_add_value_json "${result_file}" "sloc" "${sloc_object}"

json_merge "${metadata_file}" "${result_file}" "${result_file}"
submit_logstash "${result_file}"
