#!/usr/bin/env bash

# Copyright (c) 2024 ETH Zurich
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set -euo pipefail

build_directory="${1}"
current_dir=$(dirname -- "${BASH_SOURCE[0]}")

source "${current_dir}/utilities.sh"

metadata_file=$(mktemp --tmpdir metadata.XXXXXXXXXX.json)
create_metadata_file "${metadata_file}"

function submit_filesizes {
    directory="${1}"
    IFS='
'
    for line in $(ls --time-style long-iso -l $(find "${directory}" -maxdepth 1 -type f)); do
        filename=$(echo "${line}" | awk '{ print $8 }')
        filesize=$(echo "${line}" | awk '{ print $5 }')

        result_file=$(mktemp --tmpdir "$(echo ${filename} | tr -C '[:alnum:]' '_').XXXXXXXXXX.json")
        echo '{}' >"${result_file}"

        json_add_value_string "${result_file}" "files.name" "${filename}"
        json_add_value_number "${result_file}" "files.size" "${filesize}"

        json_merge "${metadata_file}" "${result_file}" "${result_file}"
        submit_logstash "${result_file}"
done
}

# Submit file name and size for files under lib and bin in the build directory
cd "${build_directory}"
submit_filesizes "bin"
submit_filesizes "lib"
