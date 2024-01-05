# Copyright (c) 2023 ETH Zurich
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

function submit_logstash {
    echo Submitting to logstash:
    jq . "${1}"

    curl \
        --request POST \
        --header "Content-Type: application/json" \
        --data "@${1}" \
        "${CSCS_LOGSTASH_URL}"
}

function json_merge {
    # Merge json files according to
    # https://stackoverflow.com/questions/19529688/how-to-merge-2-json-objects-from-2-files-using-jq
    #
    # --slurp adds all the objects from different files into an array, and add merges the objects
    # --sort-keys is used only to always have the keys in the same order
    echo $(jq --slurp --sort-keys add "${1}" "${2}") >"${3}"
}

function json_add_value {
    file=${1}
    key=${2}
    value=${3}

    jq --arg value "${value}" ".${key} += \$value" "${file}" | sponge "${file}"
}

function json_add_value_json {
    file=${1}
    key=${2}
    value=${3}

    jq --argjson value "${value}" ".${key} += \$value" "${file}" | sponge "${file}"
}

function json_add_from_env {
    file=${1}
    key=${2}

    for var in ${@:3}; do
        jq --arg value "${!var:-}" ".${key}.${var} += \$value" "${file}" | sponge "${file}"
    done
}

function json_add_from_command {
    file=${1}
    key=${2}

    for cmd in ${@:3}; do
        jq --arg value "$(${cmd})" ".${key}.${cmd} += \$value" "${file}" | sponge "${file}"
    done
}
