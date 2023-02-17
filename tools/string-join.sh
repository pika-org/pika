#!/usr/bin/env bash

# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# iterate over cpp/etc files and join strings that are broken
# then, rerun clang-format to rebreak the strings into good sizes

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
awkscript=$SCRIPT_DIR/string-join.awk
tmpfile="$(mktemp /tmp/XXXXXXXXX.cpp)" || exit 1

for file in $(git ls-files | grep -E "\.(cpp|hpp|cu)(\.in)?$"); do
    awk -f $awkscript ${file} >${tmpfile}
    if ! cmp -s <(git show :${file}) <(cat $tmpfile); then
        clang-format -i -style=file:$SCRIPT_DIR/../.clang-format ${tmpfile}
        mv ${tmpfile} ${file}
    fi
done
