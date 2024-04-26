#!/usr/bin/env bash

# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# iterate over all files in the repo and reformat those that must be checked

for file in $(git ls-files | grep -E "\.(cpp|hpp|cu)(\.in)?$"); do
    # to allow for per-directory clang format files, we cd into the dir first
    DIR=$(dirname "$file")
    pushd ${DIR} >/dev/null
    clang-format-16 -i $(basename -- ${file})
    popd >/dev/null
done
