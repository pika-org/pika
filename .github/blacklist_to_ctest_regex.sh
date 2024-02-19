#!/usr/bin/env bash
#
# Copyright (c) 2024 ETH Zurich
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Translate a list of tests in the form of one test per line to a ctest regex that matches each line
# exactly. That is, the input is transformed:
# - to escape . to be matched literally
# - to add ^ and $ to match the whole line
# - to join the lines with |

set -euo pipefail

cat "${1}" | sed 's|\.|\\.|g' | sed 's|^|\^|' | sed 's|$|\$|' | paste -s -d '|' -
