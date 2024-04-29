#!/usr/bin/env bash

# Copyright (c) 2024 ETH Zurich
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# This is a wrapper script used in testing which sorts the output of a given command. Sorting the
# output makes it easier to check for the correct output if the output isn't always in the same
# order. For testing we don't care about leading whitespace.

$@ | sort --ignore-leading-blanks
