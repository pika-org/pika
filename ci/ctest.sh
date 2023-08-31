#!/bin/bash

#
# Distributed Linear Algebra with Future (DLAF)
#
# Copyright (c) 2018-2023, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

IMAGE="$1"
THREADS_PER_NODE="$2"
SLURM_CONSTRAINT="$3"

BASE_TEMPLATE="
include:
  - remote: 'https://gitlab.com/cscs-ci/recipes/-/raw/master/templates/v2/.ci-ext.yml'

image: $IMAGE

stages:
  - test

variables:
  SLURM_EXCLUSIVE: ''
  SLURM_EXACT: ''
  SLURM_CONSTRAINT: $SLURM_CONSTRAINT

{{JOBS}}
"

JOB_TEMPLATE="
{{LABEL}}:
  stage: test
  variables:
    SLURM_CPUS_PER_TASK: {{CPUS_PER_TASK}}
    SLURM_NTASKS: {{NTASKS}}
    SLURM_TIMELIMIT: '15:00'
    PULL_IMAGE: 'YES'
    DISABLE_AFTER_SCRIPT: 'YES'
  script: ctest -R {{LABEL}}"

JOBS=""

label='tests.unit'
#for label in `ctest --print-labels | egrep -o "RANK_[1-9][0-9]?"`; do
    N=1
    C=$(( THREADS_PER_NODE / N ))

    JOB=`echo "$JOB_TEMPLATE" | sed "s|{{LABEL}}|$label|g" \
                              | sed "s|{{NTASKS}}|$N|g" \
                              | sed "s|{{CPUS_PER_TASK}}|$C|g"`

    JOBS="$JOBS$JOB"
#done

echo "${BASE_TEMPLATE/'{{JOBS}}'/$JOBS}"
