#!/bin/bash -l

# Copyright (c) 2020 ETH Zurich
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set -ex

pika_targets=("task_overhead_report_test")
pika_test_options=(
    "--pika:ini=pika.thread_queue.init_threads_count=100 \
    --pika:scheduler=local-priority --pika:threads=4 \
    --repetitions=100 --tasks=500000")

# Build binaries for performance tests
${perftests_dir}/driver.py -v -l $logfile build -b release -o build \
    --source-dir ${src_dir} --build-dir "${build_dir}" \
    -t "${pika_targets[@]}" ||
    {
        echo 'Build failed'
        configure_build_errors=1
        exit 1
    }

index=0
result_files=""

# Run and compare for each targets specified
for executable in "${pika_targets[@]}"; do
    test_opts=${pika_test_options[$index]}
    result="${build_dir}/reports/${executable}.json"
    reference="${perftests_dir}/perftest/references/daint_default/${executable}.json"
    result_files+=("${result}")
    references_files+=("${reference}")
    logfile_tmp=log_perftests_${executable}.tmp

    run_command=("./bin/${executable} ${test_opts}")

    # Run performance tests. This is run through srun so that the CPU frequency
    # can be controlled. Low significantly slows down the CPU but should reduce
    # variations due to frequency scaling.  For more details see
    # https://slurm.schedmd.com/srun.html#OPT_cpu-freq.
    srun --cpu-freq=Low "${perftests_dir}/driver.py" -v -l "$logfile_tmp" perftest run --local True \
        --run_output "$result" --targets-and-opts "${run_command[@]}" ||
        {
            echo 'Running failed'
            test_errors=1
            exit 1
        }

    index=$((index + 1))
done

# Plot comparison of current result with references
${perftests_dir}/driver.py -v -l "$logfile" perftest plot compare --references \
    ${references_files[@]} --results ${result_files[@]} \
    -o "${build_dir}/reports/reference-comparison" ||
    {
        echo 'Plotting failed: performance drop or unknown'
        plot_errors=1
        exit 1
    }
