//  Copyright (c) 2011-2014 Bryce Adelstein-Lelbach
//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2013-2014 Thomas Heller
//  Copyright (c) 2013-2014 Patricia Grubel
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "htts2.hpp"

#include <qthread/qloop.h>
#include <qthread/qthread.h>

#include <fmt/ostream.h>
#include <fmt/printf.h>

#include <chrono>
#include <cstdint>
#include <iostream>

using BaseClock = std::chrono::steady_clock;

extern "C" void stage_tasks(size_t start, size_t stop, void* payload_duration_)
{
    htts2::payload<BaseClock>(reinterpret_cast<std::uint64_t>(payload_duration_ /* = p */));
}

struct qthreads_driver : htts2::driver
{
    qthreads_driver(int argc, char** argv)
      : htts2::driver(argc, argv)
    {
    }

    void run()
    {
        setenv("QT_NUM_SHEPHERDS", std::to_string(this->osthreads_).c_str(), 1);
        setenv("QT_NUM_WORKERS_PER_SHEPHERD", "1", 1);

        qthread_initialize();

        // Cold run
        //kernel();

        // Hot run
        results_type results = kernel();
        print_results(results);
    }

private:
    using results_type = double;

    results_type kernel()
    {
        ///////////////////////////////////////////////////////////////////////

        results_type results;

        htts2::timer<BaseClock> t;

        qt_loop(0, this->tasks_ * this->osthreads_, stage_tasks,
            reinterpret_cast<void*>(this->payload_duration_));

        // w_M [nanoseconds]
        results = t.elapsed();

        return results;

        ///////////////////////////////////////////////////////////////////////
    }

    void print_results(results_type results) const
    {
        if (this->io_ == htts2::csv_with_headers)
            std::cout << "OS-threads (Independent Variable),"
                      << "Tasks per OS-thread (Control Variable) [tasks/OS-threads],"
                      << "Payload Duration (Control Variable) [nanoseconds],"
                      << "Total Walltime [nanoseconds]" << "\n";

        fmt::print(std::cout, "{},{},{},{:.14g}\n", this->osthreads_, this->tasks_,
            this->payload_duration_, results);
    }
};

int main(int argc, char** argv)
{
    qthreads_driver d(argc, argv);

    d.run();

    return 0;
}
