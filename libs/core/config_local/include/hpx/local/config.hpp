//  Copyright (c) 2007-2018 Hartmut Kaiser
//  Copyright (c) 2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/local/config/defines.hpp>

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
// On Windows, make sure winsock.h is not included even if windows.h is
// included before winsock2.h
#define _WINSOCKAPI_
#include <winsock2.h>
#endif

#include <hpx/local/config/attributes.hpp>
#include <hpx/local/config/branch_hints.hpp>
#include <hpx/local/config/compiler_fence.hpp>
#include <hpx/local/config/compiler_specific.hpp>
#include <hpx/local/config/constexpr.hpp>
#include <hpx/local/config/debug.hpp>
#include <hpx/local/config/deprecation.hpp>
#include <hpx/local/config/emulate_deleted.hpp>
#include <hpx/local/config/export_definitions.hpp>
#include <hpx/local/config/forceinline.hpp>
#include <hpx/local/config/forward.hpp>
#include <hpx/local/config/lambda_capture_this.hpp>
#include <hpx/local/config/manual_profiling.hpp>
#include <hpx/local/config/modules_enabled.hpp>
#include <hpx/local/config/move.hpp>
#include <hpx/local/config/threads_stack.hpp>
#include <hpx/local/config/version.hpp>

#include <boost/version.hpp>

#if BOOST_VERSION < 107100
// Please update your Boost installation (see www.boost.org for details).
#error HPX cannot be compiled with a Boost version earlier than 1.71.0
#endif

#include <hpx/preprocessor/cat.hpp>
#include <hpx/preprocessor/stringize.hpp>

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
// On Windows, make sure winsock.h is not included even if windows.h is
// included before winsock2.h
#define _WINSOCKAPI_
#endif

// clang-format off

///////////////////////////////////////////////////////////////////////////////
/// This defines the maximum number of possible runtime instances in one
/// executable
#if !defined(HPX_RUNTIME_INSTANCE_LIMIT)
#  define HPX_RUNTIME_INSTANCE_LIMIT 1
#endif

///////////////////////////////////////////////////////////////////////////////
/// By default, enable minimal thread deadlock detection in debug builds only.
#if !defined(HPX_SPINLOCK_DEADLOCK_DETECTION_LIMIT)
#  define HPX_SPINLOCK_DEADLOCK_DETECTION_LIMIT 1073741823
#endif

///////////////////////////////////////////////////////////////////////////////
/// This defines the default number of coroutine heaps.
#if !defined(HPX_COROUTINE_NUM_HEAPS)
#  define HPX_COROUTINE_NUM_HEAPS 7
#endif

///////////////////////////////////////////////////////////////////////////////
/// By default we do not maintain stack back-traces on suspension. This is a
/// pure debugging aid to be able to see in the debugger where a suspended
/// thread got stuck.
#if defined(HPX_HAVE_THREAD_BACKTRACE_ON_SUSPENSION) && \
  !defined(HPX_HAVE_STACKTRACES)
#  error HPX_HAVE_THREAD_BACKTRACE_ON_SUSPENSION requires HPX_HAVE_STACKTRACES to be defined!
#endif

/// By default we capture only 20 levels of stack back trace on suspension
#if !defined(HPX_HAVE_THREAD_BACKTRACE_DEPTH)
#  define HPX_HAVE_THREAD_BACKTRACE_DEPTH 20
#endif

///////////////////////////////////////////////////////////////////////////////
//  Characters used
//    - to delimit several HPX ini paths
//    - used as file extensions for shared libraries
//    - used as path delimiters
#ifdef HPX_WINDOWS  // windows
#  define HPX_INI_PATH_DELIMITER            ";"
#  define HPX_SHARED_LIB_EXTENSION          ".dll"
#  define HPX_EXECUTABLE_EXTENSION          ".exe"
#  define HPX_PATH_DELIMITERS               "\\/"
#else                 // unix like
#  define HPX_INI_PATH_DELIMITER            ":"
#  define HPX_PATH_DELIMITERS               "/"
#  ifdef __APPLE__    // apple
#    define HPX_SHARED_LIB_EXTENSION        ".dylib"
#  elif defined(HPX_HAVE_STATIC_LINKING)
#    define HPX_SHARED_LIB_EXTENSION        ".a"
#  else  // linux & co
#    define HPX_SHARED_LIB_EXTENSION        ".so"
#  endif
#  define HPX_EXECUTABLE_EXTENSION          ""
#endif

///////////////////////////////////////////////////////////////////////////////
#if defined(HPX_PREFIX_DEFAULT) && !defined(HPX_PREFIX)
#  define HPX_PREFIX HPX_PREFIX_DEFAULT
#endif

///////////////////////////////////////////////////////////////////////////////
// Count number of empty (no HPX thread available) thread manager loop executions
#if !defined(HPX_IDLE_LOOP_COUNT_MAX)
#  define HPX_IDLE_LOOP_COUNT_MAX 200000
#endif

///////////////////////////////////////////////////////////////////////////////
// Count number of busy thread manager loop executions before forcefully
// cleaning up terminated thread objects
#if !defined(HPX_BUSY_LOOP_COUNT_MAX)
#  define HPX_BUSY_LOOP_COUNT_MAX 2000
#endif

///////////////////////////////////////////////////////////////////////////////
// Maximum number of threads to create in the thread queue, except when there is
// no work to do, in which case the count will be increased in steps of
// HPX_THREAD_QUEUE_MIN_ADD_NEW_COUNT.
#if !defined(HPX_THREAD_QUEUE_MAX_THREAD_COUNT)
#  define HPX_THREAD_QUEUE_MAX_THREAD_COUNT 1000
#endif

///////////////////////////////////////////////////////////////////////////////
// Minimum number of pending tasks required to steal tasks.
#if !defined(HPX_THREAD_QUEUE_MIN_TASKS_TO_STEAL_PENDING)
#  define HPX_THREAD_QUEUE_MIN_TASKS_TO_STEAL_PENDING 0
#endif

///////////////////////////////////////////////////////////////////////////////
// Minimum number of staged tasks required to steal tasks.
#if !defined(HPX_THREAD_QUEUE_MIN_TASKS_TO_STEAL_STAGED)
#  define HPX_THREAD_QUEUE_MIN_TASKS_TO_STEAL_STAGED 0
#endif

///////////////////////////////////////////////////////////////////////////////
// Minimum number of staged tasks to add to work items queue.
#if !defined(HPX_THREAD_QUEUE_MIN_ADD_NEW_COUNT)
#  define HPX_THREAD_QUEUE_MIN_ADD_NEW_COUNT 10
#endif

///////////////////////////////////////////////////////////////////////////////
// Maximum number of staged tasks to add to work items queue.
#if !defined(HPX_THREAD_QUEUE_MAX_ADD_NEW_COUNT)
#  define HPX_THREAD_QUEUE_MAX_ADD_NEW_COUNT 10
#endif

///////////////////////////////////////////////////////////////////////////////
// Minimum number of terminated threads to delete in one go.
#if !defined(HPX_THREAD_QUEUE_MIN_DELETE_COUNT)
#  define HPX_THREAD_QUEUE_MIN_DELETE_COUNT 10
#endif

///////////////////////////////////////////////////////////////////////////////
// Maximum number of terminated threads to delete in one go.
#if !defined(HPX_THREAD_QUEUE_MAX_DELETE_COUNT)
#  define HPX_THREAD_QUEUE_MAX_DELETE_COUNT 1000
#endif

///////////////////////////////////////////////////////////////////////////////
// Maximum number of terminated threads to keep before cleaning them up.
#if !defined(HPX_THREAD_QUEUE_MAX_TERMINATED_THREADS)
#  define HPX_THREAD_QUEUE_MAX_TERMINATED_THREADS 100
#endif

///////////////////////////////////////////////////////////////////////////////
// Number of threads (of the default stack size) to pre-allocate when
// initializing a thread queue.
#if !defined(HPX_THREAD_QUEUE_INIT_THREADS_COUNT)
#  define HPX_THREAD_QUEUE_INIT_THREADS_COUNT 10
#endif

///////////////////////////////////////////////////////////////////////////////
// Maximum sleep time for idle backoff in milliseconds (used only if
// HPX_HAVE_THREAD_MANAGER_IDLE_BACKOFF is defined).
#if !defined(HPX_IDLE_BACKOFF_TIME_MAX)
#  define HPX_IDLE_BACKOFF_TIME_MAX 1000
#endif

///////////////////////////////////////////////////////////////////////////////
// This limits how deep the internal recursion of future continuations will go
// before a new operation is re-spawned.
#if !defined(HPX_CONTINUATION_MAX_RECURSION_DEPTH)
#  if defined(__has_feature)
#    if __has_feature(address_sanitizer)
// if we build under AddressSanitizer we set the max recursion depth to 1 to not
// run into stack overflows.
#      define HPX_CONTINUATION_MAX_RECURSION_DEPTH 1
#    endif
#  endif
#endif

#if !defined(HPX_CONTINUATION_MAX_RECURSION_DEPTH)
#if defined(HPX_DEBUG)
#define HPX_CONTINUATION_MAX_RECURSION_DEPTH 14
#else
#define HPX_CONTINUATION_MAX_RECURSION_DEPTH 20
#endif
#endif

///////////////////////////////////////////////////////////////////////////////
// Make sure we have support for more than 64 threads for Xeon Phi
#if defined(__MIC__) && !defined(HPX_HAVE_MORE_THAN_64_THREADS)
#  define HPX_HAVE_MORE_THAN_64_THREADS
#endif
#if defined(__MIC__) && !defined(HPX_HAVE_MAX_CPU_COUNT)
#  define HPX_HAVE_MAX_CPU_COUNT 256
#endif

// clang-format on
