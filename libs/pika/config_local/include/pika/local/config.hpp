//  Copyright (c) 2007-2018 Hartmut Kaiser
//  Copyright (c) 2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config/defines.hpp>

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
// On Windows, make sure winsock.h is not included even if windows.h is
// included before winsock2.h
#define _WINSOCKAPI_
#include <winsock2.h>
#endif

#include <pika/local/config/attributes.hpp>
#include <pika/local/config/branch_hints.hpp>
#include <pika/local/config/compiler_fence.hpp>
#include <pika/local/config/compiler_specific.hpp>
#include <pika/local/config/constexpr.hpp>
#include <pika/local/config/debug.hpp>
#include <pika/local/config/deprecation.hpp>
#include <pika/local/config/emulate_deleted.hpp>
#include <pika/local/config/export_definitions.hpp>
#include <pika/local/config/forceinline.hpp>
#include <pika/local/config/forward.hpp>
#include <pika/local/config/lambda_capture_this.hpp>
#include <pika/local/config/manual_profiling.hpp>
#include <pika/local/config/modules_enabled.hpp>
#include <pika/local/config/move.hpp>
#include <pika/local/config/threads_stack.hpp>
#include <pika/local/config/version.hpp>

#include <boost/version.hpp>

#if BOOST_VERSION < 107100
// Please update your Boost installation (see www.boost.org for details).
#error pika cannot be compiled with a Boost version earlier than 1.71.0
#endif

#include <pika/preprocessor/cat.hpp>
#include <pika/preprocessor/stringize.hpp>

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
// On Windows, make sure winsock.h is not included even if windows.h is
// included before winsock2.h
#define _WINSOCKAPI_
#endif

// clang-format off

///////////////////////////////////////////////////////////////////////////////
/// This defines the maximum number of possible runtime instances in one
/// executable
#if !defined(PIKA_RUNTIME_INSTANCE_LIMIT)
#  define PIKA_RUNTIME_INSTANCE_LIMIT 1
#endif

///////////////////////////////////////////////////////////////////////////////
/// By default, enable minimal thread deadlock detection in debug builds only.
#if !defined(PIKA_SPINLOCK_DEADLOCK_DETECTION_LIMIT)
#  define PIKA_SPINLOCK_DEADLOCK_DETECTION_LIMIT 1073741823
#endif

///////////////////////////////////////////////////////////////////////////////
/// This defines the default number of coroutine heaps.
#if !defined(PIKA_COROUTINE_NUM_HEAPS)
#  define PIKA_COROUTINE_NUM_HEAPS 7
#endif

///////////////////////////////////////////////////////////////////////////////
/// By default we do not maintain stack back-traces on suspension. This is a
/// pure debugging aid to be able to see in the debugger where a suspended
/// thread got stuck.
#if defined(PIKA_HAVE_THREAD_BACKTRACE_ON_SUSPENSION) && \
  !defined(PIKA_HAVE_STACKTRACES)
#  error PIKA_HAVE_THREAD_BACKTRACE_ON_SUSPENSION requires PIKA_HAVE_STACKTRACES to be defined!
#endif

/// By default we capture only 20 levels of stack back trace on suspension
#if !defined(PIKA_HAVE_THREAD_BACKTRACE_DEPTH)
#  define PIKA_HAVE_THREAD_BACKTRACE_DEPTH 20
#endif

///////////////////////////////////////////////////////////////////////////////
//  Characters used
//    - to delimit several pika ini paths
//    - used as file extensions for shared libraries
//    - used as path delimiters
#ifdef PIKA_WINDOWS  // windows
#  define PIKA_INI_PATH_DELIMITER            ";"
#  define PIKA_SHARED_LIB_EXTENSION          ".dll"
#  define PIKA_EXECUTABLE_EXTENSION          ".exe"
#  define PIKA_PATH_DELIMITERS               "\\/"
#else                 // unix like
#  define PIKA_INI_PATH_DELIMITER            ":"
#  define PIKA_PATH_DELIMITERS               "/"
#  ifdef __APPLE__    // apple
#    define PIKA_SHARED_LIB_EXTENSION        ".dylib"
#  elif defined(PIKA_HAVE_STATIC_LINKING)
#    define PIKA_SHARED_LIB_EXTENSION        ".a"
#  else  // linux & co
#    define PIKA_SHARED_LIB_EXTENSION        ".so"
#  endif
#  define PIKA_EXECUTABLE_EXTENSION          ""
#endif

///////////////////////////////////////////////////////////////////////////////
// Count number of empty (no pika thread available) thread manager loop executions
#if !defined(PIKA_IDLE_LOOP_COUNT_MAX)
#  define PIKA_IDLE_LOOP_COUNT_MAX 200000
#endif

///////////////////////////////////////////////////////////////////////////////
// Count number of busy thread manager loop executions before forcefully
// cleaning up terminated thread objects
#if !defined(PIKA_BUSY_LOOP_COUNT_MAX)
#  define PIKA_BUSY_LOOP_COUNT_MAX 2000
#endif

///////////////////////////////////////////////////////////////////////////////
// Maximum number of threads to create in the thread queue, except when there is
// no work to do, in which case the count will be increased in steps of
// PIKA_THREAD_QUEUE_MIN_ADD_NEW_COUNT.
#if !defined(PIKA_THREAD_QUEUE_MAX_THREAD_COUNT)
#  define PIKA_THREAD_QUEUE_MAX_THREAD_COUNT 1000
#endif

///////////////////////////////////////////////////////////////////////////////
// Minimum number of pending tasks required to steal tasks.
#if !defined(PIKA_THREAD_QUEUE_MIN_TASKS_TO_STEAL_PENDING)
#  define PIKA_THREAD_QUEUE_MIN_TASKS_TO_STEAL_PENDING 0
#endif

///////////////////////////////////////////////////////////////////////////////
// Minimum number of staged tasks required to steal tasks.
#if !defined(PIKA_THREAD_QUEUE_MIN_TASKS_TO_STEAL_STAGED)
#  define PIKA_THREAD_QUEUE_MIN_TASKS_TO_STEAL_STAGED 0
#endif

///////////////////////////////////////////////////////////////////////////////
// Minimum number of staged tasks to add to work items queue.
#if !defined(PIKA_THREAD_QUEUE_MIN_ADD_NEW_COUNT)
#  define PIKA_THREAD_QUEUE_MIN_ADD_NEW_COUNT 10
#endif

///////////////////////////////////////////////////////////////////////////////
// Maximum number of staged tasks to add to work items queue.
#if !defined(PIKA_THREAD_QUEUE_MAX_ADD_NEW_COUNT)
#  define PIKA_THREAD_QUEUE_MAX_ADD_NEW_COUNT 10
#endif

///////////////////////////////////////////////////////////////////////////////
// Minimum number of terminated threads to delete in one go.
#if !defined(PIKA_THREAD_QUEUE_MIN_DELETE_COUNT)
#  define PIKA_THREAD_QUEUE_MIN_DELETE_COUNT 10
#endif

///////////////////////////////////////////////////////////////////////////////
// Maximum number of terminated threads to delete in one go.
#if !defined(PIKA_THREAD_QUEUE_MAX_DELETE_COUNT)
#  define PIKA_THREAD_QUEUE_MAX_DELETE_COUNT 1000
#endif

///////////////////////////////////////////////////////////////////////////////
// Maximum number of terminated threads to keep before cleaning them up.
#if !defined(PIKA_THREAD_QUEUE_MAX_TERMINATED_THREADS)
#  define PIKA_THREAD_QUEUE_MAX_TERMINATED_THREADS 100
#endif

///////////////////////////////////////////////////////////////////////////////
// Number of threads (of the default stack size) to pre-allocate when
// initializing a thread queue.
#if !defined(PIKA_THREAD_QUEUE_INIT_THREADS_COUNT)
#  define PIKA_THREAD_QUEUE_INIT_THREADS_COUNT 10
#endif

///////////////////////////////////////////////////////////////////////////////
// Maximum sleep time for idle backoff in milliseconds (used only if
// PIKA_HAVE_THREAD_MANAGER_IDLE_BACKOFF is defined).
#if !defined(PIKA_IDLE_BACKOFF_TIME_MAX)
#  define PIKA_IDLE_BACKOFF_TIME_MAX 1000
#endif

///////////////////////////////////////////////////////////////////////////////
// This limits how deep the internal recursion of future continuations will go
// before a new operation is re-spawned.
#if !defined(PIKA_CONTINUATION_MAX_RECURSION_DEPTH)
#  if defined(__has_feature)
#    if __has_feature(address_sanitizer)
// if we build under AddressSanitizer we set the max recursion depth to 1 to not
// run into stack overflows.
#      define PIKA_CONTINUATION_MAX_RECURSION_DEPTH 1
#    endif
#  endif
#endif

#if !defined(PIKA_CONTINUATION_MAX_RECURSION_DEPTH)
#if defined(PIKA_DEBUG)
#define PIKA_CONTINUATION_MAX_RECURSION_DEPTH 14
#else
#define PIKA_CONTINUATION_MAX_RECURSION_DEPTH 20
#endif
#endif

///////////////////////////////////////////////////////////////////////////////
// Make sure we have support for more than 64 threads for Xeon Phi
#if defined(__MIC__) && !defined(PIKA_HAVE_MORE_THAN_64_THREADS)
#  define PIKA_HAVE_MORE_THAN_64_THREADS
#endif
#if defined(__MIC__) && !defined(PIKA_HAVE_MAX_CPU_COUNT)
#  define PIKA_HAVE_MAX_CPU_COUNT 256
#endif

// clang-format on
