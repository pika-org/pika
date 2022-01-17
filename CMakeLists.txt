# Copyright (c)      2020 Mikael Simberg
# Copyright (c) 2007-2020 Hartmut Kaiser
# Copyright (c) 2011-2014 Thomas Heller
# Copyright (c) 2007-2008 Chirag Dekate
# Copyright (c)      2011 Bryce Lelbach
# Copyright (c)      2011 Vinay C Amatya
# Copyright (c)      2013 Jeroen Habraken
# Copyright (c) 2014-2016 Andreas Schaefer
# Copyright (c) 2017      Abhimanyu Rawat
# Copyright (c) 2017      Google
# Copyright (c) 2017      Taeguk Kwon
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# Overrides must go before the project() statement, otherwise they are ignored.

# ##############################################################################
# C++ overrides
# ##############################################################################
set(CMAKE_USER_MAKE_RULES_OVERRIDE_CXX
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/pika_cxx_overrides.cmake"
)

# ##############################################################################
# Build type (needs to be handled before project command below)
# ##############################################################################
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE
      "Release"
      CACHE
        STRING
        "Configuration type (one of Debug, RelWithDebInfo, Release, MinSizeRel)"
        FORCE
  )
endif()

# ##############################################################################
# project metadata
# ##############################################################################
project(pika CXX)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# ##############################################################################
# Build all of pika
# ##############################################################################
set(PIKA_VERSION_MAJOR 0)
set(PIKA_VERSION_MINOR 1)
set(PIKA_VERSION_SUBMINOR 0)
set(PIKA_VERSION_DATE 20210714)
set(PIKA_VERSION_TAG "-trunk")

set(PIKA_VERSION
    "${PIKA_VERSION_MAJOR}.${PIKA_VERSION_MINOR}.${PIKA_VERSION_SUBMINOR}"
)
set(PIKA_LIBRARY_VERSION "${PIKA_VERSION}")
set(PIKA_SOVERSION ${PIKA_VERSION_MAJOR})
set(PIKA_PACKAGE_NAME pika)
# To keep track of the pika root when other subprojects are declared
set(PIKA_SOURCE_DIR "${PROJECT_SOURCE_DIR}")
set(PIKA_BINARY_DIR "${PROJECT_BINARY_DIR}")

# ##############################################################################
# CMake configuration
# ##############################################################################
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")

include(GNUInstallDirs)
include(pika_utils)

# explicitly set certain policies
cmake_policy(VERSION 3.18)
pika_set_cmake_policy(CMP0042 NEW)
pika_set_cmake_policy(CMP0060 NEW)
pika_set_cmake_policy(CMP0074 NEW)

# We save the passed compiler flag to a special variable. This is needed for our
# build system unit tests. Some flags might influence the created symbols
# (_GLIBCXX_DEBUG i look at you)
set(CMAKE_CXX_FLAGS_SAFE ${CMAKE_CXX_FLAGS})
include(pika_check_cxx_standard)

include(CheckCXXCompilerFlag)
include(CMakeDependentOption)

# include additional macro definitions
include(pika_add_config_test)
include(pika_add_definitions)
include(pika_create_symbolic_link)

pika_force_out_of_tree_build(
  "This project requires an out-of-source-tree build. See README.rst. Clean your CMake cache and CMakeFiles if this message persists."
)

if(NOT PIKA_CMAKE_LOGLEVEL)
  set(PIKA_CMAKE_LOGLEVEL "WARN")
endif()

# print initial diagnostics
pika_info("CMake version: ${CMAKE_VERSION}")
pika_info("pika version: ${PIKA_VERSION}")

# ##############################################################################
# reset cached variables that need to be re-filled
unset(PIKA_EXPORT_TARGETS CACHE)
unset(PIKA_EXPORT_INTERNAL_TARGETS CACHE)
unset(PIKA_LIBS CACHE)
unset(PIKA_ENABLED_MODULES CACHE)

# ##############################################################################
# Set up dummy compiler flags targets
# ##############################################################################
include(pika_compiler_flags_targets)

# ##############################################################################
# Setup platform for which pika should be compiled.
#
include(pika_set_platform)
if("${PIKA_PLATFORM_UC}" STREQUAL "ANDROID")
  unset(PIKA_LIBRARY_VERSION)
  unset(PIKA_SOVERSION)
endif()

if(MSVC)
  pika_option(
    PIKA_WITH_VS_STARTUP_PROJECT STRING
    "Define the startup project for the pika solution (default: ALL_BUILD)."
    "ALL_BUILD" ADVANCED
  )
  if(PIKA_WITH_VS_STARTUP_PROJECT)
    set(VS_STARTUP_PROJECT ${PIKA_WITH_VS_STARTUP_PROJECT})
  endif()
endif()

# ##############################################################################

# ##############################################################################
# Set our build options cache variables which are customizable by users
#

pika_option(
  PIKA_WITH_DEPRECATION_WARNINGS BOOL
  "Enable warnings for deprecated facilities. (default: ON)" ON ADVANCED
)

if(PIKA_WITH_DEPRECATION_WARNINGS)
  # enable deprecation warnings globally
  pika_add_config_cond_define(PIKA_HAVE_DEPRECATION_WARNINGS 1)
endif()

# Generic build options
set(DEFAULT_MALLOC "system")
if("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
  set(DEFAULT_MALLOC "tcmalloc")
endif()

if("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
  set(PIKA_WITH_STACKOVERFLOW_DETECTION_DEFAULT OFF)
  string(TOUPPER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE_UC)
  if("${CMAKE_BUILD_TYPE_UC}" STREQUAL "DEBUG")
    set(PIKA_WITH_STACKOVERFLOW_DETECTION_DEFAULT ON)
  endif()
  pika_option(
    PIKA_WITH_STACKOVERFLOW_DETECTION
    BOOL
    "Enable stackoverflow detection for pika threads/coroutines. (default: OFF, debug: ON)"
    ${PIKA_WITH_STACKOVERFLOW_DETECTION_DEFAULT}
    ADVANCED
  )
  if(PIKA_WITH_STACKOVERFLOW_DETECTION)
    pika_add_config_define(PIKA_HAVE_STACKOVERFLOW_DETECTION)
  endif()
endif()

pika_option(
  PIKA_WITH_MALLOC
  STRING
  "Define which allocator should be linked in. Options are: system, tcmalloc, jemalloc, mimalloc, tbbmalloc, and custom (default: tcmalloc)"
  ${DEFAULT_MALLOC}
  STRINGS "system;tcmalloc;jemalloc;mimalloc;tbbmalloc;custom"
)

# On some systems jemalloc requires an explicit prefix for the API functions
# (i.e. 'malloc' is called 'je_malloc', etc.)
if(${PIKA_WITH_MALLOC} STREQUAL "jemalloc")
  if(MSVC)
    set(PIKA_WITH_JEMALLOC_PREFIX_DEFAULT "je_")
  else()
    set(PIKA_WITH_JEMALLOC_PREFIX_DEFAULT "<none>")
  endif()
  pika_option(
    PIKA_WITH_JEMALLOC_PREFIX STRING
    "Optional naming prefix for jemalloc API functions"
    ${PIKA_WITH_JEMALLOC_PREFIX_DEFAULT} ADVANCED
  )
endif()

# Logging configuration
pika_option(
  PIKA_WITH_LOGGING BOOL
  "Build pika with logging enabled (default: ON)." ON ADVANCED
)
if(PIKA_WITH_LOGGING)
  pika_add_config_define(PIKA_HAVE_LOGGING)
endif()

# Compiler related build options
pika_option(
  PIKA_WITH_GCC_VERSION_CHECK BOOL
  "Don't ignore version reported by gcc (default: ON)" ON ADVANCED
)

pika_option(
  PIKA_WITH_COMPILER_WARNINGS BOOL "Enable compiler warnings (default: ON)"
  ON ADVANCED
)

pika_option(
  PIKA_WITH_COMPILER_WARNINGS_AS_ERRORS BOOL
  "Turn compiler warnings into errors (default: OFF)" OFF ADVANCED
)

pika_option(
  PIKA_WITH_EXECUTABLE_PREFIX STRING "Executable prefix (default: none)." ""
  CATEGORY "Build Targets"
)

pika_option(
  PIKA_WITH_BINARY_DIR
  PATH
  "Root directory for libraries and executables for the build. This does not affect the installed location of libraries and executables."
  "${PROJECT_BINARY_DIR}"
  CATEGORY "Build Targets"
  ADVANCED
)

if(WIN32)
  set(PIKA_WITH_PSEUDO_DEPENDENCIES
      OFF
      CACHE INTERNAL "" FORCE
  )
else()
  set(PIKA_WITH_PSEUDO_DEPENDENCIES
      ON
      CACHE INTERNAL "" FORCE
  )
endif()

pika_option(
  PIKA_WITH_UNITY_BUILD BOOL
  "Enable unity build for certain build targets (default: ON)" ON ADVANCED
)
if(PIKA_WITH_UNITY_BUILD)
  set(PIKA_WITH_UNITY_BUILD_OPTION UNITY_BUILD)
endif()

pika_option(
  PIKA_WITH_PRECOMPILED_HEADERS
  BOOL
  "Enable precompiled headers for certain build targets (experimental) (default: ON)"
  ON
  ADVANCED
)
if(PIKA_WITH_PRECOMPILED_HEADERS)
  set(PIKA_WITH_PRECOMPILED_HEADERS_INTERNAL ON)
  # Only create the targets here. They will be set up later once all modules are
  # known.
  add_library(pika_precompiled_headers OBJECT libs/src/dummy.cpp)
  add_executable(pika_exe_precompiled_headers libs/src/dummy.cpp)

  set_target_properties(pika_precompiled_headers PROPERTIES FOLDER "Core")
  set_target_properties(
    pika_exe_precompiled_headers PROPERTIES FOLDER "Core"
  )
endif()

# ##############################################################################
pika_option(
  PIKA_WITH_EXAMPLES BOOL "Build examples (default: OFF)" OFF
  CATEGORY "Build Targets"
)
pika_option(
  PIKA_WITH_TESTS BOOL "Build tests (default: OFF)" OFF
  CATEGORY "Build Targets"
)
pika_option(
  PIKA_WITH_TESTS_BENCHMARKS BOOL "Build benchmark tests (default: ON)" ON
  CATEGORY "Build Targets"
)
pika_option(
  PIKA_WITH_TESTS_REGRESSIONS BOOL "Build regression tests (default: ON)" ON
  CATEGORY "Build Targets"
)
pika_option(
  PIKA_WITH_TESTS_UNIT BOOL "Build unit tests (default: ON)" ON
  CATEGORY "Build Targets"
)
pika_option(
  PIKA_WITH_TESTS_HEADERS BOOL "Build header tests (default: ON)" ON
  ADVANCED CATEGORY "Build Targets"
)
pika_option(
  PIKA_WITH_TESTS_EXTERNAL_BUILD BOOL
  "Build external CMake build tests (default: ON)" ON ADVANCED
  CATEGORY "Build Targets"
)
pika_option(
  PIKA_WITH_TESTS_EXAMPLES BOOL "Add examples as tests (default: ON)" ON
  ADVANCED CATEGORY "Build Targets"
)
pika_option(
  PIKA_WITH_COMPILE_ONLY_TESTS BOOL
  "Create build system support for compile time only tests (default: ON)" ON
  CATEGORY "Build Targets"
)
pika_option(
  PIKA_WITH_FAIL_COMPILE_TESTS BOOL
  "Create build system support for fail compile tests (default: ON)" ON
  CATEGORY "Build Targets"
)

# disable all tests if PIKA_WITH_TESTS=OFF
if(NOT PIKA_WITH_TESTS)
  pika_set_option(
    PIKA_WITH_TESTS_BENCHMARKS
    VALUE OFF
    FORCE
  )
  pika_set_option(
    PIKA_WITH_TESTS_REGRESSIONS
    VALUE OFF
    FORCE
  )
  pika_set_option(
    PIKA_WITH_TESTS_UNIT
    VALUE OFF
    FORCE
  )
  pika_set_option(
    PIKA_WITH_TESTS_HEADERS
    VALUE OFF
    FORCE
  )
  pika_set_option(
    PIKA_WITH_TESTS_EXTERNAL_BUILD
    VALUE OFF
    FORCE
  )
  pika_set_option(
    PIKA_WITH_TESTS_EXAMPLES
    VALUE OFF
    FORCE
  )
endif()

pika_option(
  PIKA_WITH_TOOLS BOOL "Build tools (default: OFF)" OFF ADVANCED
  CATEGORY "Build Targets"
)

set(PIKA_FULL_RPATH_DEFAULT ON)
if(APPLE OR WIN32)
  set(PIKA_FULL_RPATH_DEFAULT OFF)
endif()
pika_option(
  PIKA_WITH_FULL_RPATH
  BOOL
  "Build and link libraries and executables with full RPATHs (default: ${PIKA_FULL_RPATH_DEFAULT})"
  ${PIKA_FULL_RPATH_DEFAULT}
  ADVANCED
)

# ##############################################################################
# CUDA/HIP configuration
# ##############################################################################
set(CUDA_OPTION_STRING "Enable support for CUDA (default: OFF)")
set(HIP_OPTION_STRING "Enable support for HIP (default: OFF)")
pika_option(PIKA_WITH_CUDA BOOL "${CUDA_OPTION_STRING}" OFF ADVANCED)

# No need for the user to specify the option explicitly
pika_option(PIKA_WITH_HIP BOOL "${HIP_OPTION_STRING}" OFF ADVANCED)
if("${CMAKE_CXX_COMPILER}" MATCHES "hipcc$")
  set(PIKA_WITH_HIP
      ON
      CACHE BOOL "${HIP_OPTION_STRING}" FORCE
  )
endif()

if(PIKA_WITH_CUDA AND PIKA_WITH_HIP)
  pika_error(
    "PIKA_WITH_CUDA=ON and PIKA_WITH_HIP=ON. Only one of them can be on at the same time.\
 Note: PIKA_WITH_HIP is automatically enabled when compiling with hipcc."
  )
endif()

if(PIKA_WITH_CUDA OR PIKA_WITH_HIP)
  pika_add_config_define(PIKA_HAVE_GPU_SUPPORT)
  set(PIKA_WITH_GPU_SUPPORT ON)
  set(PIKA_WITH_GPU_SUPPORT
      ON
      CACHE INTERNAL "" FORCE
  )
else()
  set(PIKA_WITH_GPU_SUPPORT
      OFF
      CACHE INTERNAL "" FORCE
  )
endif()

# ##############################################################################
# datapar configuration
# ##############################################################################
pika_option(
  PIKA_WITH_DATAPAR_VC
  BOOL
  "Enable data parallel algorithm support using the external Vc library (default: OFF)"
  OFF
  ADVANCED
)
if(PIKA_WITH_DATAPAR_VC)
  pika_option(
    PIKA_WITH_DATAPAR_VC_NO_LIBRARY BOOL
    "Don't link with the Vc static library (default: OFF)" OFF ADVANCED
  )
endif()

if(PIKA_WITH_DATAPAR_VC)
  pika_warn(
    "Vc support is deprecated. This option will be removed in a future release. It will be replaced with SIMD support from the C++ standard library"
  )
  include(pika_setup_vc)
endif()
if(NOT PIKA_WITH_DATAPAR_VC)
  pika_info("No vectorization library configured")
else()
  pika_option(
    PIKA_WITH_DATAPAR BOOL
    "Enable data parallel algorithm support (default: ON)" ON ADVANCED
  )
endif()

# ##############################################################################
# Threadlevel Nice option
# ##############################################################################
pika_option(
  PIKA_WITH_NICE_THREADLEVEL
  BOOL
  "Set pika worker threads to have high NICE level (may impact performance) (default: OFF)"
  OFF
  ADVANCED
)
if(PIKA_WITH_NICE_THREADLEVEL)
  pika_info("Nice threadlevel is enabled.")
  pika_add_config_define(PIKA_HAVE_NICE_THREADLEVEL)
else()
  pika_info("Nice threadlevel is disabled.")
endif()

# ##############################################################################
# Utility configuration
# ##############################################################################
set(PIKA_HIDDEN_VISIBILITY_DEFAULT ON)
if(CMAKE_COMPILER_IS_GNUCXX)
  if("${PIKA_PLATFORM_UC}" STREQUAL "ANDROID")
    set(PIKA_HIDDEN_VISIBILITY_DEFAULT OFF)
  endif()
endif()
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(PIKA_HIDDEN_VISIBILITY_DEFAULT OFF)
endif()
if(APPLE)
  set(PIKA_HIDDEN_VISIBILITY_DEFAULT OFF)
endif()

pika_option(
  PIKA_WITH_HIDDEN_VISIBILITY
  BOOL
  "Use -fvisibility=hidden for builds on platforms which support it (default: ${PIKA_HIDDEN_VISIBILITY_DEFAULT})"
  ${PIKA_HIDDEN_VISIBILITY_DEFAULT}
  ADVANCED
)

pika_option(
  PIKA_WITH_AUTOMATIC_SERIALIZATION_REGISTRATION
  BOOL
  "Use automatic serialization registration for actions and functions. This affects compatibility between pika applications compiled with different compilers (default: ON)"
  ON
  ADVANCED
)
if(PIKA_WITH_AUTOMATIC_SERIALIZATION_REGISTRATION)
  pika_add_config_define(PIKA_HAVE_AUTOMATIC_SERIALIZATION_REGISTRATION)
endif()

pika_option(
  PIKA_WITH_ZERO_COPY_SERIALIZATION_THRESHOLD
  STRING
  "The threshold in bytes to when perform zero copy optimizations (default: 128)"
  "128"
  ADVANCED
)
pika_add_config_define(
  PIKA_ZERO_COPY_SERIALIZATION_THRESHOLD
  ${PIKA_WITH_ZERO_COPY_SERIALIZATION_THRESHOLD}
)

pika_option(
  PIKA_WITH_DISABLED_SIGNAL_EXCEPTION_HANDLERS
  BOOL
  "Disables the mechanism that produces debug output for caught signals and unhandled exceptions (default: OFF)"
  OFF
  ADVANCED
)
if(PIKA_WITH_DISABLED_SIGNAL_EXCEPTION_HANDLERS)
  pika_add_config_define(PIKA_HAVE_DISABLED_SIGNAL_EXCEPTION_HANDLERS)
endif()

# Thread Manager related build options

set(PIKA_MAX_CPU_COUNT_DEFAULT "")
pika_option(
  PIKA_WITH_MAX_CPU_COUNT
  STRING
  "pika applications will not use more that this number of OS-Threads (empty string means dynamic) (default: \"${PIKA_MAX_CPU_COUNT_DEFAULT}\")"
  "${PIKA_MAX_CPU_COUNT_DEFAULT}"
  CATEGORY "Thread Manager"
  ADVANCED
)
if(PIKA_WITH_MAX_CPU_COUNT)
  pika_add_config_define(
    PIKA_HAVE_MAX_CPU_COUNT ${PIKA_WITH_MAX_CPU_COUNT}
  )
endif()
if((NOT PIKA_WITH_MAX_CPU_COUNT) OR (PIKA_WITH_MAX_CPU_COUNT GREATER 64
                                        )
)
  pika_add_config_define(PIKA_HAVE_MORE_THAN_64_THREADS)
endif()

set(PIKA_MAX_NUMA_DOMAIN_COUNT_DEFAULT "8")
pika_option(
  PIKA_WITH_MAX_NUMA_DOMAIN_COUNT
  STRING
  "pika applications will not run on machines with more NUMA domains (default: ${PIKA_MAX_NUMA_DOMAIN_COUNT_DEFAULT})"
  ${PIKA_MAX_NUMA_DOMAIN_COUNT_DEFAULT}
  CATEGORY "Thread Manager"
  ADVANCED
)
pika_add_config_define(
  PIKA_HAVE_MAX_NUMA_DOMAIN_COUNT ${PIKA_WITH_MAX_NUMA_DOMAIN_COUNT}
)

pika_option(
  PIKA_WITH_THREAD_STACK_MMAP BOOL
  "Use mmap for stack allocation on appropriate platforms" ON
  CATEGORY "Thread Manager"
  ADVANCED
)

pika_option(
  PIKA_WITH_THREAD_MANAGER_IDLE_BACKOFF BOOL
  "pika scheduler threads do exponential backoff on idle queues (default: ON)" ON
  CATEGORY "Thread Manager"
  ADVANCED
)

pika_option(
  PIKA_WITH_STACKTRACES BOOL
  "Attach backtraces to pika exceptions (default: ON)" ON
  CATEGORY "Thread Manager"
  ADVANCED
)

pika_option(
  PIKA_WITH_THREAD_BACKTRACE_ON_SUSPENSION BOOL
  "Enable thread stack back trace being captured on suspension (default: OFF)"
  OFF
  CATEGORY "Thread Manager"
  ADVANCED
)

# We create a target to contain libraries like rt, dl etc. in order to remove
# global variables
add_library(pika_base_libraries INTERFACE)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  target_link_libraries(pika_base_libraries INTERFACE imf svml irng intlc)
endif()

if(PIKA_WITH_STACKTRACES OR PIKA_WITH_THREAD_BACKTRACE_ON_SUSPENSION)
  pika_info("Stack traces are enabled.")
  pika_add_config_define(PIKA_HAVE_STACKTRACES)
  if(WIN32)
    target_link_libraries(pika_base_libraries INTERFACE dbghelp)
  endif()

  pika_option(
    PIKA_WITH_THREAD_BACKTRACE_DEPTH STRING
    "Thread stack back trace depth being captured (default: 20)" "20"
    CATEGORY "Thread Manager"
    ADVANCED
  )
  pika_add_config_define(
    PIKA_HAVE_THREAD_BACKTRACE_DEPTH ${PIKA_WITH_THREAD_BACKTRACE_DEPTH}
  )

  if(("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
     AND ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang"
          OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  )

    # This option is OFF by default as we have seen random segfaults out in the
    # wild if this is enabled.
    pika_option(
      PIKA_WITH_STACKTRACES_STATIC_SYMBOLS BOOL
      "Thread stack back trace will resolve static symbols (default: OFF)" OFF
      CATEGORY "Thread Manager"
      ADVANCED
    )
    pika_add_config_define(
      PIKA_HAVE_STACKTRACES_STATIC_SYMBOLS
      ${PIKA_WITH_STACKTRACES_STATIC_SYMBOLS}
    )

    # Demangling can segfault in certain configurations.
    pika_option(
      PIKA_WITH_STACKTRACES_DEMANGLE_SYMBOLS BOOL
      "Thread stack back trace symbols will be demangled (default: ON)" ON
      CATEGORY "Thread Manager"
      ADVANCED
    )
    if(PIKA_WITH_STACKTRACES_DEMANGLE_SYMBOLS)
      pika_add_config_define(PIKA_HAVE_STACKTRACES_DEMANGLE_SYMBOLS)
    endif()
  endif()
endif()

if(PIKA_WITH_THREAD_BACKTRACE_ON_SUSPENSION)
  pika_add_config_define(PIKA_HAVE_THREAD_BACKTRACE_ON_SUSPENSION)

  pika_option(
    PIKA_WITH_THREAD_FULLBACKTRACE_ON_SUSPENSION BOOL
    "Enable thread stack back trace being captured on suspension (default: OFF)"
    OFF
    CATEGORY "Thread Manager"
    ADVANCED
  )
  if(PIKA_WITH_THREAD_FULLBACKTRACE_ON_SUSPENSION)
    pika_add_config_define(PIKA_HAVE_THREAD_FULLBACKTRACE_ON_SUSPENSION)
  endif()
endif()

pika_option(
  PIKA_WITH_THREAD_TARGET_ADDRESS BOOL
  "Enable storing target address in thread for NUMA awareness (default: OFF)"
  OFF
  CATEGORY "Thread Manager"
  ADVANCED
)

if(PIKA_WITH_THREAD_TARGET_ADDRESS)
  pika_add_config_define(PIKA_HAVE_THREAD_TARGET_ADDRESS)
endif()

pika_option(
  PIKA_WITH_THREAD_QUEUE_WAITTIME BOOL
  "Enable collecting queue wait times for threads (default: OFF)" OFF
  CATEGORY "Thread Manager"
  ADVANCED
)

if(PIKA_WITH_THREAD_QUEUE_WAITTIME)
  pika_add_config_define(PIKA_HAVE_THREAD_QUEUE_WAITTIME)
endif()

pika_option(
  PIKA_WITH_THREAD_IDLE_RATES
  BOOL
  "Enable measuring the percentage of overhead times spent in the scheduler (default: OFF)"
  OFF
  CATEGORY "Thread Manager"
  ADVANCED
)

if(PIKA_WITH_THREAD_IDLE_RATES)
  pika_option(
    PIKA_WITH_BACKGROUND_THREAD_COUNTERS
    BOOL
    "Enable performance counters related to adaptive parcel coalescing (default: OFF)."
    OFF
    ADVANCED
  )
  if(PIKA_WITH_BACKGROUND_THREAD_COUNTERS)
    pika_add_config_define(PIKA_HAVE_BACKGROUND_THREAD_COUNTERS)
  endif()
endif()

pika_option(
  PIKA_WITH_THREAD_CREATION_AND_CLEANUP_RATES BOOL
  "Enable measuring thread creation and cleanup times (default: OFF)" OFF
  CATEGORY "Thread Manager"
  ADVANCED
)

if(PIKA_WITH_THREAD_IDLE_RATES)
  pika_add_config_define(PIKA_HAVE_THREAD_IDLE_RATES)
  if(PIKA_WITH_THREAD_CREATION_AND_CLEANUP_RATES)
    pika_add_config_define(PIKA_HAVE_THREAD_CREATION_AND_CLEANUP_RATES)
  endif()
endif()

pika_option(
  PIKA_WITH_THREAD_CUMULATIVE_COUNTS
  BOOL
  "Enable keeping track of cumulative thread counts in the schedulers (default: ON)"
  ON
  CATEGORY "Thread Manager"
  ADVANCED
)

if(PIKA_WITH_THREAD_CUMULATIVE_COUNTS)
  pika_add_config_define(PIKA_HAVE_THREAD_CUMULATIVE_COUNTS)
endif()

pika_option(
  PIKA_WITH_THREAD_STEALING_COUNTS
  BOOL
  "Enable keeping track of counts of thread stealing incidents in the schedulers (default: OFF)"
  OFF
  CATEGORY "Thread Manager"
  ADVANCED
)

if(PIKA_WITH_THREAD_STEALING_COUNTS)
  pika_add_config_define(PIKA_HAVE_THREAD_STEALING_COUNTS)
endif()

pika_option(
  PIKA_WITH_COROUTINE_COUNTERS BOOL
  "Enable keeping track of coroutine creation and rebind counts (default: OFF)"
  OFF
  CATEGORY "Thread Manager"
  ADVANCED
)
if(PIKA_WITH_COROUTINE_COUNTERS)
  pika_add_config_define(PIKA_HAVE_COROUTINE_COUNTERS)
endif()

pika_option(
  PIKA_WITH_THREAD_LOCAL_STORAGE BOOL
  "Enable thread local storage for all pika threads (default: OFF)" OFF
  CATEGORY "Thread Manager"
  ADVANCED
)

if(PIKA_WITH_THREAD_LOCAL_STORAGE)
  pika_add_config_define(PIKA_HAVE_THREAD_LOCAL_STORAGE)
endif()

pika_option(
  PIKA_WITH_SCHEDULER_LOCAL_STORAGE BOOL
  "Enable scheduler local storage for all pika schedulers (default: OFF)" OFF
  CATEGORY "Thread Manager"
  ADVANCED
)

if(PIKA_WITH_SCHEDULER_LOCAL_STORAGE)
  pika_add_config_define(PIKA_HAVE_SCHEDULER_LOCAL_STORAGE)
endif()

pika_option(
  PIKA_WITH_SPINLOCK_POOL_NUM STRING
  "Number of elements a spinlock pool manages (default: 128)" 128
  CATEGORY "Thread Manager"
  ADVANCED
)

pika_add_config_define(
  PIKA_HAVE_SPINLOCK_POOL_NUM ${PIKA_WITH_SPINLOCK_POOL_NUM}
)

pika_option(
  PIKA_WITH_SPINLOCK_DEADLOCK_DETECTION BOOL
  "Enable spinlock deadlock detection (default: OFF)" OFF
  CATEGORY "Thread Manager"
  ADVANCED
)

if(PIKA_WITH_SPINLOCK_DEADLOCK_DETECTION)
  pika_add_config_define(PIKA_HAVE_SPINLOCK_DEADLOCK_DETECTION)
endif()

pika_option(
  PIKA_WITH_ITTNOTIFY BOOL "Enable Amplifier (ITT) instrumentation support."
  OFF CATEGORY "Profiling"
)

pika_option(
  PIKA_WITH_ASYNC_MPI
  BOOL
  "Enable support for returning futures from MPI asynchronous calls (default: ON)"
  OFF
  CATEGORY "MPI"
)

if(PIKA_WITH_ASYNC_MPI)
  # mpi parcelport settings
  pika_option(
    PIKA_WITH_PARCELPORT_MPI_ENV
    STRING
    "List of environment variables checked to detect MPI (default: MV2_COMM_WORLD_RANK;PMI_RANK;OMPI_COMM_WORLD_SIZE;ALPS_APP_PE;PMIX_RANK;PALS_NODEID)."
    "MV2_COMM_WORLD_RANK;PMI_RANK;OMPI_COMM_WORLD_SIZE;ALPS_APP_PE;PMIX_RANK;PALS_NODEID"
    CATEGORY "Parcelport"
    ADVANCED
  )

  # This list is to detect whether we run inside an mpi environment. If one of
  # those environment variables is set, the MPI parcelport is enabled by
  # default. PMI_RANK: Intel MPI and MVAPICH2 OMPI_COMM_WORLD_SIZE: OpenMPI
  # starting at version 1.3
  if(PIKA_WITH_PARCELPORT_MPI_ENV)
    string(REPLACE ";" "," pika_parcelport_mpi_env_
                   "${PIKA_WITH_PARCELPORT_MPI_ENV}"
    )
    pika_add_config_define(
      PIKA_HAVE_PARCELPORT_MPI_ENV "\"${pika_parcelport_mpi_env_}\""
    )
  endif()

  pika_option(
    PIKA_WITH_PARCELPORT_MPI_MULTITHREADED BOOL
    "Turn on MPI multithreading support (default: ON)." ON
    CATEGORY "Parcelport"
    ADVANCED
  )
  if(PIKA_WITH_PARCELPORT_MPI_MULTITHREADED)
    pika_add_config_define(PIKA_HAVE_PARCELPORT_MPI_MULTITHREADED)
  endif()

  if(MSVC)
    # FIXME: add OpenMPI specific flag here for now as the
    # pika_add_compile_flag() below does not add the extra options to the
    # top level directory
    pika_add_config_define(OMPI_IMPORTS)
  endif()
endif()

# External libraries/frameworks used by sme of the examples and benchmarks
pika_option(
  PIKA_WITH_EXAMPLES_OPENMP BOOL
  "Enable examples requiring OpenMP support (default: OFF)." OFF
  CATEGORY "Build Targets"
  ADVANCED
)
if(PIKA_WITH_EXAMPLES_OPENMP)
  find_package(OpenMP REQUIRED)
endif()
pika_option(
  PIKA_WITH_EXAMPLES_TBB BOOL
  "Enable examples requiring TBB support (default: OFF)." OFF
  CATEGORY "Build Targets"
  ADVANCED
)
if(PIKA_WITH_EXAMPLES_TBB)
  find_package(TBB)
  if(NOT TBB_FOUND)
    set(PIKA_WITH_EXAMPLES_TBB OFF)
  endif()
endif()
pika_option(
  PIKA_WITH_EXAMPLES_QTHREADS BOOL
  "Enable examples requiring QThreads support (default: OFF)." OFF
  CATEGORY "Build Targets"
  ADVANCED
)
if(PIKA_WITH_EXAMPLES_QTHREADS)
  find_package(QThreads)
  if(NOT QTHREADS_FOUND)
    set(PIKA_WITH_EXAMPLES_QTHREADS OFF)
  endif()
endif()
pika_option(
  PIKA_WITH_EXAMPLES_HDF5 BOOL
  "Enable examples requiring HDF5 support (default: OFF)." OFF
  CATEGORY "Build Targets"
  ADVANCED
)
if(PIKA_WITH_EXAMPLES_HDF5)
  find_package(HDF5 COMPONENTS CXX)
  if(NOT HDF5_FOUND)
    set(PIKA_WITH_EXAMPLES_HDF5 OFF)
  endif()
endif()

# Disabling the Qt example on BG/Q as GUIs don't make sense there anyways
if(NOT "${PIKA_PLATFORM_UC}" STREQUAL "BLUEGENEQ")
  pika_option(
    PIKA_WITH_EXAMPLES_QT4 BOOL
    "Enable examples requiring Qt4 support (default: OFF)." OFF
    CATEGORY "Build Targets"
    ADVANCED
  )
  if(PIKA_WITH_EXAMPLES_QT4)
    find_package(Qt4)
    if(NOT QT4_FOUND)
      set(PIKA_WITH_EXAMPLES_QT4 OFF)
    endif()
  endif()
endif()

# Debugging related build options
pika_option(
  PIKA_WITH_VALGRIND BOOL "Enable Valgrind instrumentation support." OFF
  CATEGORY "Debugging"
)

pika_option(
  PIKA_WITH_SANITIZERS BOOL
  "Configure with sanitizer instrumentation support." OFF CATEGORY "Debugging"
)

set(PIKA_WITH_VERIFY_LOCKS_DEFAULT OFF)
if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  set(PIKA_WITH_VERIFY_LOCKS_DEFAULT ON)
endif()

pika_option(
  PIKA_WITH_VERIFY_LOCKS BOOL
  "Enable lock verification code (default: OFF, enabled in debug builds)"
  ${PIKA_WITH_VERIFY_LOCKS_DEFAULT}
  CATEGORY "Debugging"
  ADVANCED
)
pika_option(
  PIKA_WITH_VERIFY_LOCKS_BACKTRACE
  BOOL
  "Enable thread stack back trace being captured on lock registration (to be used in combination with PIKA_WITH_VERIFY_LOCKS=ON, default: OFF)"
  OFF
  CATEGORY "Debugging"
  ADVANCED
)
pika_option(
  PIKA_WITH_THREAD_DEBUG_INFO
  BOOL
  "Enable thread debugging information (default: OFF, implicitly enabled in debug builds)"
  OFF
  CATEGORY "Debugging"
  ADVANCED
)
pika_option(
  PIKA_WITH_THREAD_GUARD_PAGE BOOL "Enable thread guard page (default: ON)"
  ON
  CATEGORY "Debugging"
  ADVANCED
)

if(PIKA_WITH_VERIFY_LOCKS)
  pika_add_config_define(PIKA_HAVE_VERIFY_LOCKS)
  if(PIKA_WITH_VERIFY_LOCKS_BACKTRACE)
    pika_add_config_define(PIKA_HAVE_VERIFY_LOCKS_BACKTRACE)
  endif()
endif()

# Additional debug support
if(NOT WIN32 AND PIKA_WITH_THREAD_GUARD_PAGE)
  pika_add_config_define(PIKA_HAVE_THREAD_GUARD_PAGE)
endif()

if(NOT WIN32 AND PIKA_WITH_THREAD_STACK_MMAP)
  pika_add_config_define(PIKA_HAVE_THREAD_STACK_MMAP)
endif()

if(PIKA_WITH_THREAD_MANAGER_IDLE_BACKOFF)
  pika_add_config_define(PIKA_HAVE_THREAD_MANAGER_IDLE_BACKOFF)
endif()

pika_option(
  PIKA_WITH_THREAD_DESCRIPTION_FULL BOOL
  "Use function address for thread description (default: OFF)" OFF
  CATEGORY "Debugging"
  ADVANCED
)

pika_option(
  PIKA_WITH_ATTACH_DEBUGGER_ON_TEST_FAILURE BOOL
  "Break the debugger if a test has failed  (default: OFF)" OFF
  CATEGORY "Debugging"
  ADVANCED
)
if(PIKA_WITH_ATTACH_DEBUGGER_ON_TEST_FAILURE)
  pika_add_config_define(PIKA_HAVE_ATTACH_DEBUGGER_ON_TEST_FAILURE)
endif()

pika_option(
  PIKA_WITH_TESTS_DEBUG_LOG BOOL
  "Turn on debug logs (--pika:debug-pika-log) for tests (default: OFF)" OFF
  CATEGORY "Debugging"
  ADVANCED
)

pika_option(
  PIKA_WITH_TESTS_DEBUG_LOG_DESTINATION STRING
  "Destination for test debug logs (default: cout)" "cout"
  CATEGORY "Debugging"
  ADVANCED
)

pika_option(
  PIKA_WITH_TESTS_MAX_THREADS_PER_LOCALITY
  STRING
  "Maximum number of threads to use for tests (default: 0, use the number of threads specified by the test)"
  0
  CATEGORY "Debugging"
  ADVANCED
)

pika_option(
  PIKA_WITH_PARALLEL_TESTS_BIND_NONE
  BOOL
  "Pass --pika:bind=none to tests that may run in parallel (cmake -j flag) (default: OFF)"
  OFF
  CATEGORY "Debugging"
  ADVANCED
)

# If APEX is defined, the action timers need thread debug info.
pika_option(
  PIKA_WITH_APEX BOOL "Enable APEX instrumentation support." OFF
  CATEGORY "Profiling"
)
if(PIKA_WITH_APEX)
  pika_add_config_define(PIKA_HAVE_APEX)
  pika_add_config_define(PIKA_HAVE_THREAD_DESCRIPTION)
  pika_add_config_define(PIKA_HAVE_THREAD_PARENT_REFERENCE)
  if(PIKA_WITH_THREAD_DESCRIPTION_FULL)
    pika_add_config_define(PIKA_HAVE_THREAD_DESCRIPTION_FULL)
  endif()
endif()

if(PIKA_WITH_THREAD_DEBUG_INFO)
  pika_add_config_define(PIKA_HAVE_THREAD_PARENT_REFERENCE)
  pika_add_config_define(PIKA_HAVE_THREAD_PHASE_INFORMATION)
  pika_add_config_define(PIKA_HAVE_THREAD_DESCRIPTION)
  pika_add_config_define(PIKA_HAVE_THREAD_DEADLOCK_DETECTION)
  if(PIKA_WITH_THREAD_DESCRIPTION_FULL)
    pika_add_config_define(PIKA_HAVE_THREAD_DESCRIPTION_FULL)
  endif()
endif()

# Developer tools
pika_option(
  PIKA_WITH_VIM_YCM BOOL
  "Generate completion file for VIM YouCompleteMe plugin" OFF ADVANCED
)

# ##############################################################################
# Backwards compatibility options (edit for each release)

# BADBAD: This enables an overload of swap which is necessary to work around the
# problems caused by zip_iterator not being a real random access iterator.
# Dereferencing zip_iterator does not yield a true reference but only a
# temporary tuple holding true references.
#
# A real fix for this problem is proposed in PR0022R0
# (http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2015/p0022r0.html)
#
pika_option(
  PIKA_WITH_TUPLE_RVALUE_SWAP
  BOOL
  "Enable swapping of rvalue tuples (needed for parallel::sort_by_key, default: ON)."
  ON
  CATEGORY "Utility"
  ADVANCED
)
if(PIKA_WITH_TUPLE_RVALUE_SWAP)
  pika_add_config_define(PIKA_HAVE_TUPLE_RVALUE_SWAP)
endif()

# ##############################################################################
# Special diagnostic flags, do not enable by default, only if needed
pika_option(
  PIKA_WITH_CHECK_MODULE_DEPENDENCIES
  BOOL
  "Verify that no modules are cross-referenced from a different module category \
  (default: OFF)"
  OFF
  ADVANCED
)

# Check for compiler compatibility
#

# Check if the selected compiler versions are supposed to work with our codebase
if(CMAKE_COMPILER_IS_GNUCXX AND PIKA_WITH_GCC_VERSION_CHECK)
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.0)
    pika_error(
      "GCC 7.0 or higher is required. Specify PIKA_WITH_GCC_VERSION_CHECK=OFF to ignore this error."
    )
  endif()
endif()

if(MSVC)
  if(MSVC_VERSION LESS 1900)
    pika_error("MSVC x64 2015 or higher is required.")
  elseif(NOT CMAKE_CL_64)
    pika_warn(
      "MSVC (32Bit) will compile but will fail running larger applications because of limitations in the Windows OS."
    )
  endif()
endif()

# Setup platform specific compiler options and check for compatible compilers
if("${PIKA_PLATFORM_UC}" STREQUAL "NATIVE")
  pika_info("Compiling with the native toolset")
elseif("${PIKA_PLATFORM_UC}" STREQUAL "ANDROID")
  pika_info("Compiling for Android devices")
elseif("${PIKA_PLATFORM_UC}" STREQUAL "XEONPHI")
  pika_info("Compiling for Intel Xeon Phi devices")
  if(NOT ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel"))
    pika_error(
      "pika on the MIC can only be compiled with the Intel compiler."
    )
  endif()
elseif("${PIKA_PLATFORM_UC}" STREQUAL "BLUEGENEQ")
  if(NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    pika_error("pika on the BG/Q can only be compiled with bgclang")
  endif()
  pika_info("Compiling for BlueGene/Q")
endif()

# ##############################################################################
# Some special handling of the compilation is need on build infrastructure for
# generating packages for target architecture, see issue #3575

pika_option(
  PIKA_WITH_BUILD_BINARY_PACKAGE
  BOOL
  "Build pika on the build infrastructure on any LINUX distribution (default: OFF)."
  OFF
  ADVANCED
)

# ##############################################################################
# C++ feature tests
# ##############################################################################
# Need to include the CUDA setup before the config test to enable the CUDA
# language
include(pika_setup_cuda)
include(pika_perform_cxx_feature_tests)
pika_perform_cxx_feature_tests()

# ##############################################################################
# Set configuration option to use Boost.Context or not. This depends on the
# platform.
set(__use_generic_coroutine_context OFF)
if(APPLE)
  set(__use_generic_coroutine_context ON)
endif()
if("${PIKA_PLATFORM_UC}" STREQUAL "BLUEGENEQ")
  set(__use_generic_coroutine_context ON)
endif()
pika_option(
  PIKA_WITH_GENERIC_CONTEXT_COROUTINES
  BOOL
  "Use Boost.Context as the underlying coroutines context switch implementation."
  ${__use_generic_coroutine_context}
  ADVANCED
)

# ##############################################################################
# check for miscellaneous things
# ##############################################################################

pika_check_for_mm_prefetch(DEFINITIONS PIKA_HAVE_MM_PREFETCH)

pika_check_for_stable_inplace_merge(
  DEFINITIONS PIKA_HAVE_STABLE_INPLACE_MERGE
)

if(NOT PIKA_WITH_STABLE_INPLACE_MERGE)
  pika_warn(
    "The standard library you are using (libc++ version < 6) does not have a stable inplace_merge implementation."
  )
endif()

# ##############################################################################
# Check for misc system headers
# ##############################################################################

pika_check_for_unistd_h(DEFINITIONS PIKA_HAVE_UNISTD_H)

if(NOT WIN32)
  # ############################################################################
  # Macro definitions for system headers
  # ############################################################################
  add_definitions(-D_GNU_SOURCE)

  # ############################################################################
  # System libraries
  # ############################################################################
  if(NOT MSVC)
    if(PIKA_CXX11_STD_ATOMIC_LIBRARIES)
      target_link_libraries(
        pika_base_libraries
        INTERFACE ${PIKA_CXX11_STD_ATOMIC_LIBRARIES}
      )
    endif()
  endif()

  if(PIKA_HAVE_LIBSUPCPP)
    target_link_libraries(pika_base_libraries INTERFACE supc++)
  endif()

  if(NOT ${CMAKE_SYSTEM_NAME} MATCHES "FreeBSD")
    target_link_libraries(pika_base_libraries INTERFACE dl)
  endif()

  if(NOT APPLE AND NOT ("${PIKA_PLATFORM_UC}" STREQUAL "ANDROID"))
    target_link_libraries(pika_base_libraries INTERFACE rt)
  endif()

  if("${PIKA_PLATFORM_UC}" STREQUAL "ANDROID")
    target_link_libraries(pika_base_libraries INTERFACE log)
  endif()

  if(APPLE)
    pika_add_compile_flag_if_available(-ftemplate-depth=256)
  endif()
endif()

if(WIN32)
  if(MSVC)
    pika_add_target_compile_option(-Ox PUBLIC CONFIGURATIONS Release)

    # even VS2017 has an ICE when compiling with -Ob2
    pika_add_target_compile_option(-Ob1 PUBLIC CONFIGURATIONS Release)

    # VS2012 and above has a special flag for improving the debug experience by
    # adding more symbol information to the build (-d2Zi)
    pika_add_target_compile_option(
      -d2Zi+ PUBLIC CONFIGURATIONS RelWithDebInfo
    )

    # VS2013 and above know how to do link time constant data segment folding
    # VS2013 update 2 and above know how to remove debug information for
    # non-referenced functions and data (-Zc:inline)
    pika_add_target_compile_option(-Zc:inline PUBLIC)
    pika_add_target_compile_option(
      -Gw PUBLIC CONFIGURATIONS Release RelWithDebInfo MinSizeRel
    )
    pika_add_target_compile_option(
      -Zo PUBLIC CONFIGURATIONS RelWithDebInfo
    )
    if(PIKA_WITH_DATAPAR_VC)
      pika_add_target_compile_option(-std:c++latest PUBLIC)
      pika_add_config_cond_define(_HAS_AUTO_PTR_ETC 1)
    endif()

    # Exceptions
    pika_add_target_compile_option(-EHsc)
    if(MSVC_VERSION GREATER_EQUAL 1900)
      # assume conforming (throwing) operator new implementations
      pika_add_target_compile_option(-Zc:throwingNew PUBLIC)

      # enable faster linking (requires VS2015 Update1) disabled for now as this
      # flag crashes debugger pika_remove_link_flag(/debug CONFIGURATIONS
      # Debug) pika_add_link_flag(/debug:fastlink CONFIGURATIONS Debug)

      # Update 2 requires to set _ENABLE_ATOMIC_ALIGNMENT_FIX for it to compile
      # atomics
      pika_add_config_cond_define(_ENABLE_ATOMIC_ALIGNMENT_FIX)

      # Update 3 allows to flag rvalue misuses and enforces strict string const-
      # qualification conformance
      pika_add_target_compile_option(-Zc:rvalueCast PUBLIC)
      pika_add_target_compile_option(-Zc:strictStrings PUBLIC)

      # Update 8 requires to set _ENABLE_EXTENDED_ALIGNED_STORAGE for it to
      # compile atomics
      pika_add_config_cond_define(_ENABLE_EXTENDED_ALIGNED_STORAGE)

      # Make sure that __cplusplus is properly defined
      pika_add_target_compile_option(-Zc:__cplusplus PUBLIC)

      # Silence C++17 deprecation warnings
      pika_add_config_cond_define(_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS)

      # Silence C++20 deprecation warnings
      pika_add_config_cond_define(_SILENCE_ALL_CXX20_DEPRECATION_WARNINGS)

      # ASan is available in Visual Studion starting V16.8
      if((MSVC_VERSION GREATER_EQUAL 1928) AND PIKA_WITH_SANITIZERS)
        pika_add_target_compile_option(
          -fsanitize=address PUBLIC CONFIGURATIONS RelWithDebInfo Debug
        )
      endif()
    endif()

    # Runtime type information
    pika_add_target_compile_option(-GR PUBLIC)
    # Multiprocessor build
    pika_add_target_compile_option(-MP PUBLIC)
    # Increase the maximum size of object file sections
    pika_add_target_compile_option(-bigobj PUBLIC)
  endif()

  target_link_libraries(pika_base_libraries INTERFACE psapi shlwapi)

  # ############################################################################
  # Macro definitions for system headers
  # ############################################################################
  add_definitions(-D_WINDOWS)
  add_definitions(-D_WIN32)
  pika_add_config_cond_define(_WIN32_WINNT 0x0602)
  pika_add_config_cond_define(_SCL_SECURE_NO_WARNINGS)
  pika_add_config_cond_define(_CRT_SECURE_NO_WARNINGS)
  pika_add_config_cond_define(_SCL_SECURE_NO_DEPRECATE)
  pika_add_config_cond_define(_CRT_SECURE_NO_DEPRECATE)
  pika_add_config_cond_define(_CRT_NONSTDC_NO_WARNINGS)
  pika_add_config_cond_define(_WINSOCK_DEPRECATED_NO_WARNINGS)
  pika_add_config_cond_define(_CRT_NON_CONFORMING_SWPRINTFS)
  pika_add_config_cond_define(_SILENCE_FPOS_SEEKPOS_DEPRECATION_WARNING)

  # ############################################################################
  # Boost
  # ############################################################################

  pika_add_config_cond_define(BOOST_USE_WINDOWS_H)
  if(NOT CMAKE_CL_64)
    pika_add_config_cond_define(BOOST_NO_ALIGNMENT)
  endif()
  if(NOT PIKA_WITH_GENERIC_CONTEXT_COROUTINES)
    pika_add_config_define(PIKA_HAVE_FIBER_BASED_COROUTINES)
  endif()
  pika_add_config_cond_define(PSAPI_VERSION 1)
endif()

# Configure Warnings
if(PIKA_WITH_COMPILER_WARNINGS)
  if(MSVC) # Adding special warning settings for the MSVC compiler ...
    pika_add_compile_flag(-W3)

    # MSVC2012/2013 are overeager to report 'qualifier applied to function type
    # has no meaning; ignored'
    pika_add_compile_flag(-wd4180)

    # Boost.Lockfree triggers 'warning C4307: '+' : integral constant overflow'
    # which is benign
    pika_add_compile_flag(-wd4307)

    # object allocated on the heap may not be aligned
    pika_add_compile_flag(-wd4316)

    # max symbol length exceeded
    pika_add_compile_flag(-wd4503)

    # 'int': forcing value to bool 'true' or 'false' (performance warning)
    pika_add_compile_flag(-wd4800)

    # vcpkg enables the /utf-8 option which causes (benign) warnings in the
    # Spirit headers: The file contains a character starting at offset ... that
    # is illegal in the current source character set
    pika_add_compile_flag(-wd4828)

    if(PIKA_WITH_DATAPAR_VC)
      # unary minus operator applied to unsigned type, result still unsigned
      pika_add_compile_flag(-wd4146)

      # '<=': signed/unsigned mismatch
      pika_add_compile_flag(-wd4018)

      # 'return': conversion from 'short' to 'Vc_1::schar', possible loss of
      # data
      pika_add_compile_flag(-wd4244)
    endif()

  else() # Trial and error approach for any other compiler ...
    pika_add_compile_flag_if_available(-Wall)
    pika_add_compile_flag_if_available(-Wextra)
    pika_add_compile_flag_if_available(-Wpedantic)
    pika_add_compile_flag_if_available(-Wno-strict-aliasing)
    pika_add_compile_flag_if_available(-Wno-sign-promo)
    pika_add_compile_flag_if_available(-Wno-attributes)
    pika_add_compile_flag_if_available(-Wno-cast-align)

    # We do not in general guarantee ABI compatibility between C++ standards, so
    # we ignore this warning
    pika_add_compile_flag_if_available(-Wno-noexcept-type)

    # Be extra strict about format checks Boost.Logging is built on fprintf,
    # sadly
    pika_add_compile_flag_if_available(-Wformat=2)
    pika_add_compile_flag_if_available(-Wno-format-nonliteral)

    # Self initialization is dangerous
    pika_add_compile_flag_if_available(-Winit-self)

    # For portability
    pika_add_compile_flag_if_available(-Wdouble-promotion)

    # Warn about casting that violates qualifiers or alignment
    pika_add_compile_flag_if_available(-Wcast-qual)
    if(NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
      # Clang is overeager in reporting cast alignment problems in Boost
      pika_add_compile_flag_if_available(-Wcast-align)
    endif()

    # False positive when build with Vc and Clang
    if(NOT (PIKA_WITH_DATAPAR_VC AND "${CMAKE_CXX_COMPILER_ID}" STREQUAL
                                         "Clang")
    )
      pika_add_compile_flag_if_available(-Werror=trampolines)
    endif()
    pika_add_compile_flag_if_available(-Werror=parentheses)
    pika_add_compile_flag_if_available(-Werror=reorder)
    pika_add_compile_flag_if_available(-Werror=return-type)
    pika_add_compile_flag_if_available(-Werror=sequence-point)
    pika_add_compile_flag_if_available(-Werror=uninitialized)
    pika_add_compile_flag_if_available(-Werror=format)
    pika_add_compile_flag_if_available(-Werror=missing-braces)
    pika_add_compile_flag_if_available(-Werror=sign-compare)
  endif()
endif()

# Configure compiler warnings as errors
if(PIKA_WITH_COMPILER_WARNINGS_AS_ERRORS)
  if(MSVC)
    pika_add_compile_flag(-WX)
  else()
    pika_add_compile_flag_if_available(-Werror)
  endif()
endif()

# Diagnostics
if(MSVC)
  # Display full paths in diagnostics
  pika_add_compile_flag(-FC)
  if(CMAKE_CL_64)
    set(__target_arch "x86_64")
  else()
    set(__target_arch "x86")
  endif()
  pika_info("Architecture detected: ${__target_arch}")
else()
  # Show the flags that toggle each warning
  pika_add_compile_flag_if_available(-fdiagnostics-show-option)

  # VLAs are a GNU extensions that we forbid as they are not supported on MSVC
  pika_add_compile_flag_if_available(-Werror=vla)
  # No return statement in a non-void function can lead to garbage return values
  # in GCC.
  pika_add_compile_flag_if_available(-Werror=return-type)

  # We get false positives all over the place with this.
  if(CMAKE_COMPILER_IS_GNUCXX)
    pika_add_compile_flag_if_available(-Wno-unused-but-set-parameter)
    pika_add_compile_flag_if_available(-Wno-unused-but-set-variable)
    # Uninitialized variables are bad, earlier compilers issue spurious warnings
    pika_add_compile_flag_if_available(-Werror=uninitialized)
    pika_add_compile_flag_if_available(-Wno-unused-local-typedefs)
    # -Werror=maybe-uninitialized leads to false positives.
    pika_add_compile_flag_if_available(-Wno-maybe-uninitialized)
  endif()

  # False positive when building with Vc and Clang
  if(NOT (PIKA_WITH_DATAPAR_VC AND "${CMAKE_CXX_COMPILER_ID}" STREQUAL
                                       "Clang")
  )
    # Silence warning about __sync_fetch_and_nand changing semantics
    pika_add_compile_flag_if_available(-Wno-sync-nand)
  endif()

  # Silence warnings about deleting polymorphic objects with non-virtual dtors.
  # These come from within Boost.
  if(CMAKE_COMPILER_IS_GNUCXX)
    pika_add_compile_flag_if_available(-Wno-delete-non-virtual-dtor)
  endif()

  # Check if our libraries have unresolved symbols
  if(NOT APPLE
     AND NOT WIN32
     AND NOT PIKA_WITH_SANITIZERS
  )
    pika_add_link_flag_if_available(-Wl,-z,defs)
  endif()
  if(WIN32)
    target_link_libraries(
      pika_base_libraries INTERFACE psapi WS2_32 mswsock
    )
  endif()

  if("${PIKA_PLATFORM_UC}" STREQUAL "BLUEGENEQ")
    pika_add_compile_flag_if_available(-Wno-deprecated-register)
  endif()

  if(PIKA_WITH_HIDDEN_VISIBILITY)
    pika_add_compile_flag_if_available(-fvisibility=hidden)
    pika_add_link_flag_if_available(-fvisibility=hidden)
    pika_add_config_define(PIKA_HAVE_ELF_HIDDEN_VISIBILITY)
  endif()

  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    pika_add_compile_flag_if_available(-Wno-cast-align)
  endif()

  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    # Disable the following warnings: #1170: invalid redeclaration of nested
    # class
    pika_add_compile_flag_if_available(-wd1170)
    # #858: type qualifier on return type is meaningless
    pika_add_compile_flag_if_available(-wd858)
    # #1098: the qualifier on this friend declaration is ignored
    pika_add_compile_flag_if_available(-wd1098)
    # #488: template parameter not used in declaring the parameter type
    pika_add_compile_flag_if_available(-wd488)
    # #2203: cast discards qualifiers from target type (needed for mvapich2 mpi
    # header)
    pika_add_compile_flag_if_available(-wd2203)
    # #2536: cannot specify explicit initializer for arrays
    pika_add_compile_flag_if_available(-wd2536)
    # #1286: invalid attribute
    pika_add_compile_flag_if_available(-wd1286)
  endif()

  set(__has_timestamp_support OFF)
  include(pika_target_arch)

  pika_target_architecture(__target_arch)
  if("${__target_arch}" STREQUAL "i386"
     OR "${__target_arch}" STREQUAL "ix86"
     OR "${__target_arch}" STREQUAL "x86_64"
     OR "${__target_arch}" STREQUAL "ia64"
  )

    # rdtsc is an x86 instruction that reads the value of a CPU time stamp
    # counter. rdtscp is an extension to rdtsc. The difference is that rdtscp is
    # a serializing instruction.
    pika_cpuid("rdtsc" PIKA_WITH_RDTSC DEFINITIONS PIKA_HAVE_RDTSC)

    # One can not assume if RDTSCP is available on the hardware of the build
    # infrastructure, that it will be available on all potential target
    # hardware, see Issue  #3575
    if(NOT ${PIKA_WITH_BUILD_BINARY_PACKAGE})

      # XeonPhi's do not support RDTSCP
      if(NOT ("${PIKA_PLATFORM_UC}" STREQUAL "XEONPHI"))
        pika_cpuid(
          "rdtscp" PIKA_WITH_RDTSCP DEFINITIONS PIKA_HAVE_RDTSCP
        )
      endif()

    endif()
    if(PIKA_WITH_RDTSC OR PIKA_WITH_RDTSCP)
      set(__has_timestamp_support ON)
    endif()
  elseif(
    "${__target_arch}" STREQUAL "arm"
    OR "${__target_arch}" STREQUAL "armv5"
    OR "${__target_arch}" STREQUAL "armv6"
    OR "${__target_arch}" STREQUAL "armv7"
  )
    set(__has_timestamp_support ON)
  elseif("${__target_arch}" STREQUAL "ppc" OR "${__target_arch}" STREQUAL
                                              "ppc64"
  )
    set(__has_timestamp_support ON)
  elseif("${__target_arch}" STREQUAL "bgq")
    set(__has_timestamp_support ON)
  elseif("${__target_arch}" STREQUAL "s390x")
    set(__has_timestamp_support ON)
  endif()

  pika_info("Architecture detected: ${__target_arch}")
  if(NOT __has_timestamp_support)
    pika_warn(
      "No timestamp support is available; some performance counters may report incorrect results"
    )
  endif()
endif()

# store target architecture for later use
set(PIKA_WITH_TARGET_ARCHITECTURE
    ${__target_arch}
    CACHE INTERNAL "" FORCE
)

# Compatibility with using Boost.FileSystem, introduced in V1.4.0
set(__filesystem_compatibility_default ON)
if(PIKA_WITH_CXX17_FILESYSTEM)
  set(__filesystem_compatibility_default OFF)
endif()
pika_option(
  PIKA_FILESYSTEM_WITH_BOOST_FILESYSTEM_COMPATIBILITY
  BOOL
  "Enable Boost.FileSystem compatibility. (default: ${__filesystem_compatibility_default})"
  ${__filesystem_compatibility_default}
  ADVANCED
  CATEGORY "Modules"
)

# Compatibility with using Boost.Iterator traversal tags, introduced in V1.7.0
pika_option(
  PIKA_ITERATOR_SUPPORT_WITH_BOOST_ITERATOR_TRAVERSAL_TAG_COMPATIBILITY BOOL
  "Enable Boost.Iterator traversal tag compatibility. (default: OFF)" OFF
  ADVANCED CATEGORY "Modules"
)
if(PIKA_ITERATOR_SUPPORT_WITH_BOOST_ITERATOR_TRAVERSAL_TAG_COMPATIBILITY)
  pika_add_config_define_namespace(
    DEFINE PIKA_ITERATOR_SUPPORT_HAVE_BOOST_ITERATOR_TRAVERSAL_TAG_COMPATIBILITY
    NAMESPACE ITERATOR_SUPPORT
  )
endif()

# ##############################################################################
# Find Our dependencies: These are all dependencies needed to build the core
# library. Dependencies that are only needed by plugins, examples or tests
# should be found separately in the appropriate subdirectory.

include(pika_setup_threads)
include(pika_setup_boost)
include(pika_setup_boost_filesystem)
include(pika_setup_allocator)
include(pika_setup_hwloc)
include(pika_setup_mpi)
include(pika_setup_hip)
include(pika_setup_valgrind)

if(PIKA_WITH_SANITIZERS)
  pika_add_config_define(PIKA_HAVE_SANITIZERS)
endif()

if(PIKA_WITH_VIM_YCM)
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

# ##############################################################################
# Check Build Options based on the found dependencies. We also check for errors
# with incompatible options with the currently selected platform.
#

if(PIKA_WITH_GENERIC_CONTEXT_COROUTINES)
  # Check if we can use generic coroutine contexts without any problems
  if(NOT Boost_CONTEXT_FOUND)
    pika_error(
      "The usage of Boost.Context was selected but Boost.Context was not found."
    )
  endif()
  pika_add_config_define(PIKA_HAVE_GENERIC_CONTEXT_COROUTINES)
endif()

# ##############################################################################

# Note: on windows systems the ':' will be converted to a ';' at runtime
pika_add_config_cond_define(
  PIKA_DEFAULT_COMPONENT_PATH_SUFFIXES
  "\"/${CMAKE_INSTALL_LIBDIR}/pika:/${CMAKE_INSTALL_BINDIR}/pika:/lib/pika:/bin/pika\""
)

# ##############################################################################
# search path configuration
# ##############################################################################
if(PIKA_WITH_FULL_RPATH)
  include(pika_set_full_rpath)
endif()

# ##############################################################################
# Git commit detection
# ##############################################################################
include(pika_git_commit)
pika_add_config_define(
  PIKA_HAVE_GIT_COMMIT "\"${PIKA_WITH_GIT_COMMIT}\""
)

include(pika_set_output_paths)

# ##############################################################################
# Add custom targets for tests
# ##############################################################################
if(PIKA_WITH_TESTS)
  pika_add_pseudo_target(tests)
  if(PIKA_WITH_TESTS_UNIT)
    pika_add_pseudo_target(tests.unit)
    pika_add_pseudo_dependencies(tests tests.unit)
  endif()
  if(PIKA_WITH_TESTS_REGRESSIONS)
    pika_add_pseudo_target(tests.regressions)
    pika_add_pseudo_dependencies(tests tests.regressions)
  endif()
  if(PIKA_WITH_TESTS_BENCHMARKS)
    pika_add_pseudo_target(tests.performance)
    pika_add_pseudo_dependencies(tests tests.performance)
  endif()
  if(PIKA_WITH_TESTS_HEADERS)
    pika_add_pseudo_target(tests.headers)
    pika_add_pseudo_dependencies(tests tests.headers)
  endif()
  if(PIKA_WITH_EXAMPLES AND PIKA_WITH_TESTS_EXAMPLES)
    pika_add_pseudo_target(tests.examples)
    pika_add_pseudo_dependencies(tests tests.examples)
  endif()

  enable_testing()
  include(CTest)
endif()

if(PIKA_WITH_EXAMPLES)
  pika_add_pseudo_target(examples)
endif()

# ##############################################################################
# Debug library postfix
# ##############################################################################
set(CMAKE_DEBUG_POSTFIX "d")
set(PIKA_DEBUG_POSTFIX "d")

# ##############################################################################
# Add libraries
# ##############################################################################
add_subdirectory(libs)

# ##############################################################################
# Target specification
# ##############################################################################
if(PIKA_WITH_TOOLS OR PIKA_WITH_TESTS_BENCHMARKS)
  pika_add_pseudo_target(tools)
  add_subdirectory(tools)
endif()

# ##############################################################################
# Tests
# ##############################################################################
if(PIKA_WITH_TESTS)
  find_package(PythonInterp)

  if(NOT PYTHONINTERP_FOUND)
    pika_warn(
      "A python interpreter could not be found. The test suite can not be run automatically."
    )
  endif()

  # pseudo_target added above
  add_subdirectory(tests)
endif()

if(PIKA_WITH_EXAMPLES)
  add_subdirectory(examples)
endif()

# Set up precompiled headers
if(PIKA_WITH_PRECOMPILED_HEADERS)
  set(system_precompiled_headers
      <algorithm>
      <array>
      <atomic>
      <bitset>
      <cassert>
      <cctype>
      <cerrno>
      <chrono>
      <climits>
      <cmath>
      <complex>
      <condition_variable>
      <cstddef>
      <cstdint>
      <cstdio>
      <cstdlib>
      <cstring>
      <ctime>
      <deque>
      <exception>
      <forward_list>
      <fstream>
      <functional>
      <iomanip>
      <ios>
      <iosfwd>
      <iostream>
      <iterator>
      <limits>
      <list>
      <locale>
      <map>
      <memory>
      <mutex>
      <new>
      <numeric>
      <ostream>
      <random>
      <regex>
      <set>
      <shared_mutex>
      <sstream>
      <stack>
      <stdexcept>
      <string>
      <string_view>
      <system_error>
      <thread>
      <tuple>
      <type_traits>
      <typeinfo>
      <unordered_map>
      <unordered_set>
      <utility>
      <variant>
      <vector>
      <hwloc.h>
      <boost/config.hpp>
      <boost/intrusive/slist.hpp>
      <boost/lockfree/queue.hpp>
      <boost/optional.hpp>
      <boost/tokenizer.hpp>
  )

  if(PIKA_WITH_CXX17_FILESYSTEM)
    list(APPEND system_precompiled_headers <filesystem>)
  endif()

  if(PIKA_WITH_CXX17_STD_EXECUTION_POLICIES)
    list(APPEND system_precompiled_headers <execution>)
  endif()

  set(system_precompiled_headers_dependencies
      pika_dependencies_boost pika_private_flags
      pika_public_flags
  )

  target_link_libraries(
    pika_precompiled_headers
    PRIVATE pika_public_flags pika_private_flags
            pika_base_libraries ${system_precompiled_headers_dependencies}
  )
  target_precompile_headers(
    pika_precompiled_headers PRIVATE ${system_precompiled_headers}
  )

  # Headers that should be precompiled for things depending on pika
  # (executables, libraries).
  set(pika_precompiled_headers_modules ${PIKA_ENABLED_MODULES})
  # The init_runtime_local module cannot be precompiled because it uses macros
  # that can depend on the application (PIKA_PREFIX and
  # PIKA_APPLICATION_NAME_DEFAULT).
  list(REMOVE_ITEM pika_precompiled_headers_modules init_runtime_local)
  # config, version, and the include modules don't have pika/module headers.
  list(REMOVE_ITEM pika_precompiled_headers_modules config_local
       include_local version_local
  )
  list(TRANSFORM pika_precompiled_headers_modules PREPEND "<pika/modules/")
  list(TRANSFORM pika_precompiled_headers_modules APPEND ".hpp>")

  target_link_libraries(pika_exe_precompiled_headers PRIVATE pika)
  target_precompile_headers(
    pika_exe_precompiled_headers PRIVATE
    ${pika_precompiled_headers_modules}
  )
endif()

# ##############################################################################
# installation instructions
# ##############################################################################
# Install all pika header that have been configured using various CMake
# options
install(
  DIRECTORY "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/pika/"
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/pika
  COMPONENT core
  FILES_MATCHING
  PATTERN "*.hpp"
  PATTERN ".git" EXCLUDE
  PATTERN "CMakeFiles" EXCLUDE
  PATTERN "CTestFiles" EXCLUDE
)

install(
  # Install all pika CMake utility files
  DIRECTORY cmake/
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/pika
  COMPONENT core
  PATTERN "templates" EXCLUDE
  PATTERN "packaging" EXCLUDE
  PATTERN ".git" EXCLUDE
)

install(
  FILES "${PROJECT_SOURCE_DIR}/LICENSE_1_0.txt"
  DESTINATION ${CMAKE_INSTALL_DATADIR}/pika
  COMPONENT license
)

if(PIKA_WITH_VIM_YCM)
  set(build_dir_file ${PROJECT_BINARY_DIR}/.ycm_extra_conf.py)
  set(source_dir_file ${PROJECT_SOURCE_DIR}/.ycm_extra_conf.py)
  configure_file(
    ${PROJECT_SOURCE_DIR}/tools/vim/.ycm_extra_conf.py ${build_dir_file} @ONLY
  )
  add_custom_target(
    configure_ycm
    COMMAND ${CMAKE_COMMAND} -E copy ${build_dir_file} ${source_dir_file}
    COMMENT "Copying YCM config file to source directory"
    VERBATIM
  )
  pika_info(
    "VIM YouCompleteMe: run 'make configure_ycm' to copy config file to source directory and enable support in YCM. To enable automatic loading of configure file, add to your .vimrc option: \"let g:ycm_extra_conf_globlist = ['${PROJECT_SOURCE_DIR}/*']\""
  )
endif()

# ##############################################################################
# Add rpm packaging

pika_option(
  PIKA_WITH_RPM BOOL "Enable or disable the generation of rpm packages" OFF
  ADVANCED
)

if(PIKA_WITH_RPM)
  add_subdirectory(cmake/packaging/rpm)
endif()

# ##############################################################################
# print overall configuration summary
include(pika_print_summary)
pika_create_configuration_summary("Configuration summary:\n--" "pika")

include(pika_export_targets)
# Modules can't link to this if not exported
install(
  TARGETS pika_base_libraries
  EXPORT pika_internal_targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
          COMPONENT pika_base_libraries
)
pika_export_internal_targets(pika_base_libraries)

# ##############################################################################
# store cache vars and their values in order for them to be forwarded to the
# projects (needs to be before the PIKA_GeneratePackage call)
include(pika_forward_cache_variables)

# ##############################################################################
# External build system support (Findpika.cmake and pkg-config).
include(pika_generate_package)

message("")
message("pika will be installed to ${CMAKE_INSTALL_PREFIX}")
message("")
