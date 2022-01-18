# Copyright (c) 2019-2020 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

pika_option(
  PIKA_COROUTINES_WITH_SWAP_CONTEXT_EMULATION BOOL
  "Emulate SwapContext API for coroutines (Windows only, default: OFF)" OFF
  CATEGORY "Thread Manager"
  ADVANCED
  MODULE COROUTINES
)

set(coroutines_headers
    pika/coroutines/coroutine.hpp
    pika/coroutines/coroutine_fwd.hpp
    pika/coroutines/stackless_coroutine.hpp
    pika/coroutines/detail/combined_tagged_state.hpp
    pika/coroutines/detail/context_base.hpp
    pika/coroutines/detail/context_generic_context.hpp
    pika/coroutines/detail/context_impl.hpp
    pika/coroutines/detail/context_linux_x86.hpp
    pika/coroutines/detail/context_posix.hpp
    pika/coroutines/detail/context_windows_fibers.hpp
    pika/coroutines/detail/coroutine_accessor.hpp
    pika/coroutines/detail/coroutine_impl.hpp
    pika/coroutines/detail/coroutine_self.hpp
    pika/coroutines/detail/coroutine_stackful_self.hpp
    pika/coroutines/detail/coroutine_stackless_self.hpp
    pika/coroutines/detail/get_stack_pointer.hpp
    pika/coroutines/detail/posix_utility.hpp
    pika/coroutines/detail/swap_context.hpp
    pika/coroutines/detail/tss.hpp
    pika/coroutines/thread_enums.hpp
    pika/coroutines/thread_id_type.hpp
)

set(coroutines_sources
    detail/context_base.cpp
    detail/context_posix.cpp
    detail/coroutine_impl.cpp
    detail/coroutine_self.cpp
    detail/posix_utility.cpp
    detail/tss.cpp
    swapcontext.cpp
    thread_enums.cpp
)

if(MSVC)
  if(PIKA_COROUTINES_WITH_SWAP_CONTEXT_EMULATION)
    # ##########################################################################
    # Emulation of SwapContext on Windows
    # ##########################################################################
    enable_language(ASM_MASM)
    if(NOT CMAKE_ASM_MASM_COMPILER)
      pika_error(
        "SwitchToFiber emulation can not be enabled. The masm compiler \
         could not be found. Try setting the ASM_MASM environment variable to the \
         assembler executable (ml.exe/ml64.exe) or disable the emulation by setting \
         PIKA_COROUTINES_WITH_SWAP_CONTEXT_EMULATION to Off"
      )
    endif()

    pika_add_config_define_namespace(
      DEFINE PIKA_COROUTINES_HAVE_SWAP_CONTEXT_EMULATION NAMESPACE COROUTINES
    )

    set(switch_to_fiber_source
        "${CMAKE_CURRENT_SOURCE_DIR}/src/switch_to_fiber.asm"
    )
    set(switch_to_fiber_object
        "${CMAKE_CURRENT_BINARY_DIR}/switch_to_fiber.obj"
    )
    add_custom_command(
      OUTPUT "${switch_to_fiber_object}"
      COMMAND "${CMAKE_ASM_MASM_COMPILER}" /Fo "${switch_to_fiber_object}"
              /nologo /c "${switch_to_fiber_source}"
      DEPENDS "${switch_to_fiber_source}"
      VERBATIM
    )
  endif()
elseif(PIKA_COROUTINES_WITH_SWAP_CONTEXT_EMULATION)
  pika_error(
    "The option PIKA_COROUTINES_WITH_SWAP_CONTEXT_EMULATION is not supported on "
    "this platform, please disable the emulation by setting it to Off"
  )
endif()

set_source_files_properties(
  src/detail/coroutine_impl.cpp PROPERTIES SKIP_UNITY_BUILD_INCLUSION TRUE
)
set_source_files_properties(
  src/detail/coroutine_self.cpp PROPERTIES SKIP_UNITY_BUILD_INCLUSION TRUE
)

include(pika_add_module)
pika_add_module(
  pika coroutines
  GLOBAL_HEADER_GEN ON
  SOURCES ${coroutines_sources}
  HEADERS ${coroutines_headers}
  OBJECTS "${switch_to_fiber_object}"
  DEPENDENCIES $<TARGET_NAME_IF_EXISTS:Valgrind::valgrind>
  MODULE_DEPENDENCIES
    pika_assertion
    pika_config_local
    pika_errors
    pika_format
    pika_functional
    pika_memory
    pika_thread_support
    pika_type_support
    pika_util
  CMAKE_SUBDIRS examples tests
)

if(MSVC AND PIKA_COROUTINES_WITH_SWAP_CONTEXT_EMULATION)
  pika_info(
    "    SwitchToFiber emulation is enabled, using compiler: '${CMAKE_ASM_MASM_COMPILER}'"
  )
endif()
