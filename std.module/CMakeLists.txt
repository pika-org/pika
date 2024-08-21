# if (CMAKE_VERSION VERSION_LESS 3.28)
#   if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.27)
#     set(CMAKE_EXPERIMENTAL_CXX_MODULE_CMAKE_API "aa1f7df0-828a-4fcd-9afc-2dc80491aca7")
#   else ()
#     set(CMAKE_EXPERIMENTAL_CXX_MODULE_CMAKE_API "2182bf5c-ef0d-489a-91da-49dbc3090d2a")
#   endif()
#   set(CMAKE_EXPERIMENTAL_CXX_MODULE_DYNDEP 1)
# else()
#   cmake_policy(VERSION 3.28)
# endif()
set(CMAKE_CXX_EXTENSIONS OFF)
set(CXX_STANDARD_REQUIRED ON)

add_library(std STATIC)
# the default PMIF is 'std.ixx'
set(pmif std.ixx)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 17)
    # Clang 16 requires a horrible hack using invalid C++ syntax
    set(pmif std.cppm)
    # libstdc++ is required because libc++
    #  - is not even at C++17
    #  - is lacking essential C++20 features
    # but
    #  - Clang 16 has a bugs affecting the implementation of std::ranges in C++23 mode
    #    apply the patch to deal with them!
    #  - the TBB of libstdc++ is not importable
    set(std-lib-additional -D_PSTL_PAR_BACKEND_SERIAL)
  endif()
  # I do mean C++20
  set(std-lang-additional -fsized-deallocation -faligned-allocation)
  # libstdc++/libc++ has no support for a C++23 module 'std'
  # fall back to emulation, don't define _BUILD_STD_MODULE
endif()

target_sources(std PUBLIC FILE_SET modules TYPE CXX_MODULES FILES ${pmif})
target_compile_features(std PUBLIC cxx_std_23)

set(allstd-hu ${CMAKE_CURRENT_SOURCE_DIR}/allstd.cpp)
if (MSVC)
  set(allstd-bmi ${CMAKE_CURRENT_BINARY_DIR}/allstd.ifc)
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(allstd-bmi ${CMAKE_CURRENT_BINARY_DIR}/allstd.pcm)
else()
  message(FATAL_ERROR "Sorry, unsupported compiler!")
endif()

add_subdirectory(allstd)

if (MSVC)
  target_link_libraries(std PUBLIC allstd)
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_options(std
    PUBLIC ${std-lang-additional}
    PRIVATE -fmodule-file=${allstd-bmi}
  )
  if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 17)
    target_compile_options(std
      PRIVATE -Wno-reserved-module-identifier -Wno-experimental-header-units
  )
  endif()
set_source_files_properties(${pmif} PROPERTIES OBJECT_DEPENDS ${allstd-bmi})
endif()

find_package(Threads REQUIRED)
target_link_libraries(std PRIVATE pika_base_libraries pika_private_flags)
target_link_libraries(std PUBLIC pika_public_flags)
set_target_properties(std PROPERTIES POSITION_INDEPENDENT_CODE ON)

include(pika_export_targets)
install(
  TARGETS std
  EXPORT pika_internal_targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  CXX_MODULES_BMI DESTINATION ${CMAKE_INSTALL_BINDIR}
  FILE_SET modules DESTINATION ${CMAKE_INSTALL_BINDIR}
  COMPONENT std
)
pika_export_internal_targets(std)
