unset(CMAKE_EXPERIMENTAL_CXX_MODULE_CMAKE_API)
unset(CMAKE_EXPERIMENTAL_CXX_MODULE_DYNDEP)

add_library(allstd OBJECT ${allstd-hu})
find_package(Threads REQUIRED)
target_link_libraries(std PRIVATE pika_base_libraries pika_private_flags)
target_link_libraries(std PUBLIC pika_public_flags)
set_target_properties(std PROPERTIES POSITION_INDEPENDENT_CODE ON)
get_target_property(standard std CXX_STANDARD)
set_target_properties(allstd PROPERTIES CXX_STANDARD ${standard})
target_compile_features(allstd PUBLIC cxx_std_23)
# TODO: What's going on here? clang/CMake can't see these include directories when scanning dependencies
target_include_directories(allstd PRIVATE "/home/mjs/src/spack/opt/spack/linux-nixos24-icelake/gcc-14.1.0/hwloc-2.9.3-rhxc5cebnw7akbzkqrqsvla77e2qlkis/include")
target_include_directories(allstd PRIVATE "/home/mjs/src/spack/opt/spack/linux-nixos24-icelake/clang-18.1.7/boost-1.85.0-kppkj3cspk54vnmmct7kgghxjqx2vbrw/include")
target_include_directories(allstd PRIVATE "/home/mjs/src/spack/opt/spack/linux-nixos24-icelake/clang-18.1.7/fmt-11.0.2-qvpzmsot7ugydc2dildfzz5bgdsgsxvr/include")
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_options(allstd
    PUBLIC ${std-lang-additional}
    PRIVATE ${std-lib-additional} -pthread -Wno-builtin-macro-redefined -xc++-system-header --precompile
  )
  add_custom_target(copyBMI
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_OBJECTS:allstd> ${allstd-bmi}
    COMMAND_EXPAND_LISTS
    DEPENDS $<TARGET_OBJECTS:allstd>
    BYPRODUCTS ${allstd-bmi}
  )
endif()

