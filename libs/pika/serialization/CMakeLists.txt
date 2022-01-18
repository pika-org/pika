# Copyright (c) 2019-2021 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Compatibility with various Boost types, introduced in V1.4.0
pika_option(
  PIKA_SERIALIZATION_WITH_BOOST_TYPES BOOL
  "Enable serialization of certain Boost types. (default: ON)" ON ADVANCED
  CATEGORY "Modules"
  MODULE SERIALIZATION
)

if(PIKA_SERIALIZATION_WITH_BOOST_TYPES)
  pika_add_config_define_namespace(
    DEFINE PIKA_SERIALIZATION_HAVE_BOOST_TYPES NAMESPACE SERIALIZATION
  )
endif()

# cmake-format: off
#
# Important note: The following flags are specific for using pika as a
# backend for SHAD (https://github.com/pnnl/SHAD):
#
# - PIKA_SERIALIZATION_WITH_ALL_TYPES_ARE_BITWISE_SERIALIZABLE
# - PIKA_SERIALIZATION_WITH_ALLOW_RAW_POINTER_SERIALIZATION
# - PIKA_SERIALIZATION_WITH_ALLOW_CONST_TUPLE_MEMBERS
#
# They all enable questionable functionalities, partially they even enable
# undefined behavior. Please only use any of them if you know what you're
# doing.
#
# cmake-format: on

# This flag can be used in systems that assume that all types are bitwise
# serializable by default (like SHAD).
pika_option(
  PIKA_SERIALIZATION_WITH_ALL_TYPES_ARE_BITWISE_SERIALIZABLE BOOL
  "Assume all types are bitwise serializable. (default: OFF)" OFF ADVANCED
  CATEGORY "Modules"
  MODULE SERIALIZATION
)

if(PIKA_SERIALIZATION_WITH_ALL_TYPES_ARE_BITWISE_SERIALIZABLE)
  pika_add_config_define_namespace(
    DEFINE PIKA_SERIALIZATION_HAVE_ALL_TYPES_ARE_BITWISE_SERIALIZABLE
    NAMESPACE SERIALIZATION
  )
endif()

# This flag can be used in systems that rely on sending raw pointers to other
# localities (assuming those are being sent back to the originating locality
# before they are dereferenced (like SHAD).
pika_option(
  PIKA_SERIALIZATION_WITH_ALLOW_RAW_POINTER_SERIALIZATION BOOL
  "Enable serializing raw pointers. (default: OFF)" OFF ADVANCED
  CATEGORY "Modules"
  MODULE SERIALIZATION
)

if(PIKA_SERIALIZATION_WITH_ALLOW_RAW_POINTER_SERIALIZATION)
  pika_add_config_define_namespace(
    DEFINE PIKA_SERIALIZATION_HAVE_ALLOW_RAW_POINTER_SERIALIZATION
    NAMESPACE SERIALIZATION
  )
endif()

# This flag can be used in systems that rely on sending constant values as part
# of std::tuple. This option essentially casts away constness for tuple members.
pika_option(
  PIKA_SERIALIZATION_WITH_ALLOW_CONST_TUPLE_MEMBERS BOOL
  "Enable serializing std::tuple with const members. (default: OFF)" OFF
  ADVANCED
  CATEGORY "Modules"
  MODULE SERIALIZATION
)

if(PIKA_SERIALIZATION_WITH_ALLOW_CONST_TUPLE_MEMBERS)
  pika_add_config_define_namespace(
    DEFINE PIKA_SERIALIZATION_HAVE_ALLOW_CONST_TUPLE_MEMBERS
    NAMESPACE SERIALIZATION
  )
endif()

# Default location is $PIKA_ROOT/libs/serialization/include
set(serialization_headers
    pika/serialization.hpp
    pika/serialization/detail/constructor_selector.hpp
    pika/serialization/detail/extra_archive_data.hpp
    pika/serialization/detail/non_default_constructible.hpp
    pika/serialization/detail/pointer.hpp
    pika/serialization/detail/polymorphic_id_factory.hpp
    pika/serialization/detail/polymorphic_intrusive_factory.hpp
    pika/serialization/detail/polymorphic_nonintrusive_factory.hpp
    pika/serialization/detail/polymorphic_nonintrusive_factory_impl.hpp
    pika/serialization/detail/preprocess_container.hpp
    pika/serialization/detail/raw_ptr.hpp
    pika/serialization/detail/serialize_collection.hpp
    pika/serialization/detail/vc.hpp
    pika/serialization/array.hpp
    pika/serialization/bitset.hpp
    pika/serialization/complex.hpp
    pika/serialization/datapar.hpp
    pika/serialization/deque.hpp
    pika/serialization/dynamic_bitset.hpp
    pika/serialization/exception_ptr.hpp
    pika/serialization/list.hpp
    pika/serialization/map.hpp
    pika/serialization/multi_array.hpp
    pika/serialization/optional.hpp
    pika/serialization/set.hpp
    pika/serialization/serialize_buffer.hpp
    pika/serialization/string.hpp
    pika/serialization/std_tuple.hpp
    pika/serialization/tuple.hpp
    pika/serialization/unordered_map.hpp
    pika/serialization/vector.hpp
    pika/serialization/variant.hpp
    pika/serialization/valarray.hpp
    pika/serialization/intrusive_ptr.hpp
    pika/serialization/shared_ptr.hpp
    pika/serialization/unique_ptr.hpp
    pika/serialization/access.hpp
    pika/serialization/base_object.hpp
    pika/serialization/basic_archive.hpp
    pika/serialization/binary_filter.hpp
    pika/serialization/brace_initializable.hpp
    pika/serialization/brace_initializable_fwd.hpp
    pika/serialization/container.hpp
    pika/serialization/input_archive.hpp
    pika/serialization/input_container.hpp
    pika/serialization/output_archive.hpp
    pika/serialization/output_container.hpp
    pika/serialization/serializable_any.hpp
    pika/serialization/serialization_chunk.hpp
    pika/serialization/serialization_fwd.hpp
    pika/serialization/serialize.hpp
    pika/serialization/traits/brace_initializable_traits.hpp
    pika/serialization/traits/is_bitwise_serializable.hpp
    pika/serialization/traits/is_not_bitwise_serializable.hpp
    pika/serialization/traits/needs_automatic_registration.hpp
    pika/serialization/traits/polymorphic_traits.hpp
    pika/serialization/traits/serialization_access_data.hpp
)

if(PIKA_SERIALIZATION_WITH_BOOST_TYPES)
  set(serialization_headers ${serialization_headers}
                            pika/serialization/boost_variant.hpp
  )
endif()

# Default location is $PIKA_ROOT/libs/serialization/src
set(serialization_sources
    detail/pointer.cpp
    detail/polymorphic_id_factory.cpp
    detail/polymorphic_intrusive_factory.cpp
    detail/polymorphic_nonintrusive_factory.cpp
    exception_ptr.cpp
    serializable_any.cpp
)

if(TARGET Vc::vc)
  set(serialization_optional_dependencies Vc::vc)
endif()

include(pika_add_module)
pika_add_module(
  pika serialization
  GLOBAL_HEADER_GEN ON
  SOURCES ${serialization_sources}
  HEADERS ${serialization_headers}
  MODULE_DEPENDENCIES
    pika_assertion
    pika_config_local
    pika_datastructures
    pika_debugging
    pika_errors
    pika_format
    pika_hashing
    pika_preprocessor
    pika_type_support
  DEPENDENCIES ${serialization_optional_dependencies}
  CMAKE_SUBDIRS examples tests
)
