# Copyright (c) 2014 Thomas Heller
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(PIKA_PLATFORM_CHOICES
    "Choices are: native, Android, XeonPhi, BlueGeneQ."
)
set(PIKA_PLATFORMS_UC "NATIVE;ANDROID;XEONPHI;BLUEGENEQ")

if(NOT PIKA_PLATFORM)
  set(PIKA_PLATFORM
      "native"
      CACHE
        STRING
        "Sets special compilation flags for specific platforms. ${PIKA_PLATFORM_CHOICES}"
  )
else()
  set(PIKA_PLATFORM
      "${PIKA_PLATFORM}"
      CACHE
        STRING
        "Sets special compilation flags for specific platforms. ${PIKA_PLATFORM_CHOICES}"
  )
endif()

if(NOT PIKA_PLATFORM STREQUAL "")
  string(TOUPPER ${PIKA_PLATFORM} PIKA_PLATFORM_UC)
else()
  set(PIKA_PLATFORM
      "native"
      CACHE
        STRING
        "Sets special compilation flags for specific platforms. ${PIKA_PLATFORM_CHOICES}"
        FORCE
  )
  set(PIKA_PLATFORM_UC "NATIVE")
endif()

string(FIND "${PIKA_PLATFORMS_UC}" "${PIKA_PLATFORM_UC}"
            _PLATFORM_FOUND
)
if(_PLATFORM_FOUND EQUAL -1)
  pika_error(
    "Unknown platform in PIKA_PLATFORM. ${PIKA_PLATFORM_CHOICES}"
  )
endif()
