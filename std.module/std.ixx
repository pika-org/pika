// Copyright (c) Microsoft Corporation.
// Copyright (c) Daniela Engert
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

module;

#ifdef _BUILD_STD_MODULE

// std-gmf.hpp contains everything from
// the subset of "C headers" [tab:c.headers] corresponding to
// the "C++ headers for C library facilities" [tab:headers.cpp.c]
#include "std-gmf.hpp"

#endif

export module std;

#ifdef _MSC_VER
#  pragma comment(lib, "std.lib")
#endif

// allstd.hpp contains everything from
// "C++ library headers" [tab:headers.cpp]
// "C++ headers for C library facilities" [tab:headers.cpp.c]

#ifdef _BUILD_STD_MODULE

#include "allstd.hpp"

#else

export import "allstd.cpp";

#endif
