// Copyright (c) Microsoft Corporation.
// Copyright (c) Daniela Engert
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// "C++ library headers" [tab:headers.cpp]
#include <algorithm>
#include <any>
#include <array>
#include <atomic>
#include <barrier>
#include <bit>
#include <bitset>
#include <charconv>
#include <chrono>
#include <codecvt>
#include <compare>
#include <complex>
#include <concepts>
#include <condition_variable>
#include <coroutine>
#include <deque>
#include <exception>
#include <execution>
#if !defined(__GLIBCXX__) or !defined(__clang_major__) or __clang_major__ > 17
#   include <expected>
#endif
#include <filesystem>
#include <format>
#include <forward_list>
#include <fstream>
#include <functional>
#include <future>
#if __has_include(<generator>)
#   include <generator>
#endif
#include <initializer_list>
#include <iomanip>
#include <ios>
#include <iosfwd>
#include <iostream>
#include <istream>
#include <iterator>
#include <latch>
#include <limits>
#include <list>
#include <locale>
#include <map>
#include <memory>
#include <memory_resource>
#include <mutex>
#include <new>
#include <numbers>
#include <numeric>
#include <optional>
#include <ostream>
#if __has_include(<print>)
#   include <print>
#endif
#include <queue>
#include <random>
#if defined(__GLIBCXX__) and defined(__clang_major__) and __clang_major__ == 16
#   include "libstdc++-ranges-13.1-for-Clang16.0-bugs"
#else
#   include <ranges>
#endif
#include <ratio>
#include <regex>
#include <scoped_allocator>
#include <semaphore>
#include <set>
#include <shared_mutex>
#include <source_location>
#include <span>
#if __has_include(<spanstream>)
#   include <spanstream>
#endif
#include <sstream>
#include <stack>
#if __has_include(<stacktrace>)
#   include <stacktrace>
#endif
#include <stdexcept>
#include <stop_token>
#include <streambuf>
#include <string>
#include <string_view>
// #include <strstream> deprecated in C++98!  
#if __has_include(<syncstream>)
#   include <syncstream>
#endif
#include <system_error>
#include <thread>
#include <tuple>
#include <type_traits>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <valarray>
#include <variant>
#include <vector>
#include <version>

// "C++ headers for C library facilities" [tab:headers.cpp.c]
#include <cassert>
#include <cctype>
#include <cerrno>
#include <cfenv>
#include <cfloat>
#include <cinttypes>
#include <climits>
#include <clocale>
#include <cmath>
#include <csetjmp>
#include <csignal>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cuchar>
#include <cwchar>
#include <cwctype>

#if not defined(__cpp_lib_start_lifetime_as) \
 or not defined(__cpp_lib_print)             \
 or not defined(__cpp_lib_generator)
#   include "polyfill.hpp"
#endif

#if defined(__GLIBCXX__) and defined(__clang_major__) and __clang_major__ <= 17
// Clang 16.0 has P0848 but doesn't advertise it
// libstdc++ gates std::expected on __cpp_concepts from P2493
# if __cpp_concepts < 202202L
#   undef __cpp_concepts
#   define __cpp_concepts 202202L
# endif
#endif
#include <expected>

// Third party libraries and C headers
#include <errno.h>
#include <sys/mman.h>
#include <sys/param.h>
#include <cxxabi.h>
#include <execinfo.h>
#include <hwloc.h>
#include <dlfcn.h>
#include <boost/version.hpp>
#include <boost/crc.hpp>
#include <boost/config.hpp>
#include <boost/container/small_vector.hpp>
#include <boost/context/detail/fcontext.hpp>
#include <boost/dynamic_bitset.hpp>
#include <boost/intrusive/slist.hpp>
#include <boost/lockfree/queue.hpp>
#include <boost/optional.hpp>
#include <boost/tokenizer.hpp>
#include <fmt/base.h>
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <fmt/printf.h>
