module;

#if !defined(PIKA_HAVE_MODULE)
#include <atomic>
#include <boost/version.hpp>
#include <cstddef>
#include <cstdint>
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <fmt/printf.h>
#include <fmt/std.h>
#include <mutex>
#include <ostream>
#include <type_traits>
#include <string>
#endif

export module pika.testing;

import std;
import pika.all;

export {
// TODO: Silence warning about including in module purview
#include <pika/testing/detail.hpp>
}
