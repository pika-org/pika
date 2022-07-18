//  Copyright (c) 2019-2020 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/config.hpp>
#include <pika/debugging/environ.hpp>
#include <pika/debugging/print.hpp>

#include <boost/crc.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <bitset>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(__FreeBSD__)
PIKA_EXPORT char** freebsd_environ = nullptr;
#endif

// ------------------------------------------------------------
/// \cond NODETAIL
namespace NS_DEBUG {

    // ------------------------------------------------------------------
    // format as zero padded int
    // ------------------------------------------------------------------
    template <typename Int>
    PIKA_EXPORT void print_dec(std::ostream& os, Int const& v, int N)
    {
        os << std::right << std::setfill('0') << std::setw(N) << std::noshowbase
           << std::dec << v;
    }

    template PIKA_EXPORT void print_dec(std::ostream&, bool const&, int);
    template PIKA_EXPORT void print_dec(
        std::ostream&, std::int16_t const&, int);
    template PIKA_EXPORT void print_dec(
        std::ostream&, std::uint16_t const&, int);
    template PIKA_EXPORT void print_dec(
        std::ostream&, std::int32_t const&, int);
    template PIKA_EXPORT void print_dec(
        std::ostream&, std::uint32_t const&, int);
    template PIKA_EXPORT void print_dec(
        std::ostream&, std::int64_t const&, int);
    template PIKA_EXPORT void print_dec(
        std::ostream&, std::uint64_t const&, int);

    template PIKA_EXPORT void print_dec(
        std::ostream&, std::atomic<int> const&, int);
    template PIKA_EXPORT void print_dec(
        std::ostream&, std::atomic<unsigned int> const&, int);
    template PIKA_EXPORT void print_dec(
        std::ostream&, std::atomic<unsigned long> const&, int);

    // ------------------------------------------------------------------
    // format as pointer
    // ------------------------------------------------------------------
    ptr::ptr(void const* v)
      : data_(v)
    {
    }

    ptr::ptr(std::uintptr_t v)
      : data_(reinterpret_cast<void const*>(v))
    {
    }

    std::ostream& operator<<(std::ostream& os, ptr const& d)
    {
        os << std::internal << std::hex << std::setw(10) << std::setfill('0')
           << d.data_;
        return os;
    }

    // ------------------------------------------------------------------
    // format as zero padded hex
    // ------------------------------------------------------------------
    template <typename Int>
    void print_hex(std::ostream& os, Int v, int N)
    {
        os << std::right << "0x" << std::setfill('0') << std::setw(N)
           << std::noshowbase << std::hex << v;
    }

    template PIKA_EXPORT void print_hex(std::ostream&, std::thread::id, int);
    template PIKA_EXPORT void print_hex(std::ostream&, unsigned long, int);
    template PIKA_EXPORT void print_hex(std::ostream&, long, int);
    template PIKA_EXPORT void print_hex(std::ostream&, int, int);
    template PIKA_EXPORT void print_hex(std::ostream&, unsigned int, int);
    template PIKA_EXPORT void print_hex(std::ostream&, void*, int);

    template <typename Int>
    void print_ptr(std::ostream& os, Int v, int N)
    {
        os << std::right << std::setw(N) << std::noshowbase << std::hex << v;
    }

    template PIKA_EXPORT void print_ptr(std::ostream&, void*, int);
    template PIKA_EXPORT void print_ptr(std::ostream&, int, int);
    template PIKA_EXPORT void print_ptr(std::ostream&, long, int);

    // ------------------------------------------------------------------
    // format as binary bits
    // ------------------------------------------------------------------
    template <typename Int>
    PIKA_EXPORT void print_bin(std::ostream& os, Int v, int N)
    {
        char const* beg = reinterpret_cast<char const*>(&v);
        char const* end = beg + sizeof(v);

        N = (N + CHAR_BIT - 1) / CHAR_BIT;
        while (beg != end && N-- > 0)
        {
            os << std::bitset<CHAR_BIT>(*beg++);
        }
    }

    template PIKA_EXPORT void print_bin(std::ostream&, unsigned char, int);
    template PIKA_EXPORT void print_bin(std::ostream&, std::uint32_t, int);
    template PIKA_EXPORT void print_bin(std::ostream&, std::uint64_t, int);

#if defined(__APPLE__)
    // Explicit instantiation necessary to solve undefined symbol for MacOS
    template PIKA_EXPORT void print_bin(std::ostream&, unsigned long, int);
#endif

    // ------------------------------------------------------------------
    // format as padded string
    // ------------------------------------------------------------------
    void print_str(std::ostream& os, char const* v, int N)
    {
        os << std::left << std::setfill(' ') << std::setw(N) << v;
    }

    // ------------------------------------------------------------------
    // format as ip address
    // ------------------------------------------------------------------
    ipaddr::ipaddr(void const* a)
      : data_(reinterpret_cast<std::uint8_t const*>(a))
      , ipdata_(0)
    {
    }

    ipaddr::ipaddr(std::uint32_t a)
      : data_(reinterpret_cast<const uint8_t*>(&ipdata_))
      , ipdata_(a)
    {
    }

    std::ostream& operator<<(std::ostream& os, ipaddr const& p)
    {
        os << std::dec << int(p.data_[0]) << "." << int(p.data_[1]) << "."
           << int(p.data_[2]) << "." << int(p.data_[3]);
        return os;
    }

    // ------------------------------------------------------------------
    // helper class for printing time since start
    // ------------------------------------------------------------------
    std::ostream& operator<<(std::ostream& os, current_time_print_helper const&)
    {
        static std::chrono::steady_clock::time_point log_t_start =
            std::chrono::steady_clock::now();

        auto now = std::chrono::steady_clock::now();
        auto nowt = std::chrono::duration_cast<std::chrono::microseconds>(
            now - log_t_start)
                        .count();

        os << dec<10>(nowt) << " ";
        return os;
    }

    ///////////////////////////////////////////////////////////////////////////
    std::function<void(std::ostream&)> print_info;

    void register_print_info(void (*printer)(std::ostream&))
    {
        print_info = printer;
    }

    void generate_prefix(std::ostream& os)
    {
        os << detail::current_time_print_helper();
        if (print_info)
        {
            print_info(os);
        }
        os << detail::hostname_print_helper();
    }

    // ------------------------------------------------------------------
    // helper function for printing short memory dump and crc32
    // useful for debugging corruptions in buffers during
    // rma or other transfers
    // ------------------------------------------------------------------
    std::uint32_t crc32(void const* ptr, std::size_t size)
    {
        boost::crc_32_type result;
        result.process_bytes(ptr, size);
        return result.checksum();
        return 0;
    }

    mem_crc32::mem_crc32(void const* a, std::size_t len)
      : addr_(reinterpret_cast<const uint64_t*>(a))
      , len_(len)
    {
    }

    std::ostream& operator<<(std::ostream& os, mem_crc32 const& p)
    {
        std::uint64_t const* uintBuf =
            static_cast<std::uint64_t const*>(p.addr_);
        os << "Memory:";
        os << " address " << ptr(p.addr_) << " length " << hex<6>(p.len_)
           << " CRC32:" << hex<8>(detail::crc32(p.addr_, p.len_)) << "\n";

        for (std::size_t i = 0; i <
             (std::min)(size_t(std::ceil(static_cast<double>(p.len_) / 8.0)),
                 std::size_t(128));
             i++)
        {
            os << hex<16>(*uintBuf++) << " ";
        }
        return os;
    }

    // ------------------------------------------------------------------
    // helper class for printing time since start
    // ------------------------------------------------------------------
    char const* hostname_print_helper::get_hostname() const
    {
        static bool initialized = false;
        static char hostname_[20] = {'\0'};
        if (!initialized)
        {
            initialized = true;
#if !defined(__FreeBSD__)
            gethostname(hostname_, std::size_t(12));
#endif
            std::ostringstream temp;
            temp << '(' << std::to_string(guess_rank()) << ')';
            std::strcat(hostname_, temp.str().c_str());
        }
        return hostname_;
    }

    int hostname_print_helper::guess_rank() const
    {
#if defined(__FreeBSD__)
        char** env = freebsd_environ;
#else
        char** env = environ;
#endif
        std::vector<std::string> env_strings{"_RANK=", "_NODEID="};
        for (char** current = env; *current; current++)
        {
            auto e = std::string(*current);
            for (auto s : env_strings)
            {
                auto pos = e.find(s);
                if (pos != std::string::npos)
                {
                    //std::cout << "Got a rank string : " << e << std::endl;
                    return std::stoi(e.substr(pos + s.size(), 5));
                }
            }
        }
        return -1;
    }

    std::ostream& operator<<(std::ostream& os, hostname_print_helper const& h)
    {
        os << str<13>(h.get_hostname()) << " ";
        return os;
    }

    ///////////////////////////////////////////////////////////////////////
    template <typename T>
    PIKA_EXPORT void print_array(
        std::string const& name, T const* data, std::size_t size)
    {
        std::cout << str<20>(name.c_str()) << ": {" << dec<4>(size) << "} : ";
        std::copy(data, data + size, std::ostream_iterator<T>(std::cout, ", "));
        std::cout << "\n";
    }

    template PIKA_EXPORT void print_array(
        std::string const&, std::size_t const*, std::size_t);
}    // namespace NS_DEBUG
/// \endcond
