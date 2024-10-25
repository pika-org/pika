// Copyright Sascha Ochsenknecht 2009.
//  SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/program_options/config.hpp>
#include <pika/program_options/parsers.hpp>

#include <boost/tokenizer.hpp>

#include <string>
#include <vector>

namespace pika::program_options::detail {

    template <class Char>
    std::vector<std::basic_string<Char>>
    split_unix(std::basic_string<Char> const& cmdline, std::basic_string<Char> const& separator,
        std::basic_string<Char> const& quote, std::basic_string<Char> const& escape)
    {
        using tokenizerT = boost::tokenizer<boost::escaped_list_separator<Char>,
            typename std::basic_string<Char>::const_iterator, std::basic_string<Char>>;

        tokenizerT tok(cmdline.begin(), cmdline.end(),
            boost::escaped_list_separator<Char>(escape, separator, quote));

        std::vector<std::basic_string<Char>> result;
        for (typename tokenizerT::iterator cur_token(tok.begin()), end_token(tok.end());
             cur_token != end_token; ++cur_token)
        {
            if (!cur_token->empty()) result.push_back(*cur_token);
        }
        return result;
    }

}    // namespace pika::program_options::detail

namespace pika::program_options {

    // Take a command line string and splits in into tokens, according
    // to the given collection of separators chars.
    std::vector<std::string> split_unix(std::string const& cmdline, std::string const& separator,
        std::string const& quote, std::string const& escape)
    {
        return detail::split_unix<char>(cmdline, separator, quote, escape);
    }

    std::vector<std::wstring> split_unix(std::wstring const& cmdline, std::wstring const& separator,
        std::wstring const& quote, std::wstring const& escape)
    {
        return detail::split_unix<wchar_t>(cmdline, separator, quote, escape);
    }

}    // namespace pika::program_options
