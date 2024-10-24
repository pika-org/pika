// Copyright Vladimir Prus 2002-2004.
//  SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/program_options/config.hpp>
#include <pika/debugging/environ.hpp>
#include <pika/program_options/detail/cmdline.hpp>
#include <pika/program_options/detail/config_file.hpp>
#include <pika/program_options/detail/convert.hpp>
#include <pika/program_options/environment_iterator.hpp>
#include <pika/program_options/options_description.hpp>
#include <pika/program_options/parsers.hpp>
#include <pika/program_options/positional_options.hpp>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <functional>
#include <string>
#include <utility>

#ifdef _WIN32
# include <stdlib.h>
#else
# include <unistd.h>
#endif

using namespace std;

namespace pika::program_options {

    namespace {

        woption woption_from_option(option const& opt)
        {
            woption result;
            result.string_key = opt.string_key;
            result.position_key = opt.position_key;
            result.unregistered = opt.unregistered;

            std::transform(opt.value.begin(), opt.value.end(), back_inserter(result.value),
                std::bind(from_utf8, std::placeholders::_1));

            std::transform(opt.original_tokens.begin(), opt.original_tokens.end(),
                back_inserter(result.original_tokens), std::bind(from_utf8, std::placeholders::_1));
            return result;
        }
    }    // namespace

    basic_parsed_options<wchar_t>::basic_parsed_options(parsed_options const& po)
      : description(po.description)
      , utf8_encoded_options(po)
      , m_options_prefix(po.m_options_prefix)
    {
        for (auto const& option : po.options) options.push_back(woption_from_option(option));
    }

    template <class Char>
    basic_parsed_options<Char> parse_config_file(
        std::basic_istream<Char>& is, options_description const& desc, bool allow_unregistered)
    {
        set<string> allowed_options;

        vector<shared_ptr<option_description>> const& options = desc.options();
        for (auto const& option : options)
        {
            option_description const& d = *option;

            if (d.long_name().empty())
                throw error(
                    "abbreviated option names are not permitted in options configuration files");

            allowed_options.insert(d.long_name());
        }

        // Parser return char strings
        parsed_options result(&desc);
        copy(detail::basic_config_file_iterator<Char>(is, allowed_options, allow_unregistered),
            detail::basic_config_file_iterator<Char>(), back_inserter(result.options));
        // Convert char strings into desired type.
        return basic_parsed_options<Char>(result);
    }

    template PIKA_EXPORT basic_parsed_options<char> parse_config_file(
        std::basic_istream<char>& is, options_description const& desc, bool allow_unregistered);

    template PIKA_EXPORT basic_parsed_options<wchar_t> parse_config_file(
        std::basic_istream<wchar_t>& is, options_description const& desc, bool allow_unregistered);

    template <class Char>
    basic_parsed_options<Char> parse_config_file(
        char const* filename, options_description const& desc, bool allow_unregistered)
    {
        // Parser return char strings
        std::basic_ifstream<Char> strm(filename);
        if (!strm) { throw reading_file(filename); }

        basic_parsed_options<Char> result = parse_config_file(strm, desc, allow_unregistered);

        if (strm.bad()) { throw reading_file(filename); }

        return result;
    }

    template PIKA_EXPORT basic_parsed_options<char> parse_config_file(
        char const* filename, options_description const& desc, bool allow_unregistered);

    template PIKA_EXPORT basic_parsed_options<wchar_t> parse_config_file(
        char const* filename, options_description const& desc, bool allow_unregistered);

    parsed_options parse_environment(
        options_description const& desc, std::function<std::string(std::string)> const& name_mapper)
    {
        parsed_options result(&desc);

#if defined(__FreeBSD__)
        char** env = freebsd_environ;
#else
        char** env = environ;
#endif
        for (environment_iterator i(env), e; i != e; ++i)
        {
            string option_name = name_mapper(i->first);

            if (!option_name.empty())
            {
                option n;
                n.string_key = PIKA_MOVE(option_name);
                n.value.push_back(i->second);
                result.options.push_back(n);
            }
        }

        return result;
    }

    namespace detail {

        class prefix_name_mapper
        {
        public:
            prefix_name_mapper(std::string const& prefix)
              : prefix(prefix)
            {
            }

            std::string operator()(std::string const& s)
            {
                string result;
                if (s.find(prefix) == 0)
                {
                    for (string::size_type n = prefix.size(); n < s.size(); ++n)
                    {
                        // Intel-Win-7.1 does not understand
                        // push_back on string.
                        result += static_cast<char>(tolower(s[n]));
                    }
                }
                return result;
            }

        private:
            std::string prefix;
        };
    }    // namespace detail

    parsed_options parse_environment(options_description const& desc, std::string const& prefix)
    {
        return parse_environment(desc, detail::prefix_name_mapper(prefix));
    }

    parsed_options parse_environment(options_description const& desc, char const* prefix)
    {
        return parse_environment(desc, string(prefix));
    }

}    // namespace pika::program_options
