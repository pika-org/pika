// Copyright Vladimir Prus 2002-2004.
//  SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/program_options/config.hpp>

#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <pika/config/warnings_prefix.hpp>

namespace pika::program_options {

    inline std::string strip_prefixes(std::string const& text)
    {
        // "--foo-bar" -> "foo-bar"
        std::string::size_type i = text.find_first_not_of("-/");
        if (i == std::string::npos) { return text; }
        else { return text.substr(i); }
    }

    /** Base class for all errors in the library. */
    class PIKA_ALWAYS_EXPORT error : public std::logic_error
    {
    public:
        error(std::string const& xwhat)
          : std::logic_error(xwhat)
        {
        }
    };

    /** Class thrown when there are too many positional options.
        This is a programming error.
    */
    class PIKA_ALWAYS_EXPORT too_many_positional_options_error : public error
    {
    public:
        too_many_positional_options_error()
          : error("too many positional options have been specified on the command line")
        {
        }
    };

    /** Class thrown when there are programming error related to style */
    class PIKA_ALWAYS_EXPORT invalid_command_line_style : public error
    {
    public:
        invalid_command_line_style(std::string const& msg)
          : error(msg)
        {
        }
    };

    /** Class thrown if config file can not be read */
    class PIKA_ALWAYS_EXPORT reading_file : public error
    {
    public:
        reading_file(char const* filename)
          : error(std::string("can not read options configuration file '")
                      .append(filename)
                      .append("'"))
        {
        }
    };

    /** Base class for most exceptions in the library.
     *
     *  Substitutes the values for the parameter name
     *      placeholders in the template to create the human
     *      readable error message
     *
     *  Placeholders are surrounded by % signs: %example%
     *      Poor man's version of format
     *
     *  If a parameter name is absent, perform default substitutions
     *      instead so ugly placeholders are never left in-place.
     *
     *  Options are displayed in "canonical" form
     *      This is the most unambiguous form of the
     *      *parsed* option name and would correspond to
     *      option_description::format_name()
     *      i.e. what is shown by print_usage()
     *
     *  The "canonical" form depends on whether the option is
     *      specified in short or long form, using dashes or slashes
     *      or without a prefix (from a configuration file)
     *
     *   */
    class PIKA_ALWAYS_EXPORT error_with_option_name : public error
    {
    protected:
        /** can be
         *      0 = no prefix (config file options)
         *      allow_long
         *      allow_dash_for_short
         *      allow_slash_for_short
         *      allow_long_disguise */
        int m_option_style;

        /** substitutions
         *  from placeholders to values */
        std::map<std::string, std::string> m_substitutions;
        using string_pair = std::pair<std::string, std::string>;
        std::map<std::string, string_pair> m_substitution_defaults;

    public:
        /** template with placeholders */
        std::string m_error_template;

        error_with_option_name(std::string const& template_, std::string const& option_name = "",
            std::string const& original_token = "", int option_style = 0);

        /** gcc says that throw specification on dtor is loosened
         *  without this line
         *  */
        ~error_with_option_name() noexcept {}

        /** Substitute
         *      parameter_name->value to create the error message from
         *      the error template */
        void set_substitute(std::string const& parameter_name, std::string const& value)
        {
            m_substitutions[parameter_name] = value;
        }

        /** If the parameter is missing, then make the
         *      from->to substitution instead */
        void set_substitute_default(
            std::string const& parameter_name, std::string const& from, std::string const& to)
        {
            m_substitution_defaults[parameter_name] = std::make_pair(from, to);
        }

        /** Add context to an exception */
        // NOLINTBEGIN(bugprone-easily-swappable-parameters)
        void add_context(
            std::string const& option_name, std::string const& original_token, int option_style)
        // NOLINTEND(bugprone-easily-swappable-parameters)
        {
            set_option_name(option_name);
            set_original_token(original_token);
            set_prefix(option_style);
        }

        void set_prefix(int option_style) { m_option_style = option_style; }

        /** Overridden in error_with_no_option_name */
        virtual void set_option_name(std::string const& option_name)
        {
            set_substitute("option", option_name);
        }

        std::string get_option_name() const { return get_canonical_option_name(); }

        void set_original_token(std::string const& original_token)
        {
            set_substitute("original_token", original_token);
        }

        /** Creates the error_message on the fly
         *      Currently a thin wrapper for substitute_placeholders() */
        char const* what() const noexcept override;

    protected:
        /** Used to hold the error text returned by what() */
        mutable std::string m_message;    // For on-demand formatting in 'what'

        /** Makes all substitutions using the template */
        virtual void substitute_placeholders(std::string const& error_template) const;

        // helper function for substitute_placeholders
        void replace_token(std::string const& from, std::string const& to) const;

        /** Construct option name in accordance with the appropriate
         *  prefix style: i.e. long dash or short slash etc */
        std::string get_canonical_option_name() const;
        std::string get_canonical_option_prefix() const;
    };

    /** Class thrown when there are several option values, but
        user called a method which cannot return them all. */
    class PIKA_ALWAYS_EXPORT multiple_values : public error_with_option_name
    {
    public:
        multiple_values()
          : error_with_option_name("option '%canonical_option%' only takes a single argument")
        {
        }

        ~multiple_values() noexcept {}
    };

    /** Class thrown when there are several occurrences of an
        option, but user called a method which cannot return
        them all. */
    class PIKA_ALWAYS_EXPORT multiple_occurrences : public error_with_option_name
    {
    public:
        multiple_occurrences()
          : error_with_option_name("option '%canonical_option%' cannot be specified more than once")
        {
        }

        ~multiple_occurrences() noexcept {}
    };

    /** Class thrown when a required/mandatory option is missing */
    class PIKA_ALWAYS_EXPORT required_option : public error_with_option_name
    {
    public:
        // option name is constructed by the option_descriptor and never on the fly
        required_option(std::string const& option_name)
          : error_with_option_name(
                "the option '%canonical_option%' is required but missing", "", option_name)
        {
        }

        ~required_option() noexcept {}
    };

    /** Base class of un-parsable options,
     *  when the desired option cannot be identified.
     *
     *
     *  It makes no sense to have an option name, when we can't match an option to the
     *      parameter
     *
     *  Having this a part of the error_with_option_name hierarchy makes error
     *      handling a lot easier, even if the name indicates some sort of
     *      conceptual dissonance!
     *
     *   */
    class PIKA_ALWAYS_EXPORT error_with_no_option_name : public error_with_option_name
    {
    public:
        error_with_no_option_name(
            std::string const& template_, std::string const& original_token = "")
          : error_with_option_name(template_, "", original_token)
        {
        }

        /** Does NOT set option name, because no option name makes sense */
        void set_option_name(std::string const&) override {}

        ~error_with_no_option_name() noexcept {}
    };

    /** Class thrown when option name is not recognized. */
    class PIKA_ALWAYS_EXPORT unknown_option : public error_with_no_option_name
    {
    public:
        unknown_option(std::string const& original_token = "")
          : error_with_no_option_name("unrecognised option '%canonical_option%'", original_token)
        {
        }

        ~unknown_option() noexcept {}
    };

    /** Class thrown when there's ambiguity among several possible options. */
    class PIKA_ALWAYS_EXPORT ambiguous_option : public error_with_no_option_name
    {
    public:
        ambiguous_option(std::vector<std::string> const& xalternatives)
          : error_with_no_option_name("option '%canonical_option%' is ambiguous")
          , m_alternatives(xalternatives)
        {
        }

        ~ambiguous_option() noexcept {}

        std::vector<std::string> const& alternatives() const noexcept { return m_alternatives; }

    protected:
        /** Makes all substitutions using the template */
        void substitute_placeholders(std::string const& error_template) const override;

    private:
        // TODO: copy ctor might throw
        std::vector<std::string> m_alternatives;
    };

    /** Class thrown when there's syntax error either for command
     *  line or config file options. See derived children for
     *  concrete classes. */
    class PIKA_ALWAYS_EXPORT invalid_syntax : public error_with_option_name
    {
    public:
        enum kind_t
        {
            long_not_allowed = 30,
            long_adjacent_not_allowed,
            short_adjacent_not_allowed,
            empty_adjacent_parameter,
            missing_parameter,
            extra_parameter,
            unrecognized_line
        };

        invalid_syntax(kind_t kind, std::string const& option_name = "",
            std::string const& original_token = "", int option_style = 0)
          : error_with_option_name(get_template(kind), option_name, original_token, option_style)
          , m_kind(kind)
        {
        }

        ~invalid_syntax() noexcept {}

        kind_t kind() const { return m_kind; }

        /** Convenience functions for backwards compatibility */
        virtual std::string tokens() const { return get_option_name(); }

    protected:
        /** Used to convert kind_t to a related error text */
        std::string get_template(kind_t kind);
        kind_t m_kind;
    };

    class PIKA_ALWAYS_EXPORT invalid_config_file_syntax : public invalid_syntax
    {
    public:
        invalid_config_file_syntax(std::string const& invalid_line, kind_t kind)
          : invalid_syntax(kind)
        {
            m_substitutions["invalid_line"] = invalid_line;
        }

        ~invalid_config_file_syntax() noexcept {}

        /** Convenience functions for backwards compatibility */
        std::string tokens() const override
        {
            auto it = m_substitutions.find("invalid_line");
            if (it != m_substitutions.end()) { return it->second; }
            return "<unknown>";
        }
    };

    /** Class thrown when there are syntax errors in given command line */
    class PIKA_ALWAYS_EXPORT invalid_command_line_syntax : public invalid_syntax
    {
    public:
        invalid_command_line_syntax(kind_t kind, std::string const& option_name = "",
            std::string const& original_token = "", int option_style = 0)
          : invalid_syntax(kind, option_name, original_token, option_style)
        {
        }
        ~invalid_command_line_syntax() noexcept {}
    };

    /** Class thrown when value of option is incorrect. */
    class PIKA_ALWAYS_EXPORT validation_error : public error_with_option_name
    {
    public:
        enum kind_t
        {
            multiple_values_not_allowed = 30,
            at_least_one_value_required,
            invalid_bool_value,
            invalid_option_value,
            invalid_option
        };

    public:
        validation_error(kind_t kind, std::string const& option_name = "",
            std::string const& original_token = "", int option_style = 0)
          : error_with_option_name(get_template(kind), option_name, original_token, option_style)
          , m_kind(kind)
        {
        }

        ~validation_error() noexcept {}

        kind_t kind() const { return m_kind; }

    protected:
        /** Used to convert kind_t to a related error text */
        std::string get_template(kind_t kind);
        kind_t m_kind;
    };

    /** Class thrown if there is an invalid option value given */
    class PIKA_ALWAYS_EXPORT invalid_option_value : public validation_error
    {
    public:
        invalid_option_value(std::string const& value);
        invalid_option_value(std::wstring const& value);
    };

    /** Class thrown if there is an invalid bool value given */
    class PIKA_ALWAYS_EXPORT invalid_bool_value : public validation_error
    {
    public:
        invalid_bool_value(std::string const& value);
    };

}    // namespace pika::program_options

#include <pika/config/warnings_suffix.hpp>
