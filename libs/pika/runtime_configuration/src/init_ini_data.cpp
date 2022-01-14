//  Copyright (c) 2005-2017 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/config.hpp>
#include <pika/assert.hpp>
#include <pika/ini/ini.hpp>
#include <pika/local/version.hpp>
#include <pika/modules/errors.hpp>
#include <pika/modules/filesystem.hpp>
#include <pika/modules/logging.hpp>
#include <pika/prefix/find_prefix.hpp>
#include <pika/runtime_configuration/init_ini_data.hpp>

#include <boost/tokenizer.hpp>

#include <algorithm>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace pika { namespace util {
    ///////////////////////////////////////////////////////////////////////////
    bool handle_ini_file(section& ini, std::string const& loc)
    {
        try
        {
            namespace fs = filesystem;
            std::error_code ec;
            if (!fs::exists(loc, ec) || ec)
                return false;    // avoid exception on missing file
            ini.read(loc);
        }
        catch (pika::exception const& /*e*/)
        {
            return false;
        }
        return true;
    }

    ///////////////////////////////////////////////////////////////////////////
    bool handle_ini_file_env(
        section& ini, char const* env_var, char const* file_suffix)
    {
        char const* env = getenv(env_var);
        if (nullptr != env)
        {
            namespace fs = filesystem;

            fs::path inipath(env);
            if (nullptr != file_suffix)
                inipath /= fs::path(file_suffix);

            if (handle_ini_file(ini, inipath.string()))
            {
                LBT_(info).format("loaded configuration (${{{}}}): {}", env_var,
                    inipath.string());
                return true;
            }
        }
        return false;
    }

    ///////////////////////////////////////////////////////////////////////////
    // read system and user specified ini files
    //
    // returns true if at least one alternative location has been read
    // successfully
    bool init_ini_data_base(section& ini, std::string& pika_ini_file)
    {
        namespace fs = filesystem;

        // fall back: use compile time prefix
        std::string ini_paths(ini.get_entry("pika.master_ini_path"));
        std::string ini_paths_suffixes(
            ini.get_entry("pika.master_ini_path_suffixes"));

        // split off the separate paths from the given path list
        typedef boost::tokenizer<boost::char_separator<char>> tokenizer_type;

        boost::char_separator<char> sep(PIKA_INI_PATH_DELIMITER);
        tokenizer_type tok_paths(ini_paths, sep);
        tokenizer_type::iterator end_paths = tok_paths.end();
        tokenizer_type tok_suffixes(ini_paths_suffixes, sep);
        tokenizer_type::iterator end_suffixes = tok_suffixes.end();

        bool result = false;
        for (tokenizer_type::iterator it = tok_paths.begin(); it != end_paths;
             ++it)
        {
            for (tokenizer_type::iterator jt = tok_suffixes.begin();
                 jt != end_suffixes; ++jt)
            {
                std::string path = *it;
                path += *jt;
                bool result2 = handle_ini_file(ini, path + "/pika.ini");
                if (result2)
                {
                    LBT_(info).format("loaded configuration: {}/pika.ini", path);
                }
                result = result2 || result;
            }
        }

        // look in the current directory first
        std::string cwd = fs::current_path().string() + "/.pika.ini";
        {
            bool result2 = handle_ini_file(ini, cwd);
            if (result2)
            {
                LBT_(info).format("loaded configuration: {}", cwd);
            }
            result = result2 || result;
        }

        // look for master ini in the PIKA_INI environment
        result = handle_ini_file_env(ini, "PIKA_INI") || result;

        // afterwards in the standard locations
#if !defined(PIKA_WINDOWS)    // /etc/pika.ini doesn't make sense for Windows
        {
            bool result2 = handle_ini_file(ini, "/etc/pika.ini");
            if (result2)
            {
                LBT_(info).format("loaded configuration: /etc/pika.ini");
            }
            result = result2 || result;
        }
#endif

        result = handle_ini_file_env(ini, "HOME", ".pika.ini") || result;
        result = handle_ini_file_env(ini, "PWD", ".pika.ini") || result;

        if (!pika_ini_file.empty())
        {
            namespace fs = filesystem;
            std::error_code ec;
            if (!fs::exists(pika_ini_file, ec) || ec)
            {
                std::cerr
                    << "pika::init: command line warning: file specified using "
                       "--pika:config does not exist ("
                    << pika_ini_file << ")." << std::endl;
                pika_ini_file.clear();
                result = false;
            }
            else
            {
                bool result2 = handle_ini_file(ini, pika_ini_file);
                if (result2)
                {
                    LBT_(info).format("loaded configuration: {}", pika_ini_file);
                }
                return result || result2;
            }
        }
        return result;
    }

    ///////////////////////////////////////////////////////////////////////////
    // global function to read component ini information
    void merge_component_inis(section& ini)
    {
        namespace fs = filesystem;

        // now merge all information into one global structure
        std::string ini_path(ini.get_entry("pika.ini_path"));
        std::vector<std::string> ini_paths;

        // split off the separate paths from the given path list
        typedef boost::tokenizer<boost::char_separator<char>> tokenizer_type;

        boost::char_separator<char> sep(PIKA_INI_PATH_DELIMITER);
        tokenizer_type tok(ini_path, sep);
        tokenizer_type::iterator end = tok.end();
        for (tokenizer_type::iterator it = tok.begin(); it != end; ++it)
            ini_paths.push_back(*it);

        // have all path elements, now find ini files in there...
        std::vector<std::string>::iterator ini_end = ini_paths.end();
        for (std::vector<std::string>::iterator it = ini_paths.begin();
             it != ini_end; ++it)
        {
            try
            {
                fs::directory_iterator nodir;
                fs::path this_path(*it);

                std::error_code ec;
                if (!fs::exists(this_path, ec) || ec)
                    continue;

                for (fs::directory_iterator dir(this_path); dir != nodir; ++dir)
                {
                    if (dir->path().extension() != ".ini")
                        continue;

                    // read and merge the ini file into the main ini hierarchy
                    try
                    {
                        ini.merge(dir->path().string());
                        LBT_(info).format(
                            "loaded configuration: {}", dir->path().string());
                    }
                    catch (pika::exception const& /*e*/)
                    {
                        ;
                    }
                }
            }
            catch (fs::filesystem_error const& /*e*/)
            {
                ;
            }
        }
    }

    namespace detail {
        inline bool cmppath_less(
            std::pair<filesystem::path, std::string> const& lhs,
            std::pair<filesystem::path, std::string> const& rhs)
        {
            return lhs.first < rhs.first;
        }

        inline bool cmppath_equal(
            std::pair<filesystem::path, std::string> const& lhs,
            std::pair<filesystem::path, std::string> const& rhs)
        {
            return lhs.first == rhs.first;
        }
    }    // namespace detail
}}    // namespace pika::util
