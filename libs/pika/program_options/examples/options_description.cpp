// Copyright Vladimir Prus 2002-2004.
//  SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/modules/program_options.hpp>

#include <algorithm>
#include <exception>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

namespace po = pika::program_options;
using namespace std;

// A helper function to simplify the main part.
template <class T>
ostream& operator<<(ostream& os, vector<T> const& v)
{
    copy(v.begin(), v.end(), ostream_iterator<T>(os, " "));
    return os;
}

int main(int ac, char* av[])
{
    try
    {
        int opt;
        int portnum;
        po::options_description desc("Allowed options");
        // clang-format off
        desc.add_options()
            ("help", "produce help message")
            ("optimization", po::value<int>(&opt)->default_value(10),
                  "optimization level")
            ("verbose,v", po::value<int>()->implicit_value(1),
                  "enable verbosity (optionally specify level)")
            ("listen,l", po::value<int>(&portnum)->implicit_value(1001)
                  ->default_value(0,"no"),
                  "listen on a port.")
            ("include-path,I", po::value< vector<string> >(),
                  "include path")
            ("input-file", po::value< vector<string> >(), "input file")
        ;
        // clang-format on

        po::positional_options_description p;
        p.add("input-file", -1);

        po::variables_map vm;
        po::store(
            po::command_line_parser(ac, av).allow_unregistered().options(desc).positional(p).run(),
            vm);
        po::notify(vm);

        if (vm.count("help"))
        {
            cout << "Usage: options_description [options]\n";
            cout << desc;
            return 0;
        }

        if (vm.count("include-path"))
        {
            cout << "Include paths are: " << vm["include-path"].as<vector<string>>() << "\n";
        }

        if (vm.count("input-file"))
        {
            cout << "Input files are: " << vm["input-file"].as<vector<string>>() << "\n";
        }

        if (vm.count("verbose"))
        {
            cout << "Verbosity enabled.  Level is " << vm["verbose"].as<int>() << "\n";
        }

        cout << "Optimization level is " << opt << "\n";
        cout << "Listen port is " << portnum << "\n";
    }
    catch (std::exception const& e)
    {
        cout << e.what() << "\n";
        return 1;
    }
    return 0;
}
