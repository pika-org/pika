//  Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/pika_init.hpp>
#include <pika/iostream.hpp>

#include <pika/modules/program_options.hpp>

#include <string>
#include <vector>

int pika_main(pika::program_options::variables_map& vm)
{
    // extract value of application specific command line option
    int test = vm["test"].as<int>();
    pika::cout
        << "value for command line option --test: "
        << test << "\n";

    // extract all positional command line argument
    if (vm.count("pika:positional"))
    {
        std::vector<std::string> positional =
            vm["pika:positional"].as<std::vector<std::string> >();
        pika::cout << "positional command line options:\n";
        for (std::string const& arg : positional)
            pika::cout << arg << "\n";
    }
    else
    {
        pika::cout << "no positional command line options\n";
    }

    return pika::finalize();
}

int main(int argc, char* argv[])
{
    // Configure application-specific options.
    pika::program_options::options_description desc_commandline;

    desc_commandline.add_options()
        ("test",
         pika::program_options::value<int>()->default_value(42),
         "additional, application-specific option")
    ;

    pika::init_params init_args;
    init_args.desc_cmdline = desc_commandline;

    return pika::init(argc, argv, init_args);
}
