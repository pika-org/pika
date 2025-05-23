
//  Copyright (c) 2011-2013 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(JACOBI_SMP_NO_pika)
# include <pika/init.hpp>
#endif

#include <pika/modules/program_options.hpp>

#include <boost/phoenix.hpp>
#include <boost/spirit/include/qi_action.hpp>
#include <boost/spirit/include/qi_auxiliary.hpp>
#include <boost/spirit/include/qi_char.hpp>
#include <boost/spirit/include/qi_numeric.hpp>
#include <boost/spirit/include/qi_operator.hpp>
#include <boost/spirit/include/qi_parse.hpp>
#include <boost/spirit/include/qi_string.hpp>
#include <boost/spirit/include/support_istream_iterator.hpp>

#include <cstddef>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using pika::program_options::options_description;
using pika::program_options::parse_command_line;
using pika::program_options::store;
using pika::program_options::value;
using pika::program_options::variables_map;

#include "jacobi_nonuniform.hpp"

namespace jacobi_smp {

    void jacobi_kernel_nonuniform(crs_matrix<double> const& A, std::vector<double>& dst,
        std::vector<double> const& src, std::vector<double> const& b, std::size_t row)
    {
        double result = b[row];
        double div = 1.0;
        std::size_t const begin = A.row_begin(row);
        std::size_t const end = A.row_end(row);

        for (std::size_t j = begin; j < end; ++j)
        {
            if (row == j)
                div = div / A.values[j];
            else
                result -= A.values[j] * src[A.indices[j]];
        }
        dst[row] = result * div;
    }
}    // namespace jacobi_smp

namespace qi = boost::spirit::qi;
namespace phx = boost::phoenix;

void init(jacobi_smp::crs_matrix<double>& M, std::size_t dim, std::size_t non_zeros)
{
    M.values.reserve(non_zeros);
    M.indices.reserve(non_zeros);
    M.rows.reserve(dim + 1);
    M.rows.push_back(0);
}

void add_entry(jacobi_smp::crs_matrix<double>& M, std::size_t& row, std::size_t& n, std::size_t j,
    std::size_t i, double v)
{
    M.values.push_back(v);
    M.indices.push_back(j);
    ++n;
    if (i != row)
    {
        row = i;
        M.rows.push_back(M.rows.back() + n);
        n = 0;
    }
}

int pika_main(variables_map& vm)
{
#if !defined(JACOBI_SMP_NO_pika)
    pika::scoped_finalize f;
#endif

    {
        std::size_t iterations = vm["iterations"].as<std::size_t>();
        std::size_t block_size = vm["block-size"].as<std::size_t>();
        std::string matrix = vm["matrix"].as<std::string>();
        std::string mode = vm["mode"].as<std::string>();

        jacobi_smp::crs_matrix<double> A;
        std::vector<double> x;
        std::vector<double> b;

        {
            std::ifstream file(matrix.c_str());
            file.unsetf(std::ios_base::skipws);
            using iterator = boost::spirit::istream_iterator;
            iterator begin(file);
            iterator end;

            std::size_t current_row = 0;
            std::size_t non_zero_row = 0;
            std::size_t dim = 0;
            std::size_t non_zero_entries = 0;

            qi::phrase_parse(begin, end,
                (qi::int_[phx::ref(dim) = qi::_1] >> qi::int_ >>
                    qi::int_[phx::ref(non_zero_entries) = qi::_1])[phx::bind(
                    init, phx::ref(A), dim, non_zero_entries)] >>
                    *(    // entry
                        (qi::int_ >> qi::int_ >>
                            qi::double_)[phx::bind(add_entry, phx::ref(A), phx::ref(current_row),
                            phx::ref(non_zero_row), qi::_1, qi::_2, qi::_3)]),
                qi::space | (qi::lit("%") >> *(!qi::eol >> qi::char_) >> qi::eol));

            if (dim == 0)
            {
                std::cerr << "Parsed zero non zero values in matrix file " << matrix << "\n";
                return 1;
            }

            std::cout << "A: " << dim << "x" << dim << " number of non zeros: " << non_zero_entries
                      << "\n";
        }
        if (vm.count("vector"))
        {
            std::string vector = vm["vector"].as<std::string>();
            std::ifstream file(vector.c_str());
            file.unsetf(std::ios_base::skipws);
            using iterator = boost::spirit::istream_iterator;
            iterator begin(file);
            iterator end;
            qi::phrase_parse(begin, end, *qi::double_, qi::ascii::space, b);
        }
        else { b = std::vector<double>(A.rows.size() - 1, 1.0); }
        std::cout << "b: " << b.size() << "\n";

        if (mode == "solve") { jacobi_smp::jacobi(A, b, iterations, block_size); }
        else if (mode == "statistics")
        {
            std::size_t min_per_row = (std::numeric_limits<std::size_t>::max)();
            std::size_t max_per_row = 0;
            double mean_per_row = 0.0;
            for (std::size_t r = 0; r < b.size(); ++r)
            {
                std::size_t const begin = A.row_begin(r);
                std::size_t const end = A.row_end(r);
                std::size_t n_row = end - begin;
                mean_per_row += double(n_row);
                if (n_row > max_per_row) { max_per_row = n_row; }
                if (n_row < min_per_row) { min_per_row = n_row; }
            }
            std::cout << "Matrix has " << A.values.size() << " non zero entries\n";
            std::cout << "order: " << b.size() << "x" << b.size() << "\n";
            std::cout << "Entries per row:\n";
            std::cout << "\tmax " << max_per_row << "\n";
            std::cout << "\tmin " << min_per_row << "\n";
            std::cout << "\tmean " << mean_per_row / double(b.size()) << "\n";
            std::cout << "Density is: " << double(A.values.size()) / double(b.size() * b.size())
                      << "\n";
        }
        else
        {
            std::cout << "Unknown mode " << mode << "\n";
            return 1;
        }
    }

    return 0;
}

#if !defined(PIKA_APPLICATION_STRING)
# define PIKA_APPLICATION_STRING "jacobi_nonuniform"
#endif

int main(int argc, char** argv)
{
    options_description desc_cmd("usage: " PIKA_APPLICATION_STRING " [options]");

    // clang-format off
    desc_cmd.add_options()
        ("iterations", value<std::size_t>()->default_value(1000),
         "Number of iterations")
        ("block-size", value<std::size_t>()->default_value(256),
         "Block size of the different chunks to calculate in parallel")
        ("matrix", value<std::string>(),
         "Filename of the input matrix (Matrix Market format)")
        ("vector", value<std::string>(), "Filename of the right hand side vector")
        ("mode", value<std::string>()->default_value("solve"),
        "Mode of the program, can be solve or statistics (default: solve)");
    // clang-format on

#if defined(JACOBI_SMP_NO_pika)
    variables_map vm;
    desc_cmd.add_options()("help", "This help message");
    store(parse_command_line(argc, argv, desc_cmd), vm);
    if (vm.count("help"))
    {
        std::cout << desc_cmd;
        return 1;
    }
    return pika_main(vm);
#else
    pika::init_params init_args;
    init_args.desc_cmdline = desc_cmd;

    return pika::init(pika_main, argc, argv, init_args);
#endif
}
