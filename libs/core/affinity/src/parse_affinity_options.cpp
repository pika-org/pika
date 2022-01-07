//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/affinity/parse_affinity_options.hpp>
#include <hpx/assert.hpp>
#include <hpx/datastructures/tuple.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/topology/topology.hpp>

#include <hwloc.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threads { namespace detail {
    ///////////////////////////////////////////////////////////////////////////
    void parse_mappings(
        std::string const& spec, mappings_type& mappings, error_code& ec)
    {
        if (spec == "compact")
        {
            mappings = compact;
        }
        else if (spec == "scatter")
        {
            mappings = scatter;
        }
        else if (spec == "balanced")
        {
            mappings = balanced;
        }
        else if (spec == "numa-balanced")
        {
            mappings = numa_balanced;
        }
        else
        {
            HPX_THROWS_IF(ec, bad_parameter, "parse_affinity_options",
                "failed to parse affinity specification: \"" + spec + "\"");
        }

        if (&ec != &throws)
            ec = make_success_code();
    }

    ///////////////////////////////////////////////////////////////////////////
    //                  index,       mask
    typedef hpx::tuple<std::size_t, mask_type> mask_info;

    inline std::size_t get_index(mask_info const& smi)
    {
        return hpx::get<0>(smi);
    }
    inline mask_cref_type get_mask(mask_info const& smi)
    {
        return hpx::get<1>(smi);
    }

    ///////////////////////////////////////////////////////////////////////////
    std::vector<mask_info> extract_socket_masks(
        topology const& t, bounds_type const& b)
    {
        std::vector<mask_info> masks;
        for (std::int64_t index : b)
        {
            masks.push_back(hpx::make_tuple(static_cast<std::size_t>(index),
                t.init_socket_affinity_mask_from_socket(
                    static_cast<std::size_t>(index))));
        }
        return masks;
    }

    std::vector<mask_info> extract_numanode_masks(
        topology const& t, bounds_type const& b)
    {
        std::vector<mask_info> masks;
        for (std::int64_t index : b)
        {
            masks.push_back(hpx::make_tuple(static_cast<std::size_t>(index),
                t.init_numa_node_affinity_mask_from_numa_node(
                    static_cast<std::size_t>(index))));
        }
        return masks;
    }

    mask_cref_type extract_machine_mask(topology const& t, error_code& ec)
    {
        return t.get_machine_affinity_mask(ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    bool pu_in_process_mask(bool use_process_mask, topology& t,
        std::size_t num_core, std::size_t num_pu)
    {
        if (!use_process_mask)
        {
            return true;
        }

        threads::mask_type proc_mask = t.get_cpubind_mask();
        threads::mask_type pu_mask =
            t.init_thread_affinity_mask(num_core, num_pu);

        return threads::bit_and(proc_mask, pu_mask);
    }

    void check_num_threads(bool use_process_mask, topology& t,
        std::size_t num_threads, error_code& ec)
    {
        if (use_process_mask)
        {
            threads::mask_type proc_mask = t.get_cpubind_mask();
            std::size_t num_pus_proc_mask = threads::count(proc_mask);

            if (num_threads > num_pus_proc_mask)
            {
                HPX_THROWS_IF(ec, bad_parameter, "check_num_threads",
                    "specified number of threads ({1}) is larger than number "
                    "of processing units available in process mask ({2})",
                    num_threads, num_pus_proc_mask);
            }
        }
        else
        {
            std::size_t num_threads_available = threads::hardware_concurrency();

            if (num_threads > num_threads_available)
            {
                HPX_THROWS_IF(ec, bad_parameter, "check_num_threads",
                    "specified number of threads ({1}) is larger than number "
                    "of available processing units ({2})",
                    num_threads, num_threads_available);
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    void decode_compact_distribution(topology& t,
        std::vector<mask_type>& affinities, std::size_t used_cores,
        std::size_t max_cores, std::vector<std::size_t>& num_pus,
        bool use_process_mask, error_code& ec)
    {
        std::size_t num_threads = affinities.size();

        check_num_threads(use_process_mask, t, num_threads, ec);

        if (use_process_mask)
        {
            used_cores = 0;
            max_cores = t.get_number_of_cores();
        }

        std::size_t num_cores = (std::min)(max_cores, t.get_number_of_cores());
        num_pus.resize(num_threads);

        for (std::size_t num_thread = 0; num_thread < num_threads; /**/)
        {
            for (std::size_t num_core = 0; num_core < num_cores; ++num_core)
            {
                std::size_t num_core_pus =
                    t.get_number_of_core_pus(num_core + used_cores);
                for (std::size_t num_pu = 0; num_pu < num_core_pus; ++num_pu)
                {
                    if (!pu_in_process_mask(
                            use_process_mask, t, num_core, num_pu))
                    {
                        continue;
                    }

                    if (any(affinities[num_thread]))
                    {
                        HPX_THROWS_IF(ec, bad_parameter,
                            "decode_compact_distribution",
                            "affinity mask for thread {1} has already been set",
                            num_thread);
                        return;
                    }

                    num_pus[num_thread] =
                        t.get_pu_number(num_core + used_cores, num_pu);
                    affinities[num_thread] = t.init_thread_affinity_mask(
                        num_core + used_cores, num_pu);

                    if (++num_thread == num_threads)
                        return;
                }
            }
        }
    }

    void decode_scatter_distribution(topology& t,
        std::vector<mask_type>& affinities, std::size_t used_cores,
        std::size_t max_cores, std::vector<std::size_t>& num_pus,
        bool use_process_mask, error_code& ec)
    {
        std::size_t num_threads = affinities.size();

        check_num_threads(use_process_mask, t, num_threads, ec);

        if (use_process_mask)
        {
            used_cores = 0;
            max_cores = t.get_number_of_cores();
        }

        std::size_t num_cores = (std::min)(max_cores, t.get_number_of_cores());

        std::vector<std::size_t> next_pu_index(num_cores, 0);
        num_pus.resize(num_threads);

        for (std::size_t num_thread = 0; num_thread < num_threads; /**/)
        {
            for (std::size_t num_core = 0; num_core < num_cores; ++num_core)
            {
                if (any(affinities[num_thread]))
                {
                    HPX_THROWS_IF(ec, bad_parameter,
                        "decode_scatter_distribution",
                        "affinity mask for thread {1} has already been set",
                        num_thread);
                    return;
                }

                std::size_t num_core_pus = t.get_number_of_core_pus(num_core);
                std::size_t pu_index = next_pu_index[num_core];
                bool use_pu = false;

                // Find the next PU on this core which is in the process mask
                while (pu_index < num_core_pus)
                {
                    use_pu = pu_in_process_mask(
                        use_process_mask, t, num_core, pu_index);
                    ++pu_index;

                    if (use_pu)
                    {
                        break;
                    }
                }

                next_pu_index[num_core] = pu_index;

                if (!use_pu)
                {
                    continue;
                }

                num_pus[num_thread] = t.get_pu_number(
                    num_core + used_cores, next_pu_index[num_core] - 1);
                affinities[num_thread] = t.init_thread_affinity_mask(
                    num_core + used_cores, next_pu_index[num_core] - 1);

                if (++num_thread == num_threads)
                    return;
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    void decode_balanced_distribution(topology& t,
        std::vector<mask_type>& affinities, std::size_t used_cores,
        std::size_t max_cores, std::vector<std::size_t>& num_pus,
        bool use_process_mask, error_code& ec)
    {
        std::size_t num_threads = affinities.size();

        check_num_threads(use_process_mask, t, num_threads, ec);

        if (use_process_mask)
        {
            used_cores = 0;
            max_cores = t.get_number_of_cores();
        }

        std::size_t num_cores = (std::min)(max_cores, t.get_number_of_cores());

        std::vector<std::size_t> num_pus_cores(num_cores, 0);
        std::vector<std::size_t> next_pu_index(num_cores, 0);
        std::vector<std::vector<std::size_t>> pu_indexes(num_cores);
        num_pus.resize(num_threads);

        // At first, calculate the number of used pus per core.
        // This needs to be done to make sure that we occupy all the available
        // cores
        for (std::size_t num_thread = 0; num_thread < num_threads; /**/)
        {
            for (std::size_t num_core = 0; num_core < num_cores; ++num_core)
            {
                std::size_t num_core_pus = t.get_number_of_core_pus(num_core);
                std::size_t pu_index = next_pu_index[num_core];
                bool use_pu = false;

                // Find the next PU on this core which is in the process mask
                while (pu_index < num_core_pus)
                {
                    use_pu = pu_in_process_mask(
                        use_process_mask, t, num_core, pu_index);
                    ++pu_index;

                    if (use_pu)
                    {
                        break;
                    }
                }

                next_pu_index[num_core] = pu_index;

                if (!use_pu)
                {
                    continue;
                }

                pu_indexes[num_core].push_back(next_pu_index[num_core] - 1);

                num_pus_cores[num_core]++;
                if (++num_thread == num_threads)
                    break;
            }
        }

        // Iterate over the cores and assigned pus per core. this additional
        // loop is needed so that we have consecutive worker thread numbers
        std::size_t num_thread = 0;
        for (std::size_t num_core = 0; num_core < num_cores; ++num_core)
        {
            for (std::size_t num_pu = 0; num_pu < num_pus_cores[num_core];
                 ++num_pu)
            {
                if (any(affinities[num_thread]))
                {
                    HPX_THROWS_IF(ec, bad_parameter,
                        "decode_balanced_distribution",
                        "affinity mask for thread {1} has already been set",
                        num_thread);
                    return;
                }

                num_pus[num_thread] = t.get_pu_number(
                    num_core + used_cores, pu_indexes[num_core][num_pu]);
                affinities[num_thread] = t.init_thread_affinity_mask(
                    num_core + used_cores, pu_indexes[num_core][num_pu]);
                ++num_thread;
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    void decode_numabalanced_distribution(topology& t,
        std::vector<mask_type>& affinities, std::size_t used_cores,
        std::size_t max_cores, std::vector<std::size_t>& num_pus,
        bool use_process_mask, error_code& ec)
    {
        HPX_UNUSED(max_cores);
        std::size_t num_threads = affinities.size();

        check_num_threads(use_process_mask, t, num_threads, ec);

        if (use_process_mask)
        {
            used_cores = 0;
        }

        num_pus.resize(num_threads);

        // numa nodes
        std::size_t num_numas =
            (std::max)(std::size_t(1), t.get_number_of_numa_nodes());
        std::vector<std::size_t> num_cores_numa(num_numas, 0);
        std::vector<std::size_t> num_pus_numa(num_numas, 0);
        std::vector<std::size_t> num_threads_numa(num_numas, 0);
        for (std::size_t n = 0; n < num_numas; ++n)
        {
            num_cores_numa[n] = t.get_number_of_numa_node_cores(n);
        }

        std::size_t core_offset = 0;
        std::size_t pus_t = 0;
        for (std::size_t n = 0; n < num_numas; ++n)
        {
            for (std::size_t num_core = 0; num_core < num_cores_numa[n];
                 ++num_core)
            {
                std::size_t num_pus =
                    t.get_number_of_core_pus(num_core + core_offset);
                for (std::size_t num_pu = 0; num_pu < num_pus; ++num_pu)
                {
                    if (pu_in_process_mask(use_process_mask, t,
                            num_core + core_offset, num_pu))
                    {
                        ++num_pus_numa[n];
                    }
                }
            }

            pus_t += num_pus_numa[n];
            core_offset += num_cores_numa[n];
        }

        // how many threads should go on each domain
        std::size_t pus_t2 = 0;
        for (std::size_t n = 0; n < num_numas; ++n)
        {
            std::size_t temp = static_cast<std::size_t>(
                std::round(static_cast<double>(num_threads * num_pus_numa[n]) /
                    static_cast<double>(pus_t)));

            // due to rounding up, we might have too many threads
            if ((pus_t2 + temp) > num_threads)
                temp = num_threads - pus_t2;
            pus_t2 += temp;
            num_threads_numa[n] = temp;

            // HPX_ASSERT(num_threads_numa[n] <= num_pus_numa[n]);
        }

        // HPX_ASSERT(num_threads <= pus_t2);

        // assign threads to cores on each numa domain
        std::size_t num_thread = 0;
        core_offset = 0;
        for (std::size_t n = 0; n < num_numas; ++n)
        {
            std::vector<std::size_t> num_pus_cores(num_cores_numa[n], 0);
            std::vector<std::size_t> next_pu_index(num_cores_numa[n], 0);
            std::vector<std::vector<std::size_t>> pu_indexes(num_cores_numa[n]);

            // iterate once and count pus/core
            for (std::size_t num_thread_numa = 0;
                 num_thread_numa < num_threads_numa[n];
                /**/)
            {
                for (std::size_t num_core = 0; num_core < num_cores_numa[n];
                     ++num_core)
                {
                    std::size_t num_core_pus =
                        t.get_number_of_core_pus(num_core);
                    std::size_t pu_index = next_pu_index[num_core];
                    bool use_pu = false;

                    // Find the next PU on this core which is in the process mask
                    while (pu_index < num_core_pus)
                    {
                        use_pu = pu_in_process_mask(use_process_mask, t,
                            num_core + core_offset, pu_index);
                        ++pu_index;

                        if (use_pu)
                        {
                            break;
                        }
                    }

                    next_pu_index[num_core] = pu_index;

                    if (!use_pu)
                    {
                        continue;
                    }

                    pu_indexes[num_core].push_back(next_pu_index[num_core] - 1);

                    num_pus_cores[num_core]++;
                    if (++num_thread_numa == num_threads_numa[n])
                        break;
                }
            }

            // Iterate over the cores and assigned pus per core. this additional
            // loop is needed so that we have consecutive worker thread numbers
            for (std::size_t num_core = 0; num_core < num_cores_numa[n];
                 ++num_core)
            {
                for (std::size_t num_pu = 0; num_pu < num_pus_cores[num_core];
                     ++num_pu)
                {
                    if (any(affinities[num_thread]))
                    {
                        HPX_THROWS_IF(ec, bad_parameter,
                            "decode_numabalanced_distribution",
                            "affinity mask for thread {1} has already been set",
                            num_thread);
                        return;
                    }
                    num_pus[num_thread] = t.get_pu_number(
                        num_core + used_cores, pu_indexes[num_core][num_pu]);
                    affinities[num_thread] = t.init_thread_affinity_mask(
                        num_core + used_cores + core_offset,
                        pu_indexes[num_core][num_pu]);
                    ++num_thread;
                }
            }
            core_offset += num_cores_numa[n];
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    void decode_distribution(distribution_type d, topology& t,
        std::vector<mask_type>& affinities, std::size_t used_cores,
        std::size_t max_cores, std::size_t num_threads,
        std::vector<std::size_t>& num_pus, bool use_process_mask,
        error_code& ec)
    {
        affinities.resize(num_threads);
        switch (d)
        {
        case compact:
            decode_compact_distribution(t, affinities, used_cores, max_cores,
                num_pus, use_process_mask, ec);
            break;

        case scatter:
            decode_scatter_distribution(t, affinities, used_cores, max_cores,
                num_pus, use_process_mask, ec);
            break;

        case balanced:
            decode_balanced_distribution(t, affinities, used_cores, max_cores,
                num_pus, use_process_mask, ec);
            break;

        case numa_balanced:
            decode_numabalanced_distribution(t, affinities, used_cores,
                max_cores, num_pus, use_process_mask, ec);
            break;

        default:
            HPX_ASSERT(false);
        }
    }
}}}    // namespace hpx::threads::detail

namespace hpx { namespace threads {
    ///////////////////////////////////////////////////////////////////////////
    void parse_affinity_options(std::string const& spec,
        std::vector<mask_type>& affinities, std::size_t used_cores,
        std::size_t max_cores, std::size_t num_threads,
        std::vector<std::size_t>& num_pus, bool use_process_mask,
        error_code& ec)
    {
        detail::mappings_type mappings;
        detail::parse_mappings(spec, mappings, ec);
        if (ec)
            return;

        // We need to instantiate a new topology object as the runtime has not
        // been initialized yet
        threads::topology& t = threads::create_topology();

        detail::decode_distribution(mappings, t, affinities, used_cores,
            max_cores, num_threads, num_pus, use_process_mask, ec);
        if (ec)
            return;
    }
}}    // namespace hpx::threads
