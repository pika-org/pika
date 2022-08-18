//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/affinity/affinity_data.hpp>
#include <pika/affinity/parse_affinity_options.hpp>
#include <pika/assert.hpp>
#include <pika/errors/error_code.hpp>
#include <pika/topology/cpu_mask.hpp>
#include <pika/topology/topology.hpp>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace pika::detail {
    inline std::size_t count_initialized(
        std::vector<threads::detail::mask_type> const& masks)
    {
        std::size_t count = 0;
        for (threads::detail::mask_cref_type m : masks)
        {
            if (threads::detail::any(m))
                ++count;
        }
        return count;
    }

    affinity_data::affinity_data()
      : num_threads_(0)
      , pu_offset_(std::size_t(-1))
      , pu_step_(1)
      , used_cores_(0)
      , affinity_domain_("pu")
      , affinity_masks_()
      , pu_nums_()
      , no_affinity_()
      , use_process_mask_(true)
      , num_pus_needed_(0)
    {
        threads::detail::resize(
            no_affinity_, threads::detail::hardware_concurrency());
    }

    affinity_data::~affinity_data()
    {
        --instance_number_counter_;
    }

    // NOLINTBEGIN(bugprone-easily-swappable-parameters)
    void affinity_data::init(std::size_t num_threads, std::size_t max_cores,
        std::size_t pu_offset, std::size_t pu_step, std::size_t used_cores,
        std::string affinity_domain,    // -V813
        std::string const& affinity_description, bool use_process_mask)
    // NOLINTEND(bugprone-easily-swappable-parameters)
    {
#if defined(__APPLE__)
        use_process_mask = false;
#endif

        use_process_mask_ = use_process_mask;
        num_threads_ = num_threads;
        std::size_t num_system_pus = threads::detail::hardware_concurrency();

        if (pu_offset == std::size_t(-1))
        {
            pu_offset_ = 0;
        }
        else
        {
            pu_offset_ = pu_offset;
        }

        if (num_system_pus > 1)
        {
            pu_step_ = pu_step % num_system_pus;
        }

        affinity_domain_ = PIKA_MOVE(affinity_domain);
        pu_nums_.clear();

        init_cached_pu_nums(num_system_pus);

        auto const& topo = threads::detail::create_topology();

        if (affinity_description == "none")
        {
            // don't use any affinity for any of the os-threads
            threads::detail::resize(no_affinity_, num_system_pus);
            for (std::size_t i = 0; i != num_threads_; ++i)
                threads::detail::set(no_affinity_, get_pu_num(i));
        }
        else if (!affinity_description.empty())
        {
            affinity_masks_.clear();
            affinity_masks_.resize(num_threads_, threads::detail::mask_type{});

            for (std::size_t i = 0; i != num_threads_; ++i)
                threads::detail::resize(affinity_masks_[i], num_system_pus);

            parse_affinity_options(affinity_description, affinity_masks_,
                used_cores, max_cores, num_threads_, pu_nums_,
                use_process_mask_);

            std::size_t num_initialized = count_initialized(affinity_masks_);
            if (num_initialized != num_threads_)
            {
                PIKA_THROW_EXCEPTION(pika::error::bad_parameter,
                    "affinity_data::affinity_data",
                    "The number of OS threads requested ({1}) does not match "
                    "the number of threads to bind ({2})",
                    num_threads_, num_initialized);
            }
        }
        else if (pu_offset == std::size_t(-1))
        {
            // calculate the pu offset based on the used cores, but only if its
            // not explicitly specified
            for (std::size_t num_core = 0; num_core != used_cores; ++num_core)
            {
                pu_offset_ += topo.get_number_of_core_pus(num_core);
            }
        }

        // correct used_cores from config data if appropriate
        if (used_cores_ == 0)
        {
            used_cores_ = used_cores;
        }

        pu_offset_ %= num_system_pus;

        std::vector<std::size_t> cores;
        cores.reserve(num_threads_);
        for (std::size_t i = 0; i != num_threads_; ++i)
        {
            std::size_t add_me = topo.get_core_number(get_pu_num(i));
            cores.push_back(add_me);
        }

        std::sort(cores.begin(), cores.end());
        std::vector<std::size_t>::iterator it =
            std::unique(cores.begin(), cores.end());
        cores.erase(it, cores.end());

        std::size_t num_unique_cores = cores.size();

        num_pus_needed_ = (std::max)(num_unique_cores, max_cores);
    }

    threads::detail::mask_cref_type affinity_data::get_pu_mask(
        threads::detail::topology const& topo,
        std::size_t global_thread_num) const
    {
        // --pika:bind=none disables all affinity
        if (threads::detail::test(no_affinity_, global_thread_num))
        {
            static threads::detail::mask_type m = threads::detail::mask_type();
            threads::detail::resize(m, threads::detail::hardware_concurrency());
            return m;
        }

        // if we have individual, predefined affinity masks, return those
        if (!affinity_masks_.empty())
            return affinity_masks_[global_thread_num];

        // otherwise return mask based on affinity domain
        std::size_t pu_num = get_pu_num(global_thread_num);
        if (0 == std::string("pu").find(affinity_domain_))
        {
            // The affinity domain is 'processing unit', just convert the
            // pu-number into a bit-mask.
            return topo.get_thread_affinity_mask(pu_num);
        }
        if (0 == std::string("core").find(affinity_domain_))
        {
            // The affinity domain is 'core', return a bit mask corresponding
            // to all processing units of the core containing the given
            // pu_num.
            return topo.get_core_affinity_mask(pu_num);
        }
        if (0 == std::string("numa").find(affinity_domain_))
        {
            // The affinity domain is 'numa', return a bit mask corresponding
            // to all processing units of the NUMA domain containing the
            // given pu_num.
            return topo.get_numa_node_affinity_mask(pu_num);
        }

        // The affinity domain is 'machine', return a bit mask corresponding
        // to all processing units of the machine.
        PIKA_ASSERT(0 == std::string("machine").find(affinity_domain_));
        return topo.get_machine_affinity_mask();
    }

    threads::detail::mask_type affinity_data::get_used_pus_mask(
        threads::detail::topology const& topo, std::size_t pu_num) const
    {
        threads::detail::mask_type ret = threads::detail::mask_type();
        threads::detail::resize(ret, threads::detail::hardware_concurrency());

        // --pika:bind=none disables all affinity
        if (threads::detail::test(no_affinity_, pu_num))
        {
            threads::detail::set(ret, pu_num);
            return ret;
        }

        for (std::size_t thread_num = 0; thread_num < num_threads_;
             ++thread_num)
        {
            ret |= get_pu_mask(topo, thread_num);
        }

        return ret;
    }

    std::size_t affinity_data::get_thread_occupancy(
        threads::detail::topology const& topo, std::size_t pu_num) const
    {
        std::size_t count = 0;
        if (threads::detail::test(no_affinity_, pu_num))
        {
            ++count;
        }
        else
        {
            threads::detail::mask_type pu_mask = threads::detail::mask_type();

            threads::detail::resize(
                pu_mask, threads::detail::hardware_concurrency());
            threads::detail::set(pu_mask, pu_num);

            for (std::size_t num_thread = 0; num_thread < num_threads_;
                 ++num_thread)
            {
                threads::detail::mask_cref_type affinity_mask =
                    get_pu_mask(topo, num_thread);
                if (threads::detail::any(pu_mask & affinity_mask))
                    ++count;
            }
        }
        return count;
    }

    // means of adding a processing unit after initialization
    void affinity_data::add_punit(std::size_t virt_core, std::size_t thread_num)
    {
        std::size_t num_system_pus = threads::detail::hardware_concurrency();

        // initialize affinity_masks and set the mask for the given virt_core
        if (affinity_masks_.empty())
        {
            affinity_masks_.resize(num_threads_);
            for (std::size_t i = 0; i != num_threads_; ++i)
                threads::detail::resize(affinity_masks_[i], num_system_pus);
        }
        threads::detail::set(affinity_masks_[virt_core], thread_num);

        // find first used pu, which is then stored as the pu_offset
        std::size_t first_pu = std::size_t(-1);
        for (std::size_t i = 0; i != num_threads_; ++i)
        {
            std::size_t first = threads::detail::find_first(affinity_masks_[i]);
            first_pu = (std::min)(first_pu, first);
        }
        if (first_pu != std::size_t(-1))
            pu_offset_ = first_pu;

        init_cached_pu_nums(num_system_pus);
    }

    void affinity_data::init_cached_pu_nums(std::size_t hardware_concurrency)
    {
        if (pu_nums_.empty())
        {
            pu_nums_.resize(num_threads_);
            for (std::size_t i = 0; i != num_threads_; ++i)
            {
                pu_nums_[i] = get_pu_num(i, hardware_concurrency);
            }
        }
    }

    // NOLINTBEGIN(bugprone-easily-swappable-parameters)
    std::size_t affinity_data::get_pu_num(
        std::size_t num_thread, std::size_t hardware_concurrency) const
    // NOLINTEND(bugprone-easily-swappable-parameters)
    {
        // The offset shouldn't be larger than the number of available
        // processing units.
        PIKA_ASSERT(pu_offset_ < hardware_concurrency);

        // The distance between assigned processing units shouldn't be zero
        PIKA_ASSERT(pu_step_ > 0 && pu_step_ <= hardware_concurrency);

        // We 'scale' the thread number to compute the corresponding
        // processing unit number.
        //
        // The base line processing unit number is computed from the given
        // pu-offset and pu-step.
        std::size_t num_pu = pu_offset_ + pu_step_ * num_thread;

        // We add an additional offset, which allows to 'roll over' if the
        // pu number would get larger than the number of available
        // processing units. Note that it does not make sense to 'roll over'
        // farther than the given pu-step.
        std::size_t offset = (num_pu / hardware_concurrency) % pu_step_;

        // The resulting pu number has to be smaller than the available
        // number of processing units.
        return (num_pu + offset) % hardware_concurrency;
    }

    std::atomic<int> affinity_data::instance_number_counter_(-1);
}    // namespace pika::detail
