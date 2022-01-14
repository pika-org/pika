//  Copyright (c)      2017 Shoshana Jakobovits
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>
#include <pika/functional/function.hpp>
#include <pika/ini/ini.hpp>
#include <pika/resource_partitioner/detail/create_partitioner.hpp>
#include <pika/resource_partitioner/partitioner_fwd.hpp>
#include <pika/threading_base/scheduler_mode.hpp>

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace pika { namespace resource {
    ///////////////////////////////////////////////////////////////////////////
    class pu
    {
        static constexpr const std::size_t invalid_pu_id = std::size_t(-1);

    public:
        explicit pu(std::size_t id = invalid_pu_id, core* core = nullptr,
            std::size_t thread_occupancy = 0)
          : id_(id)
          , core_(core)
          , thread_occupancy_(thread_occupancy)
          , thread_occupancy_count_(0)
        {
        }

        std::size_t id() const
        {
            return id_;
        }

    private:
        friend class core;
        friend class numa_domain;
        friend class resource::detail::partitioner;

        std::vector<pu> pus_sharing_core();
        std::vector<pu> pus_sharing_numa_domain();

        std::size_t id_;
        core* core_;

        // indicates the number of threads that should run on this PU
        //  0: this PU is not exposed by the affinity bindings
        //  1: normal occupancy
        // >1: oversubscription
        std::size_t thread_occupancy_;

        // counts number of threads bound to this PU
        mutable std::size_t thread_occupancy_count_;
    };

    class core
    {
        static constexpr const std::size_t invalid_core_id = std::size_t(-1);

    public:
        explicit core(
            std::size_t id = invalid_core_id, numa_domain* domain = nullptr)
          : id_(id)
          , domain_(domain)
        {
        }

        std::vector<pu> const& pus() const
        {
            return pus_;
        }
        std::size_t id() const
        {
            return id_;
        }

    private:
        std::vector<core> cores_sharing_numa_domain();

        friend class pu;
        friend class numa_domain;
        friend class resource::detail::partitioner;

        std::size_t id_;
        numa_domain* domain_;
        std::vector<pu> pus_;
    };

    class numa_domain
    {
        static constexpr const std::size_t invalid_numa_domain_id =
            std::size_t(-1);

    public:
        explicit numa_domain(std::size_t id = invalid_numa_domain_id)
          : id_(id)
        {
        }

        std::vector<core> const& cores() const
        {
            return cores_;
        }
        std::size_t id() const
        {
            return id_;
        }

    private:
        friend class pu;
        friend class core;
        friend class resource::detail::partitioner;

        std::size_t id_;
        std::vector<core> cores_;
    };

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
        inline ::pika::resource::partitioner make_partitioner(
            resource::partitioner_mode rpmode, pika::util::section rtcfg,
            pika::threads::policies::detail::affinity_data affinity_data);
    }

    class partitioner
    {
    private:
        friend ::pika::resource::partitioner detail::make_partitioner(
            resource::partitioner_mode rpmode, pika::util::section rtcfg,
            pika::threads::policies::detail::affinity_data affinity_data);

        partitioner(resource::partitioner_mode rpmode, pika::util::section rtcfg,
            pika::threads::policies::detail::affinity_data affinity_data)
          : partitioner_(
                detail::create_partitioner(rpmode, rtcfg, affinity_data))
        {
        }

    public:
        ///////////////////////////////////////////////////////////////////////
        // Create one of the predefined thread pools
        PIKA_EXPORT void create_thread_pool(std::string const& name,
            scheduling_policy sched = scheduling_policy::unspecified,
            pika::threads::policies::scheduler_mode =
                pika::threads::policies::scheduler_mode::default_mode);

        // Create a custom thread pool with a callback function
        PIKA_EXPORT void create_thread_pool(
            std::string const& name, scheduler_function scheduler_creation);

        // allow the default pool to be renamed to something else
        PIKA_EXPORT void set_default_pool_name(std::string const& name);

        PIKA_EXPORT const std::string& get_default_pool_name() const;

        ///////////////////////////////////////////////////////////////////////
        // Functions to add processing units to thread pools via
        // the pu/core/numa_domain API
        void add_resource(pika::resource::pu const& p,
            std::string const& pool_name, std::size_t num_threads = 1)
        {
            add_resource(p, pool_name, true, num_threads);
        }
        PIKA_EXPORT void add_resource(pika::resource::pu const& p,
            std::string const& pool_name, bool exclusive,
            std::size_t num_threads = 1);
        PIKA_EXPORT void add_resource(
            std::vector<pika::resource::pu> const& pv,
            std::string const& pool_name, bool exclusive = true);
        PIKA_EXPORT void add_resource(pika::resource::core const& c,
            std::string const& pool_name, bool exclusive = true);
        PIKA_EXPORT void add_resource(std::vector<pika::resource::core>& cv,
            std::string const& pool_name, bool exclusive = true);
        PIKA_EXPORT void add_resource(pika::resource::numa_domain const& nd,
            std::string const& pool_name, bool exclusive = true);
        PIKA_EXPORT void add_resource(
            std::vector<pika::resource::numa_domain> const& ndv,
            std::string const& pool_name, bool exclusive = true);

        // Access all available NUMA domains
        PIKA_EXPORT std::vector<numa_domain> const& numa_domains() const;

        // Returns the threads requested at startup --pika:threads=cores
        // for example will return the number actually created
        PIKA_EXPORT std::size_t get_number_requested_threads();

        // return the topology object managed by the internal partitioner
        PIKA_EXPORT pika::threads::topology const& get_topology() const;

        // Does initialization of all resources and internal data of the
        // resource partitioner called in pika_init
        PIKA_EXPORT void configure_pools();

    private:
        detail::partitioner& partitioner_;
    };

    namespace detail {
        ::pika::resource::partitioner make_partitioner(
            resource::partitioner_mode rpmode, pika::util::section rtcfg,
            pika::threads::policies::detail::affinity_data affinity_data)
        {
            return ::pika::resource::partitioner(rpmode, rtcfg, affinity_data);
        }
    }    // namespace detail
}}       // namespace pika::resource
