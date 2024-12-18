//  Copyright (c)      2017 Shoshana Jakobovits
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/config.hpp>
#include <pika/resource_partitioner/detail/partitioner.hpp>
#include <pika/resource_partitioner/partitioner.hpp>
#include <pika/topology/cpu_mask.hpp>

#include <cstddef>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pika::resource {
    ///////////////////////////////////////////////////////////////////////////
    std::vector<pu> pu::pus_sharing_core()
    {
        std::vector<pu> result;
        result.reserve(core_->pus_.size());

        for (pu const& p : core_->pus_)
        {
            if (p.id_ != id_) { result.push_back(p); }
        }
        return result;
    }

    std::vector<pu> pu::pus_sharing_socket()
    {
        std::vector<pu> result;
        result.reserve(core_->socket_->cores_.size());

        for (core const& c : core_->socket_->cores_)
        {
            for (pu const& p : c.pus_)
            {
                if (p.id_ != id_) { result.push_back(p); }
            }
        }
        return result;
    }

    std::vector<core> core::cores_sharing_socket()
    {
        std::vector<core> result;
        result.reserve(socket_->cores_.size());

        for (core const& c : socket_->cores_)
        {
            if (c.id_ != id_) { result.push_back(c); }
        }
        return result;
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
        std::recursive_mutex& partitioner_mtx()
        {
            static std::recursive_mutex mtx;
            return mtx;
        }

        std::unique_ptr<detail::partitioner>& partitioner_ref()
        {
            static std::unique_ptr<detail::partitioner> part;
            return part;
        }

        std::unique_ptr<detail::partitioner>& get_partitioner()
        {
            std::lock_guard<std::recursive_mutex> l(partitioner_mtx());
            std::unique_ptr<detail::partitioner>& part = partitioner_ref();
            if (!part) part.reset(new detail::partitioner);
            return part;
        }

        void delete_partitioner()
        {
            // don't lock the mutex as otherwise will be still locked while
            // being destroyed (leading to problems on some platforms)
            std::unique_ptr<detail::partitioner>& part = partitioner_ref();
            if (part) part.reset();
        }
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    detail::partitioner& get_partitioner()
    {
        std::unique_ptr<detail::partitioner>& rp = detail::get_partitioner();

        if (!rp)
        {
            // if the resource partitioner is not accessed for the first time
            // if the command-line parsing has not yet been done
            PIKA_THROW_EXCEPTION(pika::error::invalid_status, "pika::resource::get_partitioner",
                "can be called only after the resource partitioner has been initialized and before "
                "it has been deleted.");
        }

        return *rp;
    }

    bool is_partitioner_valid() { return detail::partitioner_ref() != nullptr; }

    namespace detail {
        detail::partitioner& create_partitioner(resource::partitioner_mode rpmode,
            pika::detail::section rtcfg, pika::detail::affinity_data affinity_data)
        {
            std::unique_ptr<detail::partitioner>& rp = detail::get_partitioner();

            rp->init(rpmode, rtcfg, affinity_data);

            return *rp;
        }
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    partitioner::partitioner(resource::partitioner_mode rpmode, pika::detail::section rtcfg,
        pika::detail::affinity_data affinity_data)
      : partitioner_(detail::create_partitioner(rpmode, rtcfg, affinity_data))
    {
    }

    void partitioner::create_thread_pool(std::string const& name,
        scheduling_policy sched /*= scheduling_policy::unspecified*/,
        pika::threads::scheduler_mode mode)
    {
        partitioner_.create_thread_pool(name, sched, mode);
    }

    void partitioner::create_thread_pool(
        std::string const& name, scheduler_function scheduler_creation)
    {
        partitioner_.create_thread_pool(name, scheduler_creation);
    }

    void partitioner::set_default_pool_name(std::string const& name)
    {
        partitioner_.set_default_pool_name(name);
    }

    std::string const& partitioner::get_default_pool_name() const
    {
        return partitioner_.get_default_pool_name();
    }

    void partitioner::add_resource(
        pu const& p, std::string const& pool_name, bool exclusive, std::size_t num_threads /*= 1*/)
    {
        partitioner_.add_resource(p, pool_name, exclusive, num_threads);
    }

    void partitioner::add_resource(
        std::vector<pu> const& pv, std::string const& pool_name, bool exclusive /*= true*/)
    {
        partitioner_.add_resource(pv, pool_name, exclusive);
    }

    void partitioner::add_resource(
        core const& c, std::string const& pool_name, bool exclusive /*= true*/)
    {
        partitioner_.add_resource(c, pool_name, exclusive);
    }

    void partitioner::add_resource(
        std::vector<core>& cv, std::string const& pool_name, bool exclusive /*= true*/)
    {
        partitioner_.add_resource(cv, pool_name, exclusive);
    }

    void partitioner::add_resource(
        socket const& nd, std::string const& pool_name, bool exclusive /*= true*/)
    {
        partitioner_.add_resource(nd, pool_name, exclusive);
    }

    void partitioner::add_resource(
        std::vector<socket> const& ndv, std::string const& pool_name, bool exclusive /*= true*/)
    {
        partitioner_.add_resource(ndv, pool_name, exclusive);
    }

    std::vector<socket> const& partitioner::sockets() const { return partitioner_.sockets(); }

    pika::threads::detail::topology const& partitioner::get_topology() const
    {
        return partitioner_.get_topology();
    }

    std::size_t partitioner::get_number_requested_threads()
    {
        return partitioner_.threads_needed();
    }

    // Does initialization of all resources and internal data of the
    // resource partitioner called in pika_init
    void partitioner::configure_pools() { partitioner_.configure_pools(); }

    namespace detail {

        ::pika::resource::partitioner make_partitioner(resource::partitioner_mode rpmode,
            pika::detail::section rtcfg, pika::detail::affinity_data affinity_data)
        {
            return ::pika::resource::partitioner(rpmode, rtcfg, affinity_data);
        }

        char const* get_scheduling_policy_name(scheduling_policy p) noexcept
        {
            switch (p)
            {
            case scheduling_policy::user_defined: return "user_defined";
            case scheduling_policy::unspecified: return "unspecified";
            case scheduling_policy::local: return "local";
            case scheduling_policy::local_priority_fifo: return "local_priority_fifo";
            case scheduling_policy::local_priority_lifo: return "local_priority_lifo";
            case scheduling_policy::static_: return "static";
            case scheduling_policy::static_priority: return "static_priority";
            case scheduling_policy::abp_priority_fifo: return "abp_priority_fifo";
            case scheduling_policy::abp_priority_lifo: return "abp_priority_lifo";
            case scheduling_policy::shared_priority: return "shared_priority";
            default: return "unknown";
            }
        }
    }    // namespace detail
}    // namespace pika::resource
