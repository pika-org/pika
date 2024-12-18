//  Copyright (c)      2017 Shoshana Jakobovits
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/config.hpp>
#include <pika/assert.hpp>
#include <pika/functional/function.hpp>
#include <pika/ini/ini.hpp>
#include <pika/modules/errors.hpp>
#include <pika/resource_partitioner/detail/partitioner.hpp>
#include <pika/resource_partitioner/partitioner.hpp>
#include <pika/string_util/from_string.hpp>
#include <pika/thread_pools/scheduled_thread_pool.hpp>
#include <pika/threading_base/scheduler_mode.hpp>
#include <pika/threading_base/thread_pool_base.hpp>
#include <pika/topology/topology.hpp>
#include <pika/util/get_entry_as.hpp>

#include <fmt/format.h>

#include <atomic>
#include <cstddef>
#include <iosfwd>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace pika::resource::detail {
    ///////////////////////////////////////////////////////////////////////////
    [[noreturn]] void throw_runtime_error(std::string const& func, std::string const& message)
    {
        PIKA_THROW_EXCEPTION(pika::error::invalid_status, func, "{}", message);
    }

    [[noreturn]] void throw_invalid_argument(std::string const& func, std::string const& message)
    {
        PIKA_THROW_EXCEPTION(pika::error::bad_parameter, func, "{}", message);
    }

    ///////////////////////////////////////////////////////////////////////////
    std::size_t init_pool_data::num_threads_overall = 0;

    init_pool_data::init_pool_data(
        std::string const& name, scheduling_policy sched, pika::threads::scheduler_mode mode)
      : pool_name_(name)
      , scheduling_policy_(sched)
      , num_threads_(0)
      , mode_(mode)
    {
        if (name.empty())
        {
            throw_invalid_argument("init_pool_data::init_pool_data",
                "cannot instantiate a thread_pool with empty string as a name.");
        }
    }

    init_pool_data::init_pool_data(
        std::string const& name, scheduler_function create_func, pika::threads::scheduler_mode mode)
      : pool_name_(name)
      , scheduling_policy_(user_defined)
      , num_threads_(0)
      , mode_(mode)
      , create_function_(std::move(create_func))
    {
        if (name.empty())
        {
            throw_invalid_argument("init_pool_data::init_pool_data",
                "cannot instantiate a thread_pool with empty string as a name.");
        }
    }

    // mechanism for adding resources
    // num threads = number of threads desired on a PU. defaults to 1.
    // note: if num_threads > 1 => oversubscription
    void init_pool_data::add_resource(std::size_t pu_index, bool exclusive, std::size_t num_threads)
    {
        if (pu_index >= pika::threads::detail::hardware_concurrency())
        {
            throw_invalid_argument("init_pool_data::add_resource",
                "init_pool_data::add_resource: processing unit index out of bounds. The total "
                "available number of processing units on this machine is " +
                    std::to_string(pika::threads::detail::hardware_concurrency()));
        }

        // Increment thread_num count (for pool-count and global count)
        num_threads_ += num_threads;
        num_threads_overall += num_threads;

        // Add pu mask to internal data structure
        threads::detail::mask_type pu_mask = threads::detail::mask_type();
        threads::detail::resize(pu_mask, threads::detail::hardware_concurrency());
        threads::detail::set(pu_mask, pu_index);

        // Add one mask for each OS-thread
        for (std::size_t i = 0; i != num_threads; i++)
        {
            assigned_pus_.push_back(pu_mask);
            assigned_pu_nums_.push_back(std::make_tuple(pu_index, exclusive, false));
        }
    }

    void init_pool_data::print_pool(std::ostream& os) const
    {
        os << "[pool \"" << pool_name_ << "\"] with scheduler ";

        std::string sched;
        switch (scheduling_policy_)
        {
        case resource::unspecified: sched = "unspecified"; break;
        case resource::user_defined: sched = "user supplied"; break;
        case resource::local: sched = "local"; break;
        case resource::local_priority_fifo: sched = "local_priority_fifo"; break;
        case resource::local_priority_lifo: sched = "local_priority_lifo"; break;
        case resource::static_: sched = "static"; break;
        case resource::static_priority: sched = "static_priority"; break;
        case resource::abp_priority_fifo: sched = "abp_priority_fifo"; break;
        case resource::abp_priority_lifo: sched = "abp_priority_lifo"; break;
        case resource::shared_priority: sched = "shared_priority"; break;
        }

        os << "\"" << sched << "\" is running on PUs : \n";

        for (threads::detail::mask_cref_type assigned_pu : assigned_pus_)
        {
            os << pika::threads::detail::to_string(assigned_pu) << '\n';
        }
    }

    void init_pool_data::assign_pu(std::size_t virt_core)
    {
        PIKA_ASSERT(virt_core <= assigned_pu_nums_.size());
        PIKA_ASSERT(!std::get<2>(assigned_pu_nums_[virt_core]));

        std::get<2>(assigned_pu_nums_[virt_core]) = true;    // -V601
    }

    void init_pool_data::unassign_pu(std::size_t virt_core)
    {
        PIKA_ASSERT(virt_core <= assigned_pu_nums_.size());
        PIKA_ASSERT(std::get<2>(assigned_pu_nums_[virt_core]));

        std::get<2>(assigned_pu_nums_[virt_core]) = false;    // -V601
    }

    bool init_pool_data::pu_is_exclusive(std::size_t virt_core) const
    {
        PIKA_ASSERT(virt_core <= assigned_pu_nums_.size());
        PIKA_ASSERT(std::get<2>(assigned_pu_nums_[virt_core]));

        return std::get<1>(assigned_pu_nums_[virt_core]);
    }

    bool init_pool_data::pu_is_assigned(std::size_t virt_core) const
    {
        PIKA_ASSERT(virt_core <= assigned_pu_nums_.size());
        PIKA_ASSERT(std::get<2>(assigned_pu_nums_[virt_core]));

        return std::get<2>(assigned_pu_nums_[virt_core]);
    }

    // 'shift' all thread assignments up by the first_core offset
    void init_pool_data::assign_first_core(std::size_t first_core)
    {
        for (std::size_t i = 0; i != num_threads_; ++i)
        {
            std::size_t& pu_num = std::get<0>(assigned_pu_nums_[i]);
            pu_num = (pu_num + first_core) % threads::detail::hardware_concurrency();

            threads::detail::reset(assigned_pus_[i]);
            threads::detail::set(assigned_pus_[i], pu_num);
        }
    }

    ////////////////////////////////////////////////////////////////////////
    partitioner::partitioner()
      : rtcfg_()
      , first_core_(std::size_t(-1))
      , mode_(mode_default)
      , topo_(threads::detail::get_topology())
      , default_scheduler_mode_(threads::scheduler_mode::default_mode)
    {
        // allow only one partitioner instance
        if (++instance_number_counter_ > 1)
        {
            throw_runtime_error("partitioner::partitioner",
                "Cannot instantiate more than one resource partitioner");
        }

#if defined(PIKA_HAVE_MAX_CPU_COUNT)
        if (PIKA_HAVE_MAX_CPU_COUNT < topo_.get_number_of_pus())
        {
            throw_runtime_error("partitioner::partioner",
                fmt::format(
                    "Currently, PIKA_HAVE_MAX_CPU_COUNT is set to {0} while your system has {1} "
                    "processing units. Please reconfigure pika with -DPIKA_WITH_MAX_CPU_COUNT={1} "
                    "(or higher) to increase the maximal CPU count supported by pika.",
                    PIKA_HAVE_MAX_CPU_COUNT, topo_.get_number_of_pus()));
        }
#endif

        std::string default_scheduler_mode_str =
            rtcfg_.get_entry("pika.default_scheduler_mode", std::string());
        if (!default_scheduler_mode_str.empty())
        {
            default_scheduler_mode_ = threads::scheduler_mode(
                pika::detail::from_string<std::size_t>(default_scheduler_mode_str));
            PIKA_ASSERT_MSG((default_scheduler_mode_ & ~threads::scheduler_mode::all_flags) ==
                    threads::scheduler_mode{},
                "pika.default_scheduler_mode contains unknown scheduler modes");
        }

        // Create the default pool
        initial_thread_pools_.push_back(
            init_pool_data("default", scheduling_policy::unspecified, default_scheduler_mode_));
    }

    partitioner::~partitioner()
    {
        --instance_number_counter_;
        detail::init_pool_data::num_threads_overall = 0;
    }

    bool partitioner::pu_exposed(std::size_t pu_num)
    {
        threads::detail::mask_type pu_mask = threads::detail::mask_type();
        threads::detail::resize(pu_mask, threads::detail::hardware_concurrency());
        threads::detail::set(pu_mask, pu_num);
        threads::detail::topology& topo = get_topology();

        threads::detail::mask_type comp = affinity_data_.get_used_pus_mask(topo, pu_num);
        return threads::detail::any(comp & pu_mask);
    }

    void partitioner::fill_topology_vectors()
    {
        threads::detail::topology& topo = get_topology();

        std::size_t pid = 0;
        std::size_t num_sockets = topo.get_number_of_sockets();
        sockets_.reserve(num_sockets);

        // loop on the sockets
        for (std::size_t i = 0; i != num_sockets; ++i)
        {
            sockets_.emplace_back(i);        // add a socket
            socket& nd = sockets_.back();    // socket just added

            std::size_t socket_cores = topo.get_number_of_socket_cores(i);
            nd.cores_.reserve(socket_cores);

            bool socket_contains_exposed_cores = false;

            // loop on the cores
            for (std::size_t j = 0; j != socket_cores; ++j)
            {
                nd.cores_.emplace_back(j, &nd);
                core& c = nd.cores_.back();

                std::size_t core_pus = topo.get_number_of_core_pus(j);
                c.pus_.reserve(core_pus);

                bool core_contains_exposed_pus = false;

                // loop on the processing units
                for (std::size_t k = 0; k != core_pus; ++k)
                {
                    if (pu_exposed(pid))
                    {
                        c.pus_.emplace_back(
                            pid, &c, affinity_data_.get_thread_occupancy(topo, pid));
                        pu& p = c.pus_.back();

                        if (p.thread_occupancy_ == 0)
                        {
                            throw_runtime_error("partitioner::fill_topology_vectors",
                                "PU #" + std::to_string(pid) + " has thread occupancy 0");
                        }
                        core_contains_exposed_pus = true;
                    }
                    ++pid;
                }

                if (core_contains_exposed_pus) { socket_contains_exposed_cores = true; }
                else { nd.cores_.pop_back(); }
            }

            if (!socket_contains_exposed_cores) { sockets_.pop_back(); }
        }
    }

    std::size_t partitioner::assign_cores(std::size_t first_core)
    {
        std::lock_guard<mutex_type> l(mtx_);

        // adjust first_core, if needed
        if (first_core_ != first_core)
        {
            std::size_t offset = first_core;
            std::size_t num_pus_core = get_topology().get_number_of_core_pus(offset);

            if (first_core_ != std::size_t(-1)) { offset -= first_core_; }

            if (offset != 0)
            {
                offset *= num_pus_core;
                for (auto& d : initial_thread_pools_) { d.assign_first_core(offset); }
            }
            first_core_ = first_core;
            reconfigure_affinities_locked();
        }

        PIKA_ASSERT(affinity_data_.get_num_pus_needed() != std::size_t(-1));
        return affinity_data_.get_num_pus_needed();
    }

    // This function is called in pika_init, before the instantiation of the
    // runtime It takes care of configuring some internal parameters of the
    // resource partitioner related to the pools
    // -1 assigns all free resources to the default pool
    // -2 checks whether there are empty pools
    void partitioner::setup_pools()
    {
        // Assign all free resources to the default pool
        bool first = true;
        for (pika::resource::socket& d : sockets_)
        {
            for (pika::resource::core& c : d.cores_)
            {
                for (pika::resource::pu& p : c.pus_)
                {
                    if (p.thread_occupancy_count_ == 0)
                    {
                        // The default pool resources are assigned non-
                        // exclusively if dynamic pools are enabled.
                        // Also, by default, the first PU is always exclusive
                        // (to avoid deadlocks).
                        add_resource(p, get_default_pool_name(),
                            first || !(mode_ & mode_allow_dynamic_pools));
                        first = false;
                    }
                }
            }
        }

        std::unique_lock<mutex_type> l(mtx_);

        // @TODO allow empty pools
        if (get_pool_data(l, get_default_pool_name()).num_threads_ == 0)
        {
            l.unlock();

            std::string process_mask_message = affinity_data_.using_process_mask() ?
                fmt::format("pika is using a process mask: {}.",
                    pika::threads::detail::to_string(
                        get_topology().get_cpubind_mask_main_thread())) :
                "pika is not using a process mask.";
            auto omp_proc_bind = std::getenv("OMP_PROC_BIND");
            std::string omp_proc_bind_message = omp_proc_bind ?
                fmt::format("OMP_PROC_BIND is currently set to \"{}\".", omp_proc_bind) :
                "OMP_PROC_BIND is currently unset.";
            throw_runtime_error("partitioner::setup_pools",
                fmt::format(
                    "The default pool (\"{}\") has no threads assigned. Make sure that the pika "
                    "runtime is started with enough threads for all thread pools. This can be "
                    "caused by the pika runtime being configured to use additional thread pools "
                    "(e.g. for MPI) but starting the runtime with too few threads, leaving no "
                    "threads for the default pool. It can also be caused by OpenMP using thread "
                    "pinning (through OMP_PROC_BIND) and OpenMP changing the process mask before "
                    "pika can read the mask. pika is currently configured to use {} thread{} and "
                    "{} thread pool{}. {} {}",
                    get_default_pool_name(), get_num_threads(), get_num_threads() != 1 ? "s" : "",
                    get_num_pools(), get_num_pools() != 1 ? "s" : "", process_mask_message,
                    omp_proc_bind_message));
            // TODO: Explain most common failure modes?
        }

        // Check whether any of the pools defined up to now are empty
        if (check_empty_pools())
        {
            l.unlock();
            print_init_pool_data(std::cout);
            throw_runtime_error("partitioner::setup_pools",
                "Pools empty of resources are not allowed. Please re-run this application with "
                "allow-empty-pool-policy (not implemented yet)");
        }
        //! FIXME add allow-empty-pools policy. Wait, does this even make sense??
    }

    // This function is called in pika_init, before the instantiation of the runtime
    // It takes care of configuring some internal parameters of the resource partitioner
    // related to the pools' schedulers
    void partitioner::setup_schedulers()
    {
        // select the default scheduler
        scheduling_policy default_scheduler;

        std::string default_scheduler_str = rtcfg_.get_entry("pika.scheduler", std::string());

        if (0 == std::string("local").find(default_scheduler_str))
        {
            default_scheduler = scheduling_policy::local;
        }
        else if (0 == std::string("local-priority-fifo").find(default_scheduler_str))
        {
            default_scheduler = scheduling_policy::local_priority_fifo;
        }
        else if (0 == std::string("local-priority-lifo").find(default_scheduler_str))
        {
            default_scheduler = scheduling_policy::local_priority_lifo;
        }
        else if (0 == std::string("static").find(default_scheduler_str))
        {
            default_scheduler = scheduling_policy::static_;
        }
        else if (0 == std::string("static-priority").find(default_scheduler_str))
        {
            default_scheduler = scheduling_policy::static_priority;
        }
        else if (0 == std::string("abp-priority-fifo").find(default_scheduler_str))
        {
            default_scheduler = scheduling_policy::abp_priority_fifo;
        }
        else if (0 == std::string("abp-priority-lifo").find(default_scheduler_str))
        {
            default_scheduler = scheduling_policy::abp_priority_lifo;
        }
        else if (0 == std::string("shared-priority").find(default_scheduler_str))
        {
            default_scheduler = scheduling_policy::shared_priority;
        }
        else
        {
            throw pika::detail::command_line_error(
                "Bad value for command line option --pika:queuing");
        }

        // set this scheduler on the pools that do not have a specified scheduler yet
        std::lock_guard<mutex_type> l(mtx_);
        std::size_t npools = initial_thread_pools_.size();
        for (std::size_t i = 0; i != npools; ++i)
        {
            if (initial_thread_pools_[i].scheduling_policy_ == unspecified)
            {
                initial_thread_pools_[i].scheduling_policy_ = default_scheduler;
            }
        }
    }

    // This function is called in pika_init, before the instantiation of the
    // runtime. It takes care of configuring some internal parameters of the
    // resource partitioner related to the affinity bindings
    //
    // If we use the resource partitioner, OS-thread numbering gets slightly
    // complicated: The affinity_masks_ data member of affinity_data considers
    // OS-threads to be numbered in order of occupation of the consecutive
    // processing units, while the thread manager will consider them to be
    // ordered according to their assignment to pools (first all threads
    // belonging to the default pool, then all threads belonging to the first
    // pool created, etc.) and instantiate them according to this system.
    // We need to re-write affinity_data_ with the masks in the correct order
    // at this stage.
    void partitioner::reconfigure_affinities()
    {
        std::lock_guard<mutex_type> l(mtx_);
        reconfigure_affinities_locked();
    }

    void partitioner::reconfigure_affinities_locked()
    {
        std::vector<std::size_t> new_pu_nums;
        std::vector<threads::detail::mask_type> new_affinity_masks;

        new_pu_nums.reserve(initial_thread_pools_.size());
        new_affinity_masks.reserve(initial_thread_pools_.size());

        {
            for (auto& itp : initial_thread_pools_)
            {
                for (auto const& mask : itp.assigned_pus_) { new_affinity_masks.push_back(mask); }
                for (auto const& pu_num : itp.assigned_pu_nums_)
                {
                    new_pu_nums.push_back(std::get<0>(pu_num));
                }
            }
        }

        affinity_data_.set_num_threads(new_pu_nums.size());
        affinity_data_.set_pu_nums(std::move(new_pu_nums));
        affinity_data_.set_affinity_masks(std::move(new_affinity_masks));
    }

    // Returns true if any of the pools defined by the user is empty of resources
    // called in set_default_pool()
    bool partitioner::check_empty_pools() const
    {
        std::size_t num_thread_pools = initial_thread_pools_.size();

        for (std::size_t i = 0; i != num_thread_pools; i++)
        {
            if (initial_thread_pools_[i].assigned_pus_.empty()) { return true; }
            for (auto assigned_pus : initial_thread_pools_[i].assigned_pus_)
            {
                if (!threads::detail::any(assigned_pus)) { return true; }
            }
        }

        return false;
    }

    // create a new thread_pool
    void partitioner::create_thread_pool(
        std::string const& pool_name, scheduling_policy sched, pika::threads::scheduler_mode mode)
    {
        if (pool_name.empty())
        {
            throw std::invalid_argument("partitioner::create_thread_pool: cannot instantiate a "
                                        "initial_thread_pool with empty string as a name.");
        }

        std::unique_lock<mutex_type> l(mtx_);

        if (pool_name == get_default_pool_name())
        {
            initial_thread_pools_[0] = detail::init_pool_data(get_default_pool_name(), sched, mode);
            return;
        }

        //! if there already exists a pool with this name
        std::size_t num_thread_pools = initial_thread_pools_.size();
        for (std::size_t i = 1; i < num_thread_pools; i++)
        {
            if (pool_name == initial_thread_pools_[i].pool_name_)
            {
                l.unlock();
                throw std::invalid_argument(
                    "partitioner::create_thread_pool: there already exists a pool named '" +
                    pool_name + "'.\n");
            }
        }

        initial_thread_pools_.push_back(detail::init_pool_data(pool_name, sched, mode));
    }

    // create a new thread_pool
    void partitioner::create_thread_pool(
        std::string const& pool_name, scheduler_function scheduler_creation)
    {
        if (pool_name.empty())
        {
            throw std::invalid_argument("partitioner::create_thread_pool: cannot instantiate a "
                                        "initial_thread_pool with empty string as a name.");
        }

        std::unique_lock<mutex_type> l(mtx_);

        if (pool_name == get_default_pool_name())
        {
            initial_thread_pools_[0] = detail::init_pool_data(
                get_default_pool_name(), std::move(scheduler_creation), default_scheduler_mode_);
            return;
        }

        //! if there already exists a pool with this name
        std::size_t num_thread_pools = initial_thread_pools_.size();
        for (std::size_t i = 1; i != num_thread_pools; ++i)
        {
            if (pool_name == initial_thread_pools_[i].pool_name_)
            {
                l.unlock();
                throw std::invalid_argument(
                    "partitioner::create_thread_pool: there already exists a pool named '" +
                    pool_name + "'.\n");
            }
        }

        initial_thread_pools_.push_back(detail::init_pool_data(
            pool_name, std::move(scheduler_creation), default_scheduler_mode_));
    }

    // ----------------------------------------------------------------------
    // Add processing units to pools via pu/core/socket api
    // ----------------------------------------------------------------------
    void partitioner::add_resource(
        pu const& p, std::string const& pool_name, bool exclusive, std::size_t num_threads)
    {
        std::unique_lock<mutex_type> l(mtx_);

        if (!exclusive && !(mode_ & mode_allow_dynamic_pools))
        {
            l.unlock();
            throw std::invalid_argument("partitioner::add_resource: dynamic pools have not been "
                                        "enabled for this partitioner");
        }

        if (mode_ & mode_allow_oversubscription)
        {
            // increment occupancy counter
            get_pool_data(l, pool_name).add_resource(p.id_, exclusive, num_threads);
            ++p.thread_occupancy_count_;
            return;
        }

        // check occupancy counter and increment it
        if (p.thread_occupancy_count_ == 0)
        {
            get_pool_data(l, pool_name).add_resource(p.id_, exclusive, num_threads);
            ++p.thread_occupancy_count_;

            // Make sure the total number of requested threads does not exceed
            // the number of threads requested on the command line
            std::size_t num_threads =
                ::pika::detail::get_entry_as<std::size_t>(rtcfg_, "pika.os_threads", 0);
            PIKA_ASSERT(num_threads != 0);

            if (detail::init_pool_data::num_threads_overall > num_threads)
            {
                l.unlock();
                throw std::runtime_error("partitioner::add_resource: Creation of " +
                    std::to_string(detail::init_pool_data::num_threads_overall) +
                    " threads requested by the resource partitioner, but only " +
                    std::to_string(num_threads) + " provided on the command-line.");
            }
        }
        else
        {
            l.unlock();
            throw std::runtime_error("partitioner::add_resource: PU #" + std::to_string(p.id_) +
                " can be assigned only " + std::to_string(p.thread_occupancy_) +
                " threads according to affinity bindings.");
        }
    }

    void partitioner::add_resource(
        std::vector<pu> const& pv, std::string const& pool_name, bool exclusive)
    {
        for (pu const& p : pv) { add_resource(p, pool_name, exclusive); }
    }

    void partitioner::add_resource(core const& c, std::string const& pool_name, bool exclusive)
    {
        add_resource(c.pus_, pool_name, exclusive);
    }

    void partitioner::add_resource(
        std::vector<core> const& cv, std::string const& pool_name, bool exclusive)
    {
        for (core const& c : cv) { add_resource(c.pus_, pool_name, exclusive); }
    }

    void partitioner::add_resource(socket const& nd, std::string const& pool_name, bool exclusive)
    {
        add_resource(nd.cores_, pool_name, exclusive);
    }

    void partitioner::add_resource(
        std::vector<socket> const& ndv, std::string const& pool_name, bool exclusive)
    {
        for (socket const& d : ndv) { add_resource(d, pool_name, exclusive); }
    }

    void partitioner::set_scheduler(scheduling_policy sched, std::string const& pool_name)
    {
        std::unique_lock<mutex_type> l(mtx_);
        get_pool_data(l, pool_name).scheduling_policy_ = sched;
    }

    void partitioner::configure_pools()
    {
        setup_pools();
        setup_schedulers();
        reconfigure_affinities();
    }

    ////////////////////////////////////////////////////////////////////////
    // this function is called in the constructor of thread_pool
    // returns a scheduler (moved) that thread pool should have as a data member
    scheduling_policy partitioner::which_scheduler(std::string const& pool_name)
    {
        std::unique_lock<mutex_type> l(mtx_);

        // look up which scheduler is needed
        scheduling_policy sched_type = get_pool_data(l, pool_name).scheduling_policy_;
        if (sched_type == unspecified)
        {
            l.unlock();
            throw std::invalid_argument("partitioner::which_scheduler: Thread pool " + pool_name +
                " cannot be instantiated with unspecified scheduler type.");
        }
        return sched_type;
    }

    threads::detail::topology& partitioner::get_topology() const { return topo_; }

    std::size_t partitioner::get_num_threads() const
    {
        std::size_t num_threads = 0;

        {
            std::unique_lock<mutex_type> l(mtx_);
            std::size_t num_thread_pools = initial_thread_pools_.size();
            for (size_t i = 0; i != num_thread_pools; ++i)
            {
                num_threads += get_pool_data(l, i).num_threads_;
            }
        }

        // the number of allocated threads should be the same as the number of
        // threads to create (if no over-subscription is allowed)
        PIKA_ASSERT(mode_ & mode_allow_oversubscription ||
            num_threads ==
                ::pika::detail::get_entry_as<std::size_t>(
                    rtcfg_, "pika.os_threads", std::size_t(-1)));

        return num_threads;
    }

    std::size_t partitioner::get_num_pools() const
    {
        std::lock_guard<mutex_type> l(mtx_);
        return initial_thread_pools_.size();
    }

    std::size_t partitioner::get_num_threads(std::size_t pool_index) const
    {
        std::unique_lock<mutex_type> l(mtx_);
        return get_pool_data(l, pool_index).num_threads_;
    }

    std::size_t partitioner::get_num_threads(std::string const& pool_name) const
    {
        std::unique_lock<mutex_type> l(mtx_);
        return get_pool_data(l, pool_name).num_threads_;
    }

    pika::threads::scheduler_mode partitioner::get_scheduler_mode(std::size_t pool_index) const
    {
        std::unique_lock<mutex_type> l(mtx_);
        return get_pool_data(l, pool_index).mode_;
    }

    detail::init_pool_data const& partitioner::get_pool_data(
        std::unique_lock<mutex_type>& l, std::size_t pool_index) const
    {
        if (pool_index >= initial_thread_pools_.size())
        {
            l.unlock();
            throw_invalid_argument("partitioner::get_pool_data",
                "pool index " + std::to_string(pool_index) +
                    " too large: the resource partitioner owns only " +
                    std::to_string(initial_thread_pools_.size()) + " thread pools.");
        }
        return initial_thread_pools_[pool_index];
    }

    std::string const& partitioner::get_pool_name(std::size_t index) const
    {
        if (index >= initial_thread_pools_.size())
        {
            throw_invalid_argument("partitioner::get_pool_name: ",
                "pool " + std::to_string(index) +
                    " (zero-based index) requested out of bounds. The partitioner owns only " +
                    std::to_string(initial_thread_pools_.size()) + " pools");
        }
        return initial_thread_pools_[index].pool_name_;
    }

    size_t partitioner::get_pu_num(std::size_t global_thread_num)
    {
        return affinity_data_.get_pu_num(global_thread_num);
    }

    threads::detail::mask_cref_type partitioner::get_pu_mask(std::size_t global_thread_num) const
    {
        return affinity_data_.get_pu_mask(topo_, global_thread_num);
    }

    void partitioner::init(resource::partitioner_mode rpmode, pika::detail::section rtcfg,
        pika::detail::affinity_data affinity_data)
    {
        mode_ = rpmode;
        rtcfg_ = rtcfg;
        affinity_data_ = affinity_data;

        fill_topology_vectors();
    }

    scheduler_function partitioner::get_pool_creator(std::size_t index) const
    {
        std::unique_lock<mutex_type> l(mtx_);
        if (index >= initial_thread_pools_.size())
        {
            l.unlock();
            throw std::invalid_argument(
                "partitioner::get_pool_creator: pool requested out of bounds.");
        }
        return get_pool_data(l, index).create_function_;
    }

    ///////////////////////////////////////////////////////////////////////////
    void partitioner::assign_pu(std::string const& pool_name, std::size_t virt_core)
    {
        std::unique_lock<mutex_type> l(mtx_);
        detail::init_pool_data& data = get_pool_data(l, pool_name);
        data.assign_pu(virt_core);
    }

    void partitioner::unassign_pu(std::string const& pool_name, std::size_t virt_core)
    {
        std::unique_lock<mutex_type> l(mtx_);
        detail::init_pool_data& data = get_pool_data(l, pool_name);
        data.unassign_pu(virt_core);
    }

    std::size_t partitioner::shrink_pool(
        std::string const& pool_name, util::detail::function<void(std::size_t)> const& remove_pu)
    {
        if (!(mode_ & mode_allow_dynamic_pools))
        {
            PIKA_THROW_EXCEPTION(pika::error::bad_parameter, "partitioner::shrink_pool",
                "dynamic pools have not been enabled for the partitioner");
        }

        std::vector<std::size_t> pu_nums_to_remove;
        bool has_non_exclusive_pus = false;

        {
            std::unique_lock<mutex_type> l(mtx_);
            detail::init_pool_data const& data = get_pool_data(l, pool_name);

            pu_nums_to_remove.reserve(data.num_threads_);

            for (std::size_t i = 0; i != data.num_threads_; ++i)
            {
                if (!data.pu_is_exclusive(i))
                {
                    has_non_exclusive_pus = true;
                    if (data.pu_is_assigned(i)) { pu_nums_to_remove.push_back(i); }
                }
            }
        }

        if (!has_non_exclusive_pus)
        {
            PIKA_THROW_EXCEPTION(pika::error::bad_parameter, "partitioner::shrink_pool",
                "pool '{}' has no non-exclusive pus associated", pool_name);
        }

        for (std::size_t pu_num : pu_nums_to_remove) { remove_pu(pu_num); }

        return pu_nums_to_remove.size();
    }

    std::size_t partitioner::expand_pool(
        std::string const& pool_name, util::detail::function<void(std::size_t)> const& add_pu)
    {
        if (!(mode_ & mode_allow_dynamic_pools))
        {
            PIKA_THROW_EXCEPTION(pika::error::bad_parameter, "partitioner::expand_pool",
                "dynamic pools have not been enabled for the partitioner");
        }

        std::vector<std::size_t> pu_nums_to_add;
        bool has_non_exclusive_pus = false;

        {
            std::unique_lock<mutex_type> l(mtx_);
            detail::init_pool_data const& data = get_pool_data(l, pool_name);

            pu_nums_to_add.reserve(data.num_threads_);

            for (std::size_t i = 0; i != data.num_threads_; ++i)
            {
                if (!data.pu_is_exclusive(i))
                {
                    has_non_exclusive_pus = true;
                    if (!data.pu_is_assigned(i)) { pu_nums_to_add.push_back(i); }
                }
            }
        }

        if (!has_non_exclusive_pus)
        {
            PIKA_THROW_EXCEPTION(pika::error::bad_parameter, "partitioner::expand_pool",
                "pool '{}' has no non-exclusive pus associated", pool_name);
        }

        for (std::size_t pu_num : pu_nums_to_add) { add_pu(pu_num); }

        return pu_nums_to_add.size();
    }

    ////////////////////////////////////////////////////////////////////////
    std::size_t partitioner::get_pool_index(std::string const& pool_name) const
    {
        // the default pool is always index 0, it may be renamed
        // but the user can always ask for "default"
        if (pool_name == "default") { return 0; }
        {
            std::lock_guard<mutex_type> l(mtx_);
            std::size_t num_pools = initial_thread_pools_.size();
            for (std::size_t i = 0; i < num_pools; i++)
            {
                if (initial_thread_pools_[i].pool_name_ == pool_name) { return i; }
            }
        }

        throw_invalid_argument("partitioner::get_pool_index",
            "the resource partitioner does not own a thread pool named '" + pool_name + "'");
    }

    // has to be private bc pointers become invalid after data member
    // thread_pools_ is resized we don't want to allow the user to use it
    detail::init_pool_data const& partitioner::get_pool_data(
        std::unique_lock<mutex_type>& l, std::string const& pool_name) const
    {
        auto pool = std::find_if(initial_thread_pools_.begin(), initial_thread_pools_.end(),
            [&pool_name](detail::init_pool_data const& itp) -> bool {
                return (itp.pool_name_ == pool_name);
            });

        if (pool != initial_thread_pools_.end()) { return *pool; }

        l.unlock();
        throw_invalid_argument("partitioner::get_pool_data",
            "the resource partitioner does not own a thread pool named '" + pool_name + "'");
    }

    detail::init_pool_data& partitioner::get_pool_data(
        std::unique_lock<mutex_type>& l, std::string const& pool_name)
    {
        auto pool = std::find_if(initial_thread_pools_.begin(), initial_thread_pools_.end(),
            [&pool_name](detail::init_pool_data const& itp) -> bool {
                return (itp.pool_name_ == pool_name);
            });

        if (pool != initial_thread_pools_.end()) { return *pool; }

        l.unlock();
        throw_invalid_argument("partitioner::get_pool_data",
            "the resource partitioner does not own a thread pool named '" + pool_name + "'");
    }

    void partitioner::print_init_pool_data(std::ostream& os) const
    {
        std::lock_guard<mutex_type> l(mtx_);

        //! make this prettier
        os << "the resource partitioner owns " << initial_thread_pools_.size()
           << " pool(s) : \n";    // -V128
        for (auto itp : initial_thread_pools_) { itp.print_pool(os); }
    }

    ////////////////////////////////////////////////////////////////////////
    std::atomic<int> partitioner::instance_number_counter_(-1);
}    // namespace pika::resource::detail
