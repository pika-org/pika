//  Copyright (c) 2015-2016 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <plugins/parcelport/libfabric/parcelport_libfabric.hpp>

// TODO: cleanup includes

// config
#include <hpx/config.hpp>
// util
#include <hpx/synchronization/condition_variable.hpp>
#include <hpx/threading_base/thread_data.hpp>
#include <hpx/command_line_handling/command_line_handling.hpp>
#include <hpx/timing/high_resolution_timer.hpp>
#include <hpx/runtime_configuration/runtime_configuration.hpp>

// The memory pool specialization need to be pulled in before encode_parcels
#include <hpx/plugins/parcelport_factory.hpp>
#include <hpx/runtime.hpp>
#include <hpx/runtime/parcelset/decode_parcels.hpp>
#include <hpx/runtime/parcelset/encode_parcels.hpp>
#include <hpx/runtime/parcelset/parcel_buffer.hpp>
#include <hpx/runtime/parcelset/parcelport.hpp>
#include <hpx/runtime/parcelset/parcelport_impl.hpp>
//
#include <hpx/assert.hpp>
#include <hpx/runtime_local/thread_stacktrace.hpp>
//
#include <asio/ip/host_name.hpp>
//
// This header is generated by CMake and contains a number of configurable
// setting that affect the parcelport. It needs to be #included before
// other files that are part of the parcelport
#include <hpx/config/parcelport_defines.hpp>

// --------------------------------------------------------------------
// Controls whether we are allowed to suspend threads that are sending
// when we have maxed out the number of sends we can handle
#define HPX_PARCELPORT_LIBFABRIC_SUSPEND_WAKE  (HPX_PARCELPORT_LIBFABRIC_THROTTLE_SENDS/2)

// --------------------------------------------------------------------
// Enable the use of hpx small_vector for certain short lived storage
// elements within the parcelport. This can reduce some memory allocations
#define HPX_PARCELPORT_LIBFABRIC_USE_SMALL_VECTOR    true

// --------------------------------------------------------------------
#include <plugins/parcelport/unordered_map.hpp>
#include <plugins/parcelport/libfabric/header.hpp>
#include <plugins/parcelport/libfabric/locality.hpp>

#include <plugins/parcelport/libfabric/libfabric_region_provider.hpp>
#include <plugins/parcelport/performance_counter.hpp>
#include <plugins/parcelport/rma_memory_pool.hpp>
#include <plugins/parcelport/libfabric/connection_handler.hpp>
#include <plugins/parcelport/parcelport_logging.hpp>
#include <plugins/parcelport/libfabric/rdma_locks.hpp>
#include <plugins/parcelport/libfabric/libfabric_controller.hpp>

//
#if HPX_PARCELPORT_LIBFABRIC_USE_SMALL_VECTOR
#include <hpx/datastructures/detail/small_vector.hpp>
#endif
//
#include <unordered_map>
#include <memory>
#include <mutex>
#include <sstream>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <list>
#include <string>
#include <utility>
#include <vector>

using namespace hpx::parcelset::policies;

namespace hpx {
namespace parcelset {
namespace policies {
namespace libfabric
{
    // --------------------------------------------------------------------
    // parcelport, the implementation of the parcelport itself
    // --------------------------------------------------------------------

    // --------------------------------------------------------------------
    // Constructor : mostly just initializes the superclass with 'here'
    // --------------------------------------------------------------------
    parcelport::parcelport(util::runtime_configuration const& ini,
        threads::policies::callback_notifier const& notifier)
      : base_type(ini, locality(), notifier)
      , stopped_(false)
      , completions_handled_(0)
      , senders_in_use_(0)
    {
        FUNC_START_DEBUG_MSG;

        // if we are not enabled, then skip allocating resources
        parcelport_enabled_ = hpx::util::get_entry_as<bool>(ini,
            "hpx.parcel.libfabric.enable", 0);
        LOG_DEBUG_MSG("Got enabled " << parcelport_enabled_);

        bootstrap_enabled_ = ("libfabric" ==
            hpx::util::get_entry_as<std::string>(ini, "hpx.parcel.bootstrap", ""));
        LOG_DEBUG_MSG("Got bootstrap " << bootstrap_enabled_);

        if (!parcelport_enabled_) return;

        // Get parameters that determine our fabric selection
        std::string provider = ini.get_entry("hpx.parcel.libfabric.provider",
            HPX_PARCELPORT_LIBFABRIC_PROVIDER);
        std::string domain = ini.get_entry("hpx.parcel.libfabric.domain",
            HPX_PARCELPORT_LIBFABRIC_DOMAIN);
        std::string endpoint = ini.get_entry("hpx.parcel.libfabric.endpoint",
            HPX_PARCELPORT_LIBFABRIC_ENDPOINT);

        LOG_DEBUG_MSG("libfabric parcelport function using attributes "
            << provider << " " << domain << " " << endpoint);

        // create our main fabric control structure
        libfabric_controller_ = std::make_shared<libfabric_controller>(
            provider, domain, endpoint);

        // get 'this' locality from the controller
        LOG_DEBUG_MSG("Getting local locality object");
        const locality & local = libfabric_controller_->here();
        here_ = parcelset::locality(local);
        // and make a note of our ip address for convenience
        ip_addr_ = local.ip_address();

        FUNC_END_DEBUG_MSG;
    }

    // --------------------------------------------------------------------
    // during bootup, this is used by the service threads
    void parcelport::io_service_work()
    {
        while (hpx::is_starting())
        {
            background_work(0, parcelport_background_mode_all);
        }
        LOG_DEBUG_MSG("io service task completed");
    }

    // --------------------------------------------------------------------
    // Start the handling of communication.
    bool parcelport::do_run()
    {
        if (!parcelport_enabled_) return false;

#ifndef HPX_PARCELPORT_LIBFABRIC_HAVE_PMI
        auto &as = naming::get_agas_client();
        libfabric_controller_->initialize_localities(as);
#endif

        FUNC_START_DEBUG_MSG;
        libfabric_controller_->startup(this);

        LOG_DEBUG_MSG("Fetching memory pool");
        chunk_pool_ = &libfabric_controller_->get_memory_pool();

        for (std::size_t i = 0; i < HPX_PARCELPORT_LIBFABRIC_THROTTLE_SENDS; ++i)
        {
            sender *snd =
               new sender(this,
                    libfabric_controller_->ep_active_,
                    libfabric_controller_->get_domain(),
                    chunk_pool_);
            snd->postprocess_handler_ = [this](sender* s)
                {
                    --senders_in_use_;
                    senders_.push(s);
                    trigger_pending_work();
                };
           senders_.push(snd);
        }

        if (bootstrap_enabled_)
        {
            for (std::size_t i = 0; i != io_service_pool_.size(); ++i)
            {
                io_service_pool_.get_io_service(int(i)).post(
                    hpx::util::bind(
                        &parcelport::io_service_work, this));
            }
        }
        return true;
   }

    // --------------------------------------------------------------------
    // return a sender object back to the parcelport_impl
    // this is used by the send_immediate version of parcelport_impl
    // --------------------------------------------------------------------
    sender* parcelport::get_connection(
        parcelset::locality const& dest, fi_addr_t &fi_addr)
    {
        sender* snd = nullptr;
        if (senders_.pop(snd))
        {
            FUNC_START_DEBUG_MSG;
            const locality &fabric_locality = dest.get<locality>();
            LOG_DEBUG_MSG("get_fabric_address           from "
                << ipaddress(here_.get<locality>().ip_address()) << "to "
                << ipaddress(fabric_locality.ip_address()));
            ++senders_in_use_;
            fi_addr = libfabric_controller_->get_fabric_address(fabric_locality);
            FUNC_END_DEBUG_MSG;
            return snd;
        }
        //    else if(threads::get_self_ptr())
        // //    else if(this_thread::has_sufficient_stack_space())
        //    {
        // //        background_work_OS_thread();
        //        hpx::this_thread::suspend(
        //            hpx::threads::thread_schedule_state::pending_boost,
        //            "libfabric::parcelport::async_write");
        //    }

        // if no senders are available shutdown
        FUNC_END_DEBUG_MSG;
        return nullptr;
    }

    void parcelport::reclaim_connection(sender* s)
    {
        --senders_in_use_;
        senders_.push(s);
    }

    // --------------------------------------------------------------------
    // return a sender object back to the parcelport_impl
    // this is for compatibility with non send_immediate operation
    // --------------------------------------------------------------------
    std::shared_ptr<sender> parcelport::create_connection(
        parcelset::locality const& dest, error_code& ec)
    {
        LOG_DEBUG_MSG("Creating new sender");
        return std::shared_ptr<sender>();
    }

    // --------------------------------------------------------------------
    // cleanup
    parcelport::~parcelport() {
        FUNC_START_DEBUG_MSG;
        scoped_lock lk(stop_mutex);
        sender *snd = nullptr;

        unsigned int sends_posted  = 0;
        unsigned int sends_deleted = 0;
        unsigned int acks_received = 0;
        //
        while (senders_.pop(snd)) {
            LOG_DEBUG_MSG("Popped a sender for delete " << hexpointer(snd));
            sends_posted  += snd->sends_posted_;
            sends_deleted += snd->sends_deleted_;
            acks_received += snd->acks_received_;
            delete snd;
        }
        LOG_DEBUG_MSG(
               "sends_posted "  << decnumber(sends_posted)
            << "sends_deleted " << decnumber(sends_deleted)
            << "acks_received " << decnumber(acks_received)
            << "non_rma-send "  << decnumber(sends_posted-acks_received));
        //
        libfabric_controller_ = nullptr;
        FUNC_END_DEBUG_MSG;
    }

    // --------------------------------------------------------------------
    /// Should not be used any more as parcelport_impl handles this?
    bool parcelport::can_bootstrap() const {
        FUNC_START_DEBUG_MSG;
        bool can_boot = HPX_PARCELPORT_LIBFABRIC_HAVE_BOOTSTRAPPING();
        LOG_TRACE_MSG("Returning " << can_boot << " from can_bootstrap")
        FUNC_END_DEBUG_MSG;
        return can_boot;
    }

    // --------------------------------------------------------------------
    /// return a string form of the locality name
    std::string parcelport::get_locality_name() const
    {
        FUNC_START_DEBUG_MSG;
        // return hostname:iblibfabric ip address
        std::stringstream temp;
        temp << asio::ip::host_name() << ":" << ipaddress(ip_addr_);
        std::string tstr = temp.str();
        FUNC_END_DEBUG_MSG;
        return tstr.substr(0, tstr.size()-1);
    }

    // --------------------------------------------------------------------
    // the root node has spacial handling, this returns its Id
    parcelset::locality parcelport::
    agas_locality(util::runtime_configuration const & ini) const
    {
        FUNC_START_DEBUG_MSG;
        // load all components as described in the configuration information
        if (!bootstrap_enabled_)
        {
            LOG_ERROR_MSG("Should only return agas locality when bootstrapping");
        }
        FUNC_END_DEBUG_MSG;
        return libfabric_controller_->agas_;
    }

    // --------------------------------------------------------------------
    parcelset::locality parcelport::create_locality() const {
        FUNC_START_DEBUG_MSG;
        FUNC_END_DEBUG_MSG;
        return parcelset::locality(locality());
    }

    // --------------------------------------------------------------------
    /// for debugging
    void parcelport::suspended_task_debug(const std::string &match)
    {
        std::string temp = hpx::util::debug::suspended_task_backtraces();
        if (match.size()==0 ||
            temp.find(match)!=std::string::npos)
        {
            LOG_DEBUG_MSG("Suspended threads " << temp);
        }
    }

    // --------------------------------------------------------------------
    /// stop the parcelport, prior to shutdown
    void parcelport::do_stop() {
        LOG_DEBUG_MSG("Entering libfabric stop ");
        FUNC_START_DEBUG_MSG;
        if (!stopped_) {
            // we don't want multiple threads trying to stop the clients
            scoped_lock lock(stop_mutex);

            LOG_DEBUG_MSG("Removing all initiated connections");
            libfabric_controller_->disconnect_all();

            // wait for all clients initiated elsewhere to be disconnected
            while (libfabric_controller_->active() /*&& !hpx::is_stopped()*/) {
                completions_handled_ += libfabric_controller_->poll_endpoints(true);
                LOG_TIMED_INIT(disconnect_poll);
                LOG_TIMED_BLOCK(disconnect_poll, DEVEL, 5.0,
                    {
                        LOG_DEBUG_MSG("Polling before shutdown");
                    }
                )
            }
            LOG_DEBUG_MSG("stopped removing clients and terminating");
        }
        stopped_ = true;
        // Stop receiving and sending of parcels
    }

    // --------------------------------------------------------------------
    bool parcelport::can_send_immediate()
    {
        // hpx::util::yield_while([this]()
        //     {
        //         this->background_work(0);
        //         return this->senders_.empty();
        //     }, "libfabric::can_send_immediate");

        return true;
    }

    // --------------------------------------------------------------------
    template <typename Handler>
    bool parcelport::async_write(Handler && handler,
        sender *snd, fi_addr_t addr,
        snd_buffer_type &buffer)
    {
        LOG_DEBUG_MSG("parcelport::async_write using sender " << hexpointer(snd));
        snd->dst_addr_ = addr;
        snd->buffer_ = std::move(buffer);
        HPX_ASSERT(!snd->handler_);
        snd->handler_ = std::forward<Handler>(handler);
        snd->async_write_impl();
        // after a send poll to make progress on the network and
        // reduce latencies for receives coming in
        // background_work_OS_thread();
        // if (hpx::threads::get_self_ptr())
        //     hpx::this_thread::suspend(
        //         hpx::threads::thread_schedule_state::pending_boost,
        //         "libfabric::parcelport::async_write");
        return true;
    }

    // --------------------------------------------------------------------
    // This is called to poll for completions and handle all incoming messages
    // as well as complete outgoing messages.
    //
    // Since the parcelport can be serviced by hpx threads or by OS threads,
    // we must use extra care when dealing with mutexes and condition_variables
    // since we do not want to suspend an OS thread, but we do want to suspend
    // hpx threads when necessary.
    //
    // NB: There is no difference any more between background polling work
    // on OS or HPX as all has been tested thoroughly
    // --------------------------------------------------------------------
    inline bool parcelport::background_work_OS_thread() {
        LOG_TIMED_INIT(background);
        bool done = false;
        do {
            LOG_TIMED_BLOCK(background, DEVEL, 5.0, {
                LOG_DEBUG_MSG("number of senders in use "
                    << decnumber(senders_in_use_));
            });
            // if an event comes in, we may spend time processing/handling it
            // and another may arrive during this handling,
            // so keep checking until none are received
            // libfabric_controller_->refill_client_receives(false);
            int numc = libfabric_controller_->poll_endpoints();
            completions_handled_ += numc;
            done = (numc==0);
        } while (!done);
        return (done!=0);
    }

    // --------------------------------------------------------------------
    // Background work
    //
    // This is called whenever the main thread scheduler is idling,
    // is used to poll for events, messages on the libfabric connection
    // --------------------------------------------------------------------
    bool parcelport::background_work(
        std::size_t num_thread, parcelport_background_mode mode)
    {
        if (stopped_ || hpx::is_stopped()) {
            return false;
        }
        return background_work_OS_thread();
    }
}}}}

HPX_REGISTER_PARCELPORT(hpx::parcelset::policies::libfabric::parcelport, libfabric)
