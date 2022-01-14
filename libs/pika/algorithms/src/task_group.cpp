//  Copyright (c) 2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/local/config.hpp>
#include <pika/assert.hpp>
#include <pika/futures/traits/future_access.hpp>
#include <pika/lcos_local/detail/preprocess_future.hpp>
#include <pika/modules/errors.hpp>
#include <pika/modules/serialization.hpp>
#include <pika/parallel/task_group.hpp>

#include <exception>

namespace pika::execution::experimental {

    ///////////////////////////////////////////////////////////////////////////
    task_group::on_exit::on_exit(task_group& tg)
      : latch_(&tg.latch_)
    {
        if (latch_->reset_if_needed_and_count_up(1, 1))
        {
            tg.has_arrived_.store(false, std::memory_order_release);
        }
    }

    task_group::on_exit::~on_exit()
    {
        if (latch_)
        {
            latch_->count_down(1);
        }
    }

    task_group::on_exit::on_exit(on_exit&& rhs) noexcept
      : latch_(rhs.latch_)
    {
        rhs.latch_ = nullptr;
    }

    task_group::on_exit& task_group::on_exit::operator=(on_exit&& rhs) noexcept
    {
        latch_ = rhs.latch_;
        rhs.latch_ = nullptr;
        return *this;
    }

    ///////////////////////////////////////////////////////////////////////////
    task_group::task_group()
      : latch_(1)
      , has_arrived_(false)
    {
    }

#if defined(PIKA_DEBUG)
    task_group::~task_group()
    {
        // wait() must have been called
        PIKA_ASSERT(latch_.is_ready());
    }
#else
    task_group::~task_group() = default;
#endif

    void task_group::wait()
    {
        bool expected = false;
        if (has_arrived_.compare_exchange_strong(expected, true))
        {
            latch_.arrive_and_wait();
            if (errors_.size() != 0)
            {
                throw errors_;
            }

            auto state = PIKA_MOVE(state_);
            if (state)
            {
                state->set_value(pika::util::unused);
            }
        }
    }

    void task_group::add_exception(std::exception_ptr p)
    {
        errors_.add(PIKA_MOVE(p));
    }

    void task_group::serialize(
        serialization::output_archive& ar, unsigned const)
    {
        if (!latch_.is_ready())
        {
            if (ar.is_preprocessing())
            {
                using init_no_addref =
                    typename shared_state_type::init_no_addref;
                state_.reset(new shared_state_type(init_no_addref{}), false);
                preprocess_future(ar, *state_);
            }
            else
            {
                PIKA_THROW_EXCEPTION(invalid_status, "task_group::serialize",
                    "task_group must be ready in order for it to be "
                    "serialized");
            }
            return;
        }

        // the state is not needed anymore
        state_.reset();
    }

    void task_group::serialize(serialization::input_archive&, unsigned const) {}
}    // namespace pika::execution::experimental
