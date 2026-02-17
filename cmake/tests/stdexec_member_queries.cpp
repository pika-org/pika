//  Copyright (c) 2026 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <stdexec/execution.hpp>

// Earlier versions of stdexec only support tag_invoke for customization points. Newer versions
// support a query() member function protocol. This test checks if get_completion_scheduler_t
// works with the member query protocol.
struct my_scheduler
{
    struct sender
    {
        using sender_concept = stdexec::sender_t;
        using completion_signatures = stdexec::completion_signatures<stdexec::set_value_t()>;

        struct env
        {
            template <class Tag>
            my_scheduler query(stdexec::get_completion_scheduler_t<Tag>) const noexcept
            {
                return {};
            }
        };

        env get_env() const noexcept { return {}; }
    };

    using scheduler_concept = stdexec::scheduler_t;

    sender schedule() const noexcept { return {}; }

    bool operator==(my_scheduler const&) const noexcept = default;
};

int main()
{
    my_scheduler s;
    auto snd = s.schedule();
    [[maybe_unused]] auto cs =
        stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(snd));
}
