<!--- Copyright (c) 2022 ETH Zurich -->
<!----->
<!--- SPDX-License-Identifier: BSL-1.0 -->
<!--- Distributed under the Boost Software License, Version 1.0. (See accompanying -->
<!--- file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt) -->

## 0.2.0 (2022-02-TODO)

### New features

- Added a P2300 `cuda_scheduler` along with various helper functionalities.
  ([#37](https://github.com/pika-org/pika/pull/37),
  [#128](https://github.com/pika-org/pika/pull/128))
- Re-enabled support for APEX.
  ([#104](https://github.com/pika-org/pika/pull/104))
- Added P2300 scheduler queries.
  ([#102](https://github.com/pika-org/pika/pull/102))
- Added top-level API headers for CUDA, MPI, and added thread manager, resource
  partitioner functionality to `pika/runtime.hpp` header.
  ([#117](https://github.com/pika-org/pika/pull/117),
  [#123](https://github.com/pika-org/pika/pull/123))
- Added `when_all_vector`, a variant of `when_all` that works with vectors of
  senders. ([#109](https://github.com/pika-org/pika/pull/109),
  [#132](https://github.com/pika-org/pika/pull/132))

### Breaking changes

- Bumped the minimum required compiler versions to GCC 9 and clang 9.
  ([#70](https://github.com/pika-org/pika/pull/70))
- Removed the `filesystem` compatibility layer based on Boost.Filesystem.
  `std::filesystem` support is now required from the standard library.
  ([#70](https://github.com/pika-org/pika/pull/70))
- Changed the default value of the configuration option
  `pika.exception_verbosity` to `1` (previously `2`). Exceptions will now by
  default not print the pika configuration and environment variables.
  ([#99](https://github.com/pika-org/pika/pull/99))
- Yielding of pika threads is now disallowed with uncaught exceptions (with an
  assertion) to prevent hard-to-debug errors when thread stealing is enabled.
  ([#112](https://github.com/pika-org/pika/pull/112))

### Bugfixes

- pika threads are now again rescheduled on the worker thread where they are
  suspended. ([#110](https://github.com/pika-org/pika/pull/110))
- Fixed a bug in the reference counting of the shared state in `ensure_started`
  and `split` that prevented it from being freed.
  ([#111](https://github.com/pika-org/pika/pull/111))
- Fixed deadlocks in `stop_token`.
  ([#113](https://github.com/pika-org/pika/pull/113))

## 0.1.0 (2022-01-31)

This is the first release of pika.
