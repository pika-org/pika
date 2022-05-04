<!--- Copyright (c) 2022 ETH Zurich -->
<!----->
<!--- SPDX-License-Identifier: BSL-1.0 -->
<!--- Distributed under the Boost Software License, Version 1.0. (See accompanying -->
<!--- file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt) -->

## 0.4.0 (2022-05-04)

### New features

- Annotations are now inherited from the scheduling task instead of the spawning task for `transfer`. ([#188](https://github.com/pika-org/pika/pull/188))
- Annotations for bulk regions are now lifted up around the work loop on each worker thread using a scoped annotation. ([#197](https://github.com/pika-org/pika/pull/197))
- It is now allowed to pass a lambda with `auto` parameters to `then`. ([#182](https://github.com/pika-org/pika/pull/182))
- A scheduler that spawns `std::thread`s is now available. ([#200](https://github.com/pika-org/pika/pull/200))

### Breaking changes

- Cleaned up various modules and moved minor functionality into `detail` namespaces. ([#179](https://github.com/pika-org/pika/pull/179), [#196](https://github.com/pika-org/pika/pull/196))

### Bugfixes

- The sender returned by `ensure_started` can now be discarded without being connected to its corresponding receiver. ([#180](https://github.com/pika-org/pika/pull/180))
- The lifetime issue in the `split` sender adaptor is now fixed. ([#203](https://github.com/pika-org/pika/pull/203))
- The scheduler forwarding in `schedule_from` is now properly handled. ([#186](https://github.com/pika-org/pika/pull/186))
- Missing includes to `transform_mpi.hpp` have now been added. ([#176](https://github.com/pika-org/pika/pull/176))
- Remove unnecessary Boost dependencies when `PIKA_WITH_GENERIC_CONTEXT_COROUTINES=ON`. ([#185](https://github.com/pika-org/pika/pull/185))
- The segmentation fault in the shared priority queue scheduler has now been fixed. ([#210](https://github.com/pika-org/pika/pull/210))

## 0.3.0 (2022-04-06)

### New features

- Using `pika::mpi::experimental::transform_mpi` in debug mode now checks that polling has been enabled. ([#142](https://github.com/pika-org/pika/pull/142))
- `pika_main` no longer requires non-const `argc` and `argv`. ([#146](https://github.com/pika-org/pika/pull/146))

### Breaking changes

- `pika::execution::experimental::ensure_started` no longer sends const references to receivers, like `pika::execution::experimental::split`. It now sends values by rvalue reference. ([#143](https://github.com/pika-org/pika/pull/143))
- All serialization support has been removed. ([#145](https://github.com/pika-org/pika/pull/145), [#150](https://github.com/pika-org/pika/pull/150))
- `pika::bind_front` no longer unwraps `std::reference_wrapper`s to match the behaviour of `std::bind_front`. ([#140](https://github.com/pika-org/pika/pull/140))
- Cleaned up various modules and moved minor functionality into `detail` namespaces. ([#152](https://github.com/pika-org/pika/pull/152), [#153](https://github.com/pika-org/pika/pull/153), [#155](https://github.com/pika-org/pika/pull/155), [#158](https://github.com/pika-org/pika/pull/158), [#160](https://github.com/pika-org/pika/pull/160))
- Move `pika::execution::experimental::sync_wait` to `pika::this_thread::experimental::sync_wait` to match the namespace of `sync_wait` in P2300. ([#159](https://github.com/pika-org/pika/pull/159))

### Bugfixes

- `pika::execution::experimental::make_future` releases its operation state as soon as the wrapped sender has signaled completion. ([#139](https://github.com/pika-org/pika/pull/139))
- `pika::cuda::experimental::then_with_stream` now correctly checks for invocability of the given callable with lvalue references. ([#144](https://github.com/pika-org/pika/pull/144))
- `pika::mpi::experimental::transform_mpi` now stores the values sent from the predecessor in the operation state to ensure stable references. ([#156](https://github.com/pika-org/pika/pull/156))
- `pika::cuda::experimental::then_with_stream` now stores the values sent from the predecessor in the operation state to ensure stable references. ([#162](https://github.com/pika-org/pika/pull/162))
- Tasks scheduled with `pika::execution::experimental::thread_pool_scheduler` now use the annotation of the spawning task as a fallback if no explicit annotation has been given. ([#173](https://github.com/pika-org/pika/pull/173))

## 0.2.0 (2022-03-08)

### New features

- Added a P2300 `cuda_scheduler` along with various helper functionalities. ([#37](https://github.com/pika-org/pika/pull/37), [#128](https://github.com/pika-org/pika/pull/128))
- Re-enabled support for APEX. ([#104](https://github.com/pika-org/pika/pull/104))
- Added P2300 scheduler queries. ([#102](https://github.com/pika-org/pika/pull/102))
- Added top-level API headers for CUDA, MPI, and added thread manager, resource partitioner functionality to `pika/runtime.hpp` header. ([#117](https://github.com/pika-org/pika/pull/117), [#123](https://github.com/pika-org/pika/pull/123))
- Added `when_all_vector`, a variant of `when_all` that works with vectors of senders. ([#109](https://github.com/pika-org/pika/pull/109), [#132](https://github.com/pika-org/pika/pull/132))

### Breaking changes

- Bumped the minimum required compiler versions to GCC 9 and clang 9. ([#70](https://github.com/pika-org/pika/pull/70))
- Removed the `filesystem` compatibility layer based on Boost.Filesystem. `std::filesystem` support is now required from the standard library. ([#70](https://github.com/pika-org/pika/pull/70))
- Changed the default value of the configuration option `pika.exception_verbosity` to `1` (previously `2`). Exceptions will now by default not print the pika configuration and environment variables. ([#99](https://github.com/pika-org/pika/pull/99))
- Yielding of pika threads is now disallowed with uncaught exceptions (with an assertion) to prevent hard-to-debug errors when thread stealing is enabled. ([#112](https://github.com/pika-org/pika/pull/112))

### Bugfixes

- pika threads are now again rescheduled on the worker thread where they are suspended. ([#110](https://github.com/pika-org/pika/pull/110))
- Fixed a bug in the reference counting of the shared state in `ensure_started` and `split` that prevented it from being freed. ([#111](https://github.com/pika-org/pika/pull/111))
- Fixed deadlocks in `stop_token`. ([#113](https://github.com/pika-org/pika/pull/113))

## 0.1.0 (2022-01-31)

This is the first release of pika.
