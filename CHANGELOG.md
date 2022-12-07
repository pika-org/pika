<!--- Copyright (c) 2022 ETH Zurich -->
<!----->
<!--- SPDX-License-Identifier: BSL-1.0 -->
<!--- Distributed under the Boost Software License, Version 1.0. (See accompanying -->
<!--- file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt) -->

## 0.11.0 (2022-12-07)

### New features

### Breaking changes

- All parallel algorithms have been moved to a new repository that depends on pika: https://github.com/pika-org/pika-algorithms. ([#505](https://github.com/pika-org/pika/pull/505))
- fmt is now a required dependency. ([#487](https://github.com/pika-org/pika/pull/487))
- The default allocator has been changed from tcmalloc to mimalloc. ([#501](https://github.com/pika-org/pika/pull/501))
- Cleaned up various modules and moved minor functionality into `detail` namespaces. ([#483](https://github.com/pika-org/pika/pull/483), [#508](https://github.com/pika-org/pika/pull/508), [#509](https://github.com/pika-org/pika/pull/509))

### Bugfixes

## 0.10.0 (2022-11-02)

### New features

### Breaking changes

- More functionality in the `algorithms` module has been moved into `detail` namespaces. ([#475](https://github.com/pika-org/pika/pull/475))

### Bugfixes

- Many sender adaptors have been updated to correctly handle reference types. ([#472](https://github.com/pika-org/pika/pull/472), [#484](https://github.com/pika-org/pika/pull/484), [#492](https://github.com/pika-org/pika/pull/492), )
- `then_with_stream` now correctly stores the values sent by the predecessor sender for the duration of the CUDA operation launched by it. ([#485](https://github.com/pika-org/pika/pull/485))

## 0.9.0 (2022-10-05)

### New features

- Signal handlers are now optional, they can be set with `--pika:install_signal_handlers=1`. They are enabled by default when `--pika:attach-debugger=exception` is set. ([#458](https://github.com/pika-org/pika/pull/458))

### Breaking changes

- The P2300 reference implementation is now found through a `find_package` instead of a `fetch_content` in CMake and is required when `PIKA_WITH_P2300_REFERENCE_IMPLEMENTATION` in `ON`. ([#436](https://github.com/pika-org/pika/pull/436))
- [whip](https://github.com/eth-cscs/whip) is now a dependency to replace the GPU abstraction layer we previously used. ([#423](https://github.com/pika-org/pika/pull/423))
- Use rocBLAS directly instead of hipBLAS. ([#391](https://github.com/pika-org/pika/pull/391))
- Move more internal functionality into the `detail` namespace. ([#445](https://github.com/pika-org/pika/pull/445), [#446](https://github.com/pika-org/pika/pull/446), [#449](https://github.com/pika-org/pika/pull/449), [#461](https://github.com/pika-org/pika/pull/461), [#462](https://github.com/pika-org/pika/pull/462))

### Bugfixes

- Add `set_stopped_t()` to `(unique_)any_sender` completion signatures. ([#464](https://github.com/pika-org/pika/pull/464))
- Fix compilation on Arm64 with `PIKA_WITH_GENERIC_CONTEXT_COROUTINES=OFF`. ([#439](https://github.com/pika-org/pika/pull/439))
- Add a missing default entry for `pika.diagnostics_on_terminate`. ([#458](https://github.com/pika-org/pika/pull/458))

## 0.8.0 (2022-09-07)

### New features

- The `PIKA_WITH_P2300_REFERENCE_IMPLEMENTATION` option can now be enabled together with `PIKA_WITH_CUDA` (with clang as the device compiler) and `PIKA_WITH_HIP`. ([#330](https://github.com/pika-org/pika/pull/330))
- CMake options related to tests and examples now use `cmake_dependent_option` where appropriate. This means that options like `PIKA_WITH_TESTS_UNIT` will correctly be enabled when reconfiguring with `PIKA_WITH_TESTS=ON` even if pika was initially configured with `PIKA_WITH_TESTS=OFF`. ([#356](https://github.com/pika-org/pika/pull/356))
- `pika::finalize` no longer has to be called on a pika thread. ([#366](https://github.com/pika-org/pika/pull/366))

### Breaking changes

- Removed `operator|` overloads for `sync_wait` and `start_detached` to align the implementation with P2300. ([#346](https://github.com/pika-org/pika/pull/346))
- Removed `parallel_executor_aggregated`. ([#372](https://github.com/pika-org/pika/pull/372))
- Moved more internal functionality into the `detail` namespace. ([#374](https://github.com/pika-org/pika/pull/374), [#377](https://github.com/pika-org/pika/pull/377), [#379](https://github.com/pika-org/pika/pull/379), [#386](https://github.com/pika-org/pika/pull/386), [#400](https://github.com/pika-org/pika/pull/400), [#411](https://github.com/pika-org/pika/pull/411), [#420](https://github.com/pika-org/pika/pull/420), [#428](https://github.com/pika-org/pika/pull/428), [#429](https://github.com/pika-org/pika/pull/429))
- Allow compiling only device code with `hipcc` when `PIKA_WITH_HIP` is enabled instead of requiring `hipcc` to be used for host code as well. The `PIKA_WITH_HIP` option now has to be enabled explicitly like CUDA support instead of being automatically detected with `hipcc` set as the C++ compiler. ([#402](https://github.com/pika-org/pika/pull/402))

### Bugfixes

- Fixed handling of reference types in `ensure_started` and `let_error`. ([#338](https://github.com/pika-org/pika/pull/338))
- Fixed compilation for powerpc. ([#341](https://github.com/pika-org/pika/pull/341))
- Correctly set the stream in `cusolver_handle::set_stream`. ([#344](https://github.com/pika-org/pika/pull/344))
- Fix the `--pika:ignore-process-mask` command line option. It was previously being ignored. ([#355](https://github.com/pika-org/pika/pull/355))
- Fix a visibility issue in the `program_options` implementation. ([#359](https://github.com/pika-org/pika/pull/359))
- Change detection of builtins to be more robust against mixing compilers. ([#390](https://github.com/pika-org/pika/pull/390))
- Fixed compilation for arm64. ([#393](https://github.com/pika-org/pika/pull/393))
- Only check for `CMAKE_CUDA_STANDARD` and `PIKA_WITH_CXX_STANDARD` when building pika itself. This could previously lead to false positive configuration errors. ([#396](https://github.com/pika-org/pika/pull/396))
- Fix compilation on macOS with `PIKA_WITH_MPI` enabled. ([#405](https://github.com/pika-org/pika/pull/405))

## 0.7.0 (2022-08-03)

### New features

- The CUDA polling now uses both normal and high priority queues based on the flags passed to the `cuda_stream`. ([#286](https://github.com/pika-org/pika/pull/286))
- Eagerly check completion of the MPI requests and add throttling of MPI traffic to help prevent excessive message queues. ([#277](https://github.com/pika-org/pika/pull/277))
- Eagerly check completion of CUDA kernels. ([#306](https://github.com/pika-org/pika/pull/306))

### Breaking changes

- Remove static and thread local storage emulation. ([#321](https://github.com/pika-org/pika/pull/321))
- Moved internal functionality into the `detail` namespace. ([#209](https://github.com/pika-org/pika/pull/209), [#276](https://github.com/pika-org/pika/pull/276), [#324](https://github.com/pika-org/pika/pull/324))
- Remove specialization for Intel KNL. ([#309](https://github.com/pika-org/pika/pull/309))

### Bugfixes

- Fix a compilation error with posix coroutines implementation. ([#314](https://github.com/pika-org/pika/pull/314))
- Fix handling of reference values and errors types sent by predecessors to `when_all`, `split` and `sync_wait`. ([#285](https://github.com/pika-org/pika/pull/285), [#307](https://github.com/pika-org/pika/pull/307), [#320](https://github.com/pika-org/pika/pull/320))

## 0.6.0 (2022-07-06)

### New features

- Added basic support for Tracy. The Tracy integration can currently only be used with threads that do not yield. ([#252](https://github.com/pika-org/pika/pull/252))
- Added `make_any_sender` and `make_unique_any_sender` helpers for deducing the template parameters of `any_sender` and `unique_any_sender`. ([#259](https://github.com/pika-org/pika/pull/259))
- Added a `drop_value` sender adaptor which ignores values sent from the predecessor. ([#262](https://github.com/pika-org/pika/pull/262))
- Allow passing flags to the `cuda_stream` and `cuda_pool` constructor. ([#270](https://github.com/pika-org/pika/pull/270))
- Allow using any version of mimalloc. The version was previously unnecessarily constrained to 1. ([#273](https://github.com/pika-org/pika/pull/273))
- Further relax the requirements for constness on `argc` and `argv` passed to `pika::init` and `pika::start`. ([#275](https://github.com/pika-org/pika/pull/275))

### Breaking changes

- If a process mask is set the pika runtime now uses the mask by default to restrict the number of threads. The command-line option `--pika:use-process-mask` which was previously used to enable this behaviour has been removed along with the corresponding configuration option. The process mask can be explicitly ignored with the command-line option `--pika:ignore-process-mask` or the configuration option `pika.ignore_process_mask`. ([#242](https://github.com/pika-org/pika/pull/242))
- Moved internal functionality into the `detail` namespace. ([#246](https://github.com/pika-org/pika/pull/246), [#248](https://github.com/pika-org/pika/pull/248), [#257](https://github.com/pika-org/pika/pull/257))

### Bugfixes

- Fix handling of reference types sent by predecessors to `ensure_started` and `schedule_from`. ([#282](https://github.com/pika-org/pika/pull/282))

## 0.5.0 (2022-06-02)

### New features

- The `then_with_cublas` and `then_with_cusolver` sender adaptors can now also be used with hipBLAS and hipSOLVER. ([#220](https://github.com/pika-org/pika/pull/220))
- There is now experimental support for using the [P2300 reference implementation](https://github.com/brycelelbach/wg21_p2300_std_execution) in place of pika's own implementation. This can be enabled with the `PIKA_WITH_P2300_REFERENCE_IMPLEMENTATION` CMake option. ([#215](https://github.com/pika-org/pika/pull/215))

### Breaking changes

- The `--pika:help` command-line no longer takes any arguments. ([#219](https://github.com/pika-org/pika/pull/219))
- Vc support has been removed. ([#223](https://github.com/pika-org/pika/pull/223))
- Cleaned up the `command_line_handling` module and moved minor functionality into the `detail` namespace. ([#216](https://github.com/pika-org/pika/pull/216))
- Removed the `then_with_any_cuda` sender adaptor. ([#243](https://github.com/pika-org/pika/pull/243))

### Bugfixes

- Scheduler properties can now be used with `prefer`. ([#214](https://github.com/pika-org/pika/pull/214))

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
