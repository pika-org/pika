<!--- Copyright (c) 2022-2024 ETH Zurich -->
<!----->
<!--- SPDX-License-Identifier: BSL-1.0 -->
<!--- Distributed under the Boost Software License, Version 1.0. (See accompanying -->
<!--- file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt) -->

# Changelog

## 0.24.0 (2024-04-12)

### New features

- Avoid unnecessary copies of CUDA streams and handles to improve profiling appearance. ([#1056](https://github.com/pika-org/pika/pull/1056))
- Avoid use of `std::invoke_result` in `tag_invoke_result` variants to improve compilation times. ([#1058](https://github.com/pika-org/pika/pull/1058), [#1060](https://github.com/pika-org/pika/pull/1060))

### Breaking changes

### Bugfixes

- Fix use of `--pika:print-bind` with `--pika:bind=none`. ([#1082](https://github.com/pika-org/pika/pull/1082), [#1087](https://github.com/pika-org/pika/pull/1087))
- Work around compilation issue with CUDA 12.4. ([#1084](https://github.com/pika-org/pika/pull/1084))
- Make sure `main` is never defined in `libpika.so`. ([#1088](https://github.com/pika-org/pika/pull/1088))

## 0.23.0 (2024-03-07)

### New features

- Further improved performance, particularly on ARM64 systems. ([#1023](https://github.com/pika-org/pika/pull/1023), [#1033](https://github.com/pika-org/pika/pull/1033), [#1035](https://github.com/pika-org/pika/pull/1035), [#1041](https://github.com/pika-org/pika/pull/1041))
- Allow compilation on ARM systems with address sanitizer enabled. ([#1045](https://github.com/pika-org/pika/pull/1045))

### Breaking changes

### Bugfixes

- Allow the use of the `require_started` sender adaptor with `unique_any_sender` and `any_sender`. ([#1044](https://github.com/pika-org/pika/pull/1044))
- Fix a data race in CUDA/HIP event polling. ([#1051](https://github.com/pika-org/pika/pull/1051))

## 0.22.2 (2024-02-09)

### Bugfixes

- Fix incorrect worker thread numbers when using more than one thread pool. ([#1016](https://github.com/pika-org/pika/pull/1016))
- Unrevert [#872](https://github.com/pika-org/pika/pull/872) with an indexing error fixed. ([#1017](https://github.com/pika-org/pika/pull/1017))

## 0.22.1 (2024-01-29)

### Bugfixes

- Revert [#872](https://github.com/pika-org/pika/pull/872) as it was found to cause issues in applications. ([#1009](https://github.com/pika-org/pika/pull/1009))

## 0.22.0 (2024-01-24)

### New features

- A new function `pika::is_runtime_initialized` has been added. ([#808](https://github.com/pika-org/pika/pull/808))
- CUDA and HIP handles are now guaranteed to be released together with `cuda_pool` instead of on program exit. ([#872](https://github.com/pika-org/pika/pull/872))
- Spinloop performance has been significantly improved on ARM64 systems. ([#923](https://github.com/pika-org/pika/pull/923), [#927](https://github.com/pika-org/pika/pull/927))
- The `pika::barrier` now scales significantly better with the number of cores. ([#940](https://github.com/pika-org/pika/pull/940))
- Exceptions thrown in the main entry point, e.g. `pika_main`, are now reported with the message of the exception, if available. ([#959](https://github.com/pika-org/pika/pull/959))

### Breaking changes

### Bugfixes

- The CMake configuration now sets the policy CMP0144 to silence warnings about CMake package root directory variables. ([#885](https://github.com/pika-org/pika/pull/885))
- The permissions on the installed `pika-bind` helper script have been fixed. ([#915](https://github.com/pika-org/pika/pull/915))
- A missing include causing compilation failures with `PIKA_WITH_UNITY_BUILD=OFF` has been added. ([#955](https://github.com/pika-org/pika/pull/955))
- A use-after-free has been fixed in `when_all_vector`. ([#966](https://github.com/pika-org/pika/pull/966))
- A use-after-free has been fixed in `sync_wait`. ([#976](https://github.com/pika-org/pika/pull/976))
- A use-after-free has been fixed in `default_agent`. ([#979](https://github.com/pika-org/pika/pull/979))
- An initialization order issue has been fixed in debug printing facilities. ([#983](https://github.com/pika-org/pika/pull/983))
- A potential cause for dangling references has been fixed in `thread_pool_init_parameters`. ([#984](https://github.com/pika-org/pika/pull/984))
- A few data races have been fixed in the schedulers. ([#985](https://github.com/pika-org/pika/pull/985), [#986](https://github.com/pika-org/pika/pull/986))
- Forwarding of callable values in `execution::then` has been fixed. ([#994](https://github.com/pika-org/pika/pull/994))
- A data race in `condition_variable::notify_all` has been fixed. ([#998](https://github.com/pika-org/pika/pull/998))

## 0.21.0 (2023-12-06)

### New features

- A new sender adaptor `require_started` allows to detect unstarted senders. ([#869](https://github.com/pika-org/pika/pull/869))
- The conversion from `any_sender` to `unique_any_sender` has been optimized, reusing the same
  storage. ([#844](https://github.com/pika-org/pika/pull/844))
- The number of streams created by the `cuda_pool` is now proportional to the number of threads used by the runtime instead of `hardware_concurrency`. ([#864](https://github.com/pika-org/pika/pull/864))

### Breaking changes

- `pika::start` and `pika::finalize` now return `void`. Most runtime management functions no longer take an `error_code` and always throw an exception on failure. ([#825](https://github.com/pika-org/pika/pull/825))

### Bugfixes

- One lifetime bug in split has been fixed. ([#839](https://github.com/pika-org/pika/pull/839))
- `yield_while` is now able to warn about potential deadlocks when suspension is disabled. ([#856](https://github.com/pika-org/pika/pull/856))

## 0.20.0 (2023-11-01)

### New features

- The MPI rank is now printed with `--pika:print-bind` and when handling exceptions, if MPI support is enabled. ([#805](https://github.com/pika-org/pika/pull/805), [#822](https://github.com/pika-org/pika/pull/822))
- A warning message is now printed on macOS when using `--pika:process-mask` since thread bindings are unsupported. ([#806](https://github.com/pika-org/pika/pull/806))
- Thread bindings can now be printed using the environment variable `PIKA_PRINT_BIND` in addition to the command line option `--pika:print-bind`. ([#828](https://github.com/pika-org/pika/pull/828))
- The `pika-bind` helper script has been added to more conveniently set `PIKA_PROCESS_MASK` based on the environment. ([#834](https://github.com/pika-org/pika/pull/834))

### Breaking changes

- All remaining locality-related functions and files have been removed. ([#823](https://github.com/pika-org/pika/pull/823))

### Bugfixes

- Handling of explicitly specified process masks with `--pika:process-mask` or `PIKA_PROCESS_MASK` has been fixed. ([#807](https://github.com/pika-org/pika/pull/807))

## 0.19.1 (2023-10-09)

### Bugfixes

- Fix a bug in `drop_operation_state` when the predecessor sender is sending a tuple. ([#801](https://github.com/pika-org/pika/pull/801))

## 0.19.0 (2023-10-04)

### New features

- A `transfer_when_all` sender adaptor has been introduced. ([#792](https://github.com/pika-org/pika/pull/792))
- A `drop_operation_state` sender adaptor has been introduced. ([#772](https://github.com/pika-org/pika/pull/772))

### Breaking changes

- The `PIKA_WITH_DISABLED_SIGNAL_EXCEPTION_HANDLERS` CMake option has been removed. This option can be controlled at runtime as before. ([#773](https://github.com/pika-org/pika/pull/773))
- The `PIKA_WITH_THREAD_GUARD_PAGE` CMake option has been removed. This option can be controlled at runtime as before. ([#782](https://github.com/pika-org/pika/pull/782))
- `thread_priority::critical` has been removed as it is an alias to `high_recursive` and is unused.([#783](https://github.com/pika-org/pika/pull/783))

### Bugfixes

- Fix a few instances of the wrong type being forwarded in `split_tuple` and `when_all` sender adaptors. ([#781](https://github.com/pika-org/pika/pull/781))
- Fix a hang introduced by the global activity count when using MPI polling. ([#778](https://github.com/pika-org/pika/pull/778))
- Fix a use-after-free in `ensure_started`. ([#795](https://github.com/pika-org/pika/pull/795))
- Fix lifetime bug in `ensure_started` when the sender is dropped without being connected. ([#797](https://github.com/pika-org/pika/pull/797))

## 0.18.0 (2023-09-06)

### New features

- A documentation site has been created on [pikacpp.org](https://pikacpp.org). ([#723](https://github.com/pika-org/pika/pull/723))
- A new command line option `--pika:process-mask` has been added to allow overriding the mask detected at startup. The process mask is also now read before the pika runtime is started to avoid problems with e.g. OpenMP resetting the mask before pika can read it. ([#738](https://github.com/pika-org/pika/pull/738), [#739](https://github.com/pika-org/pika/pull/739))
- An overload of `pika::start` which takes no callable and is equivalent to passing `nullptr_t` or an empty callable as the entry point has been added. ([#761](https://github.com/pika-org/pika/pull/761))

### Breaking changes

- The `any_receiver` `set_value_t` overload now accepts types which may throw in their move and copy constructors. ([#702](https://github.com/pika-org/pika/pull/702))
- The `PIKA_WITH_GENERIC_CONTEXT_COROUTINES` CMake option has been renamed to `PIKA_WITH_BOOST_CONTEXT`. ([#729](https://github.com/pika-org/pika/pull/729))
- The `then` and `unpack` sender adaptors now correctly have `noexcept` `get_env_t` customizations. ([#732](https://github.com/pika-org/pika/pull/732))
- mimalloc is now the default allocator. ([#730](https://github.com/pika-org/pika/pull/730))
- The `pika::this_thread::experimental::sync_wait` receiver now correctly advertises itself as a receiver when using stdexec. ([#735](https://github.com/pika-org/pika/pull/735))
- Various outdated and unused utilities and configuration options have been removed. ([#744](https://github.com/pika-org/pika/pull/744), [#753](https://github.com/pika-org/pika/pull/753), [#758](https://github.com/pika-org/pika/pull/758), [#759](https://github.com/pika-org/pika/pull/759))

### Bugfixes

- The small buffer optimization in `any_sender` and its companions has been disabled due to issues with the implementation. ([#760](https://github.com/pika-org/pika/pull/760))

## 0.17.0 (2023-08-02)

### New features

- Improve MPI polling: continuations are not triggered under lock anymore and can be explicitly transferred to a new task/pool, throttling is possible on a per stream basis, the number of completions to handle per poll iteration may be controlled. ([#593](https://github.com/pika-org/pika/pull/593))
- Add `pika::wait` to wait for the runtime to be idle. ([#704](https://github.com/pika-org/pika/pull/704))
- Failure information is now printed before attaching a debugger. ([#712](https://github.com/pika-org/pika/pull/712))
- `--pika:print-bind` now also prints the thread pool of a thread when thread binding is disabled to be consistent with the output when thread binding is enabled. ([#710](https://github.com/pika-org/pika/pull/710))
- Allow to explicitly reset `{unique_,}any_sender`.  ([#719](https://github.com/pika-org/pika/pull/719))
- Add `execution::unpack` sender adaptor  unpack tuple-like types sent by predecessor senders. ([#721](https://github.com/pika-org/pika/pull/721))

### Bugfixes

- Fix warnings when CMake unity build is disabled. ([#697](https://github.com/pika-org/pika/pull/697))
- Fix bogus error when `--pika:print-bind` and `--pika:bind=none` are used together. ([#710](https://github.com/pika-org/pika/pull/710))
- Fix memory leak with stack overflow detection enabled. ([#714](https://github.com/pika-org/pika/pull/714))
- Fix freeing stack when guard pages are disabled. ([#716](https://github.com/pika-org/pika/pull/716))

## 0.16.0 (2023-05-31)

### New features

- pika can now be compiled with CUDA 12 and C++20 when `PIKA_WITH_STDEXEC` is disabled. ([#684](https://github.com/pika-org/pika/pull/684))
- `pika::barrier` can now optionally do a timed blocking wait. The default behaviour is unchanged. ([#685](https://github.com/pika-org/pika/pull/685))

### Breaking changes

- `pika::spinlock` has been removed and replaced with a private implementation. ([#672](https://github.com/pika-org/pika/pull/672))

### Bugfixes

- Compilation with the CMake option `PIKA_WITH_VERIFY_LOCKS_BACKTRACE` has been fixed. ([#680](https://github.com/pika-org/pika/pull/680))
- Compilation with fmt 10 and CUDA/HIP has been fixed. ([#691](https://github.com/pika-org/pika/pull/691))

## 0.15.1 (2023-05-12)

### Bugfixes

- Eagerly reset shared state in `async_rw_mutex`. This prevents deadlocks in certain use cases of `async_rw_mutex`. ([#677](https://github.com/pika-org/pika/pull/677))
- Use `pika::spinlock` instead of `pika::mutex` in `async_rw_mutex`. This allows use of `async_rw_mutex` from non-pika threads. ([#679](https://github.com/pika-org/pika/pull/679))

## 0.15.0 (2023-05-03)

### New features

- `async_rw_mutex` has been moved to a public header: `pika/async_rw_mutex.hpp`. The functionality is still experimental in the `pika::execution::experimental` namespace. `async_rw_mutex_access_type` and `async_rw_mutex_access_wrapper` have also been moved out of the `detail` namespace. ([#655](https://github.com/pika-org/pika/pull/655))

### Breaking changes

- The `any_sender` and `unique_any_sender` `operator bool()`, which can be used to check whether the sender contains a valid sender, is now `explicit` to avoid accidental conversions. ([#653](https://github.com/pika-org/pika/pull/653))
- Scheduler idling was disabled by default. This typically improves performance. If performance is less important than resource usage idling may be beneficial to enable explicitly. ([#661](https://github.com/pika-org/pika/pull/661))
- The CMake option `PIKA_WITH_THREAD_CUMULATIVE_COUNTS` was disabled by default. This often improves performance. ([#662](https://github.com/pika-org/pika/pull/662))
- Thread guard pages were disabled by default. This often improves performance. They can still be enabled at runtime with the configuration option  `pika.stacks.use_guard_pages=1` to debug e.g. stack overflows. ([#663](https://github.com/pika-org/pika/pull/663))
- The `fast_idle` and `delay_exit` scheduler modes were completely removed as they added overhead and were not used in any meaningful way in the scheduler. ([#664](https://github.com/pika-org/pika/pull/664))
- The ability to run background threads in the scheduler was completely removed. ([#665](https://github.com/pika-org/pika/pull/665), [#668](https://github.com/pika-org/pika/pull/668))

### Bugfixes

- Fixed an inconsistent preprocessor guard that affected Apple M1 and M2 systems. ([#657](https://github.com/pika-org/pika/pull/657))
- Fixed preprocessor guards to enable deadlock detection in debug builds. The deadlock detection was never enabled previously. ([#658](https://github.com/pika-org/pika/pull/658))
- Thread deadlock detection will now correctly print potentially deadlocked threads. ([#659](https://github.com/pika-org/pika/pull/659))

## 0.14.0 (2023-04-05)

### New features

- pika can now be compiled with NVHPC. The support is experimental. ([#606](https://github.com/pika-org/pika/pull/606))
- CUDA polling was improved. Among other changes polling continuations are no longer called under a lock. ([#609](https://github.com/pika-org/pika/pull/609))
- Improved the error message when pika is configured with multiple thread pools but there are not enough resources for all thread pools. ([#619](https://github.com/pika-org/pika/pull/619))

### Breaking changes

- Cleaned up modules and moved internal functionality into `detail` namespaces. ([#625](https://github.com/pika-org/pika/pull/625), [#631](https://github.com/pika-org/pika/pull/631), [#632](https://github.com/pika-org/pika/pull/632), [#633](https://github.com/pika-org/pika/pull/633), [#634](https://github.com/pika-org/pika/pull/634))
- Renamed the CMake option `PIKA_WITH_P2300_REFERENCE_IMPLEMENTATION` to `PIKA_WITH_STDEXEC` to better reflect what it does. ([#641](https://github.com/pika-org/pika/pull/641))

### Bugfixes

## 0.13.0 (2023-03-08)

### New features

- Add better compile-time error messages to diagnose one-shot senders being used as multi-shot senders. ([#586](https://github.com/pika-org/pika/pull/586))

### Breaking changes

- Remove the `PIKA_WITH_BACKGROUND_THREAD_COUNTERS` CMake option. These counters are no longer available. ([#588](https://github.com/pika-org/pika/pull/588))
- Update required stdexec commit. pika is now tested with `6510f5bd69cc03b24668f26eda3dd3cca7e81bb2` ([#597](https://github.com/pika-org/pika/pull/597))
- Cleaned up modules and moved minor functionality into `detail` namespaces. ([#594](https://github.com/pika-org/pika/pull/594), [#595](https://github.com/pika-org/pika/pull/595), [#596](https://github.com/pika-org/pika/pull/596), [#599](https://github.com/pika-org/pika/pull/599), [#607](https://github.com/pika-org/pika/pull/607))

### Bugfixes

- Initialize HIP early to avoid concurrent initialization. ([#591](https://github.com/pika-org/pika/pull/591))


## 0.12.0 (2023-02-01)

### New features

- Make read-only access senders of `async_rw_mutex` connectable with l-value references. ([#548](https://github.com/pika-org/pika/pull/548))
- Add `split_tuple` sender adaptor which allows transforming a sender of tuples into a tuple of senders. ([#549](https://github.com/pika-org/pika/pull/549))
- Add `bool` conversion operator and `empty` member function to `unique_any_sender` and `any_sender`. ([#551](https://github.com/pika-org/pika/pull/551))

### Breaking changes

- Remove the conversion operators from wrapper types in `async_rw_mutex`. Wrappers must explicitly be unwrapped using `get`. ([#548](https://github.com/pika-org/pika/pull/548))
- Require whip 0.1.0. ([#565](https://github.com/pika-org/pika/pull/565))

### Bugfixes

- Make the `ensure_started` sender noncopyable. ([#539](https://github.com/pika-org/pika/pull/539))
- Fix compilation failure on macOS with C++20 enabled. ([#541](https://github.com/pika-org/pika/pull/541))
- Fix deadlocks in certain use cases of `async_rw_mutex`. ([#548](https://github.com/pika-org/pika/pull/548))
- Fix certain use cases of `any_sender` and `when_all`. ([#555](https://github.com/pika-org/pika/pull/555))

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
