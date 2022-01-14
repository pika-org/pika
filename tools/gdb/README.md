<!-- Copyright (c) 2014 Thomas Heller                                             -->
<!--                                                                              -->
<!-- SPDX-License-Identifier: BSL-1.0                                             -->
<!-- Distributed under the Boost Software License, Version 1.0. (See accompanying -->
<!-- file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)        -->

This directory contains python script to help debugging pika applications with
gdb. In order to use it, please add the following lines to your ~/.gdbinit:

    source $PIKA_SRC_DIR/tools/gdb/pika.py

    define hook-continue
    pika thread restore
    end

For a list of commands see `help pika` inside of gdb.

Notes:

 - The scripts currently only work when pika is compiled against Boost 1.56
 - The scripts currently only work with the local priority scheduler
 - The scripts currently only work on 64 bit executables
 - The scripts currently only work with the `x86_linux_context` context
   implementation (which is the default on Linux X86_64)
 - The hook-continue is needed because we currently just overwrite the current
   frame with the selected pika user level context.

By not doing a `pika thread restore` before gdb continues execution, your
program will abort.
