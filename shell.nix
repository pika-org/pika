# Copyright (c) 2022 ETH Zurich
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

{ pkgs ? import <nixpkgs> { } }:
pkgs.mkShell.override { stdenv = pkgs.gcc13Stdenv; } {
  buildInputs = with pkgs; [
    boost
    ccache
    cmake-format
    cmakeCurses
    fmt
    gperftools
    hwloc
    mpich
    ninja
    pkg-config
    spdlog
    doxygen
    python311Packages.breathe
    python311Packages.sphinx
    python311Packages.sphinx-material
    python311Packages.recommonmark
  ];

  hardeningDisable = [ "fortify" ];

  CMAKE_GENERATOR = "Ninja";
  CMAKE_CXX_COMPILER_LAUNCHER = "ccache";
  CXXFLAGS = "-ftemplate-backtrace-limit=0 -fdiagnostics-color=always";
}
