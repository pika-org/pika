{ pkgs ? import <nixpkgs> { } }:
pkgs.mkShell.override { stdenv = pkgs.gcc11Stdenv; } {
  buildInputs = with pkgs; [
    boost177
    ccache
    cmake-format
    cmakeCurses
    fmt_9
    gperftools
    hwloc
    mpich
    ninja
    pkgconfig
  ];

  hardeningDisable = [ "fortify" ];

  CMAKE_GENERATOR = "Ninja";
  CMAKE_CXX_COMPILER_LAUNCHER = "ccache";
  CXXFLAGS = "-ftemplate-backtrace-limit=0 -fdiagnostics-color=always";
}
