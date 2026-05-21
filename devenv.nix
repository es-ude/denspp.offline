{
  pkgs,
  lib,
  config,
  inputs,
  ...
}: let
  unstablePkgs = import inputs.nixpkgs-unstable {system = pkgs.stdenv.system;};
in {

  packages = [
    pkgs.git
    pkgs.tombi
    pkgs.zlib # needed as dependency cocotb/ghdl under circumstances
    pkgs.iverilog
  ];

  languages.c.enable = false;
  languages.nix.enable = true;
  cachix.pull = ["nixpkgs-python"];
  languages.python = {
    enable = true;
    package = pkgs.python313;
    uv.enable = true;
    uv.sync.enable = true;
    uv.sync.allExtras = true;
    uv.package = unstablePkgs.uv;
  };

  processes = {
    serve_docs.exec = "serve_docs";
  };

  scripts = {
    serve_docs = {
      exec = "${unstablePkgs.uv}/bin/uv run sphinx-autobuild -j auto docs build/docs/";
    };
  };

  tasks = let
    uv_run = "${unstablePkgs.uv}/bin/uv run";
  in {
    "package:build" = {
      exec = "${unstablePkgs.uv}/bin/uv build";
    };

    "docs:single-page" = {
      exec = ''
        export LC_ALL=C  # necessary to run in github action
        ${uv_run} sphinx-build -b singlehtml docs build/docs
      '';
    };

    "docs:build" = {
      exec = ''
        export LC_ALL=C  # necessary to run in github action
        ${uv_run} sphinx-build -j auto -b html docs build/docs
        touch build/docs/.nojekyll  # prevent github from trying to build the docs
      '';
    };

    "docs:clean" = {
      exec = ''
        rm -rf build/docs/*
      '';
    };

    "check:tests" = {
      after = [
        "check:slow-tests"
        "check:fast-tests"
      ];
    };

    "check:slow-tests" = {
      exec = "${uv_run} --all-packages python -m pytest  -m '(simulation or slow) and not hardware'";
      before = ["check:tests"];
    };

    "check:fast-tests" = {
      exec = ''
        ${uv_run} --all-packages coverage run
        ${uv_run} coverage xml
      '';
      before = ["check:tests"];
    };

    "check:python-lint" = {
      exec = "${uv_run} ruff check";
      before = ["check:code-lint"];
    };

    "check:formatting" = {
      exec = "${uv_run} ruff format --check";
      before = ["check:code-lint"];
    };

    "check:toml-formatting" = {
      exec = "${pkgs.tombi}/bin/tombi format --check";
    };

    "check:code-lint" = {
      after = [
        "check:nix-lint"
        "check:python-lint"
        "check:toml-formatting"
      ];
    };
  };
}
