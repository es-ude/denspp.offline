{
  pkgs,
  lib,
  config,
  inputs,
  ...
}:
let
  pkgs-unstable = import inputs.nixpkgs-unstable { system = pkgs.stdenv.system; };
in
{
  packages = [
    pkgs.git
    pkgs.tombi
    pkgs.zlib # needed as dependency cocotb/ghdl under circumstances
    pkgs.iverilog
    pkgs.cocogitto
  ];

  languages = {
    c = {
      enable = false;
    };
    nix = {
      enable = true;
    };
    python = {
      enable = true;
      version = "3.13";
      venv.enable = true;
      uv = {
        enable = true;
        package = pkgs-unstable.uv;
        sync.enable = true;
        sync.allGroups = true;
      };
    };
  };

  processes = {
    serve_docs.exec = "serve_docs";
  };

  scripts = {
    serve_docs = {
      exec = "${pkgs-unstable.uv}/bin/uv run sphinx-autobuild -j auto docs build/docs/";
    };
  };

  tasks =
    let
      uv_run = "${pkgs-unstable.uv}/bin/uv run";
      uv_build = "${pkgs-unstable.uv}/bin/uv build";
      cog_check = "${pkgs.cocogitto}/bin/cog check";
    in
    {
      "package:build" = {
        exec = "${uv_build}";
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

      "check:fast-tests" = {
        exec = ''
          ${uv_run} --all-packages coverage run -m pytest -m "not (simulation or slow or hardware)"
        '';
        before = [ "check:tests" ];
      };
      "check:slow-tests" = {
        exec = ''
          ${uv_run} --all-packages python -m pytest  -m '(simulation or slow) and not hardware'
        '';
        before = [ "check:tests" ];
      };
      "check:hardware-tests" = {
        exec = ''
          ${uv_run} --all-packages python -m pytest  -m 'hardware and not (simulation or slow)'
        '';
        before = [ "check:tests" ];
      };
      "check:coverage-report" = {
        exec = ''
          ${uv_run} coverage report -m
          ${uv_run} coverage xml
        '';
      };
      "check:tests" = {
        after = [
          "check:fast-tests"
          "check:slow-tests"
          "check:hardware-tests"
        ];
      };

      "check:conventional-commit" = {
        exec = ''
          if [ -n "$CI" ]; then
            ${cog_check} ..$GITHUB_SOURCE_REF
          else
            ${cog_check} main..
          fi
        '';
      };

      "check:toml-lint" = {
        exec = ''
          ${pkgs.tombi}/bin/tombi format --check
        '';
        before = [ "check:code-lint" ];
      };
      "check:python-lint" = {
        exec = ''
          ${uv_run} ruff format --check
        '';
        before = [ "check:code-lint" ];
      };
      "check:python-types" = {
        exec = ''
          ${uv_run} ty check denspp/
        '';
        before = [ "check:code-lint" ];
      };
      "check:code-lint" = {
        after = [
          "check:python-lint"
          "check:python-types"
          "check:toml-lint"
        ];
      };
    };
} # See full reference at https://devenv.sh/reference/options/
