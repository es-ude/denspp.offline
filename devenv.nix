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
  ];

  languages.c.enable = false;
  languages.nix.enable = true;
  languages.python = {
    enable = true;
    package = pkgs.python313;
    uv.enable = true;
    uv.package = unstablePkgs.uv;
    uv.sync.enable = true;
    uv.sync.allExtras = true;
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
    "check:fast-tests" = {
      exec = ''
        ${uv_run} coverage run
        ${uv_run} coverage xml
      '';
      before = ["check:tests"];
    };

    "check:types" = {
      exec = "${uv_run} mypy -p denspp.offline";
      before = ["check:code-lint"];
    };

    "check:python-lint" = {
      exec = "${uv_run} ruff check";
      before = ["check:code-lint"];
    };

    "check:formatting" = {
      exec = "${uv_run} ruff format --check";
      before = ["check:code-lint"];
    };

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

    "check:code-lint" = {
    };

    "check:tests" = {
    };
  };
}
## Commented out while we're configuring pre-commit manually
# pre-commit.hooks = {
#   shellcheck.enable = true;
#   ripsecrets.enable = true; # don't commit secrets
#   ruff.enable = true; # lint and automatically fix simple problems/reformat
#   taplo.enable = true; # reformat toml
#   nixfmt-rfc-style.enable = true; # reformat nix
#   ruff-format.enable = true;
#   mypy = {
#     enable = false;
#   }; # check type annotations
#   end-of-file-fixer.enable = true;
#   commitizen.enable = true; # help adhering to commit style guidelines
#   check-toml.enable = true; # check toml syntax
#   check-case-conflicts.enable = true;
#   check-added-large-files.enable = true;
# };
