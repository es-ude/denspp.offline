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
    pkgs.cocogitto
    pkgs.alejandra # nix formatter
    pkgs.iverilog
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
        exec = ''
             coverage run -m unittest discover -s denspp -p "*_test.py"
             coverage report -m
        '';
    };

    "check:types" = {
        exec = ''
            ${uv_run} mypy -p elasticai.creator
            ${uv_run} ty check --config-file ty_config_for_ci.toml
        '';
        before = ["check:code-lint"];
    };

    "check:python-lint" = {
        exec = "${uv_run} ruff check";
        before = ["check:code-lint"];
    };

    "check:commit-lint" = {
        exec = ''
            if [ -n "$CI" ]; then
                ${pkgs.cocogitto}/bin/cog check ..$GITHUB_SOURCE_REF
            else
                ${pkgs.cocogitto}/bin/cog check main..
            fi
        '';
    };

    "check:nix-lint" = {
        exec = "${pkgs.alejandra}/bin/alejandra --exclude ./.devenv --exclude ./.devenv.flake.nix -c .";
        before = ["check:code-lint"];
    };

    "check:toml-formatting" = {
        exec = "${pkgs.tombi}/bin/tombi format --check";
    };

    "check:formatting" = {
        exec = "${uv_run} ruff format --check";
        before = ["check:code-lint"];
    };

    "check:code-lint" = {
        after = [
            "check:nix-lint"
            "check:python-lint"
            "check:types"
            "check:toml-formatting"
        ];
    };
  };
}
