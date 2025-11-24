{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

    git-hooks = {
      url = "github:cachix/git-hooks.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = {
    self,
    nixpkgs,
    git-hooks,
  }: let
    applySystems = nixpkgs.lib.genAttrs ["x86_64-linux"];
    forAllSystems = f:
      applySystems (
        system:
          f (import nixpkgs {
            inherit system;
            config = {
              allowUnfree = true;
            };
          })
      );
  in {
    checks = forAllSystems (
      pkgs: {
        pre-commit-check = git-hooks.lib.${pkgs.stdenv.hostPlatform.system}.run {
          src = ./.;
          hooks =
            {
              commit-name = {
                enable = true;
                name = "commit name";
                stages = ["commit-msg"];
                entry = ''
                  ${pkgs.python310.interpreter} ${./scripts/apply-commit-convention.py}
                '';
              };
            }
            // pkgs.lib.genAttrs [
              "black"
              "isort"
              "trim-trailing-whitespace"
              "deadnix"
              "alejandra"
            ] (_: {enable = true;});
        };
      }
    );
    devShells = forAllSystems (pkgs: let
      py-env = pkgs.python3.withPackages (
        _:
          with self.packages.${pkgs.stdenv.hostPlatform.system}.default;
            optional-dependencies.dev
      );
    in {
      default = pkgs.mkShell {
        inherit (self.checks.${pkgs.stdenv.hostPlatform.system}.pre-commit-check) shellHook;
        packages = [
          py-env
        ];
      };
    });
    packages = forAllSystems (pkgs:
      with pkgs; {
        default = python3Packages.buildPythonApplication {
          name = "my_torch_analyzer";
          version = "0.0.1";
          pyproject = true;

          src = ./.;

          build-system = [python3Packages.hatchling];

          optional-dependencies = with python3Packages; {
            dev = [
              black
              isort
            ];
          };

          meta = {
            description = "A clash of kings";
            license = lib.licenses.bsd3;
            maintainers = with lib.maintainers; [cizniarova];
            mainProgram = "my_torch_analyzer";
          };
        };
      });
  };
}
