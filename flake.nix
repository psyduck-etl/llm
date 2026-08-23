{
  description = "psyduck-etl/llm: LLM-backed filter and transform plugin";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        packages.default = pkgs.buildGoModule {
          pname = "llm";
          version = self.shortRev or self.dirtyShortRev or "dev";

          src = self;
          vendorHash = "sha256-bo7qyfXuoCcrTyA1Jdi3dknd/c0spz3GSAj/Gx6IRYM=";
          env.CGO_ENABLED = 0;

          ldflags = [
            "-s"
            "-w"
          ];

          meta = {
            description = "psyduck ETL plugin: LLM-backed filter and transform (ollama/anthropic providers)";
            homepage = "https://github.com/psyduck-etl/llm";
            mainProgram = "llm";
          };
        };

        devShells.default = pkgs.mkShell {
          packages = [ pkgs.go ];
        };
      }
    );
}
