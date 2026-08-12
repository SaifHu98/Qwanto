{
  description = "qwanto — run GLM-5.2 (744B MoE) on a consumer machine with ~25 GB RAM";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };

        # Python with the packages needed by the offline converter tools
        pythonEnv = pkgs.python3.withPackages (ps: with ps; [
          torch
          safetensors
          huggingface-hub
          numpy
          tokenizers
          datasets
        ]);

        qwanto = pkgs.stdenv.mkDerivation {
          pname = "qwanto";
          version = "1.0";
          src = ./.;

          nativeBuildInputs = [ pkgs.makeWrapper ];

          buildInputs = [
            pkgs.gcc
            pkgs.gmp
          ];

          # Use x86-64-v3 (AVX2) for a portable binary; override with ARCH=native for local builds
          ARCH = "x86-64-v3";

          buildPhase = ''
            runHook preBuild
            make -C c glm ARCH="$ARCH"
            runHook postBuild
          '';

          installPhase = ''
            runHook preInstall
            mkdir -p $out/bin
            cp c/glm $out/bin/glm

            # Wrap coli (the Python CLI) so it finds the right python and the engine
            mkdir -p $out/share/qwanto
            cp c/coli $out/share/qwanto/qwanto
            chmod +x $out/share/qwanto/qwanto
            cp -r c/tools $out/share/qwanto/tools

            makeWrapper ${pythonEnv}/bin/python $out/bin/qwanto \
              --add-flags "$out/share/qwanto/qwanto" \
              --set PYTHONPATH "${pythonEnv}/${pkgs.python3.sitePackages}"
            runHook postInstall
          '';

          checkPhase = ''
            runHook preCheck
            cd c
            make test-c
            cd ..
            runHook postCheck
          '';

          doCheck = true;

          meta = with pkgs.lib; {
            description = "Run GLM-5.2 (744B MoE) on a consumer machine with ~25 GB RAM";
            homepage = "https://github.com/JustVugg/qwanto";
            license = licenses.asl20;
            platforms = platforms.linux;
            mainProgram = "glm";
          };
        };
      in
      rec {
        packages = {
          default = qwanto;
          inherit qwanto;
        };

        apps = {
          default = {
            type = "app";
            program = "${qwanto}/bin/glm";
          };
          qwanto = {
            type = "app";
            program = "${qwanto}/bin/qwanto";
          };
        };

        devShells.default = pkgs.mkShell {
          inputsFrom = [ qwanto ];

          packages = [
            pythonEnv
            pkgs.gcc
            pkgs.gnumake
            pkgs.clang-tools          # clangd / clang-tidy for IDE support
            pkgs.pkg-config
          ];

          shellHook = ''
            echo "🐦 qwanto dev shell"
            echo "  gcc: $(gcc --version | head -1)"
            echo "  python: $(python3 --version)"
            echo ""
            echo "Build the engine:   make -C c glm"
            echo "Run the converter:  python c/qwanto convert --model /path/to/glm52_i4"
            echo "Chat:               QWANTO_MODEL=/path/to/glm52_i4 ./c/glm ..."
          '';
        };
      }
    );
}
