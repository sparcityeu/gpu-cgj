{
  stdenv,
  fetchFromGitHub,
  python3,
  ...
}:
stdenv.mkDerivation rec {
  pname = "Krill";
  version = "2021-10-27";
  src = fetchFromGitHub {
    owner = "chhzh123";
    repo = "Krill";
    rev = "ba17e4ff4cf66c571b55fee7f3a768ab1e67231c";
    sha256 = "sha256-I/igA6v2TH2/3kskmIbR8eSca0YQes+nhhgWYeegyMo=";
  };
  sourceRoot = "source/apps";
  buildInputs = [ python3 ];
  # Cilk Plus was dropped from GCC
  preBuild = ''
    export OPENMP=1
  '';
  installPhase = ''
    mkdir -p $out/bin
    cp Homo1     $out/bin/krill-homo1
    cp Homo2     $out/bin/krill-homo2
    cp Heter     $out/bin/krill-heter
    cp M-BFS     $out/bin/krill-m-bfs
    cp M-SSSP    $out/bin/krill-m-sssp
    cp Multi-BFS $out/bin/krill-multi-bfs
  '';
}
