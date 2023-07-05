{
  stdenv,
  fetchFromGitHub,
  ...
}:
stdenv.mkDerivation rec {
  pname = "Ligra";
  version = "2021-06-04";
  src = fetchFromGitHub {
    owner = "jshun";
    repo = "ligra";
    rev = "7755d95fbac4a587ee7c5920d1b927c545f97d07";
    sha256 = "sha256-gR3tehBIcu/69Zmi0Dv78TXMXc8KG2mQfjAqPKhk/bI=";
  };
  sourceRoot = "source/apps";
  # Cilk Plus was dropped from GCC
  preBuild = ''
    export OPENMP=1
  '';
  installPhase = ''
    mkdir -p $out/bin
    cp BFS                 $out/bin/ligra-bfs
    cp BC                  $out/bin/ligra-bc
    cp BellmanFord         $out/bin/ligra-bellmanford
    cp Components          $out/bin/ligra-components
    cp Components-Shortcut $out/bin/ligra-components-shortcut
    cp Radii               $out/bin/ligra-radii
    cp PageRank            $out/bin/ligra-pagerank
    cp PageRankDelta       $out/bin/ligra-pagerankdelta
    cp BFSCC               $out/bin/ligra-bfscc
    cp BFS-Bitvector       $out/bin/ligra-bfs-bitvector
    cp KCore               $out/bin/ligra-kcore
    cp MIS                 $out/bin/ligra-mis
    cp Triangle            $out/bin/ligra-triangle
    cp CF                  $out/bin/ligra-cf
  '';
}
