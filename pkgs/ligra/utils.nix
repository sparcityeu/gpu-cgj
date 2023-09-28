{
  stdenv,
  fetchFromGitHub,
  ...
}:
stdenv.mkDerivation {
  pname = "Ligra";
  version = "2021-06-04";
  src = fetchFromGitHub {
    owner = "jshun";
    repo = "ligra";
    rev = "7755d95fbac4a587ee7c5920d1b927c545f97d07";
    sha256 = "sha256-gR3tehBIcu/69Zmi0Dv78TXMXc8KG2mQfjAqPKhk/bI=";
  };
  sourceRoot = "source/utils";
  # Cilk Plus was dropped from GCC
  preBuild = ''
    export OPENMP=1
  '';
  installPhase = ''
    mkdir -p $out/bin
    cp rMatGraph gridGraph randLocalGraph SNAPtoAdj wghSNAPtoAdj $out/bin
    cp adjGraphAddWeights adjToBinary communityToHyperAdj $out/bin
    cp hyperAdjToBinary adjHypergraphAddWeights randHypergraph $out/bin
    cp KONECTtoHyperAdj KONECTtoClique communityToClique $out/bin
    cp communityToMESH KONECTtoMESH $out/bin
  '';
}
