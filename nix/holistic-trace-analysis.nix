{ buildPythonPackage
, fetchPypi
}:

buildPythonPackage rec {
  pname = "HolisticTraceAnalysis";
  version = "0.2.0";
  src = fetchPypi {
    inherit pname version;
    sha256 = "sha256-++/54wua9I1ULgDn/Hwe2Eb943Y3j02zyAm3RT+EtXA=";
  };
  doCheck = false;
}
