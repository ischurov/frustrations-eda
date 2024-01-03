{ buildPythonPackage
, fetchPypi
}:
buildPythonPackage rec {
  pname = "combinadics";
  version = "0.0.3";
  src = fetchPypi {
    inherit pname version;
    sha256 = "sha256-CFbtDgcbrFKEYknegVRSUZbc+jS0OCGN53ZYBAUAFD4=";
  };
  doCheck = false;
}
