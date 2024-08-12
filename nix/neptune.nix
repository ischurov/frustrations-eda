{ buildPythonPackage
, fetchPypi
}:
buildPythonPackage rec {
  pname = "neptune";
  version = "1.10.4";
  src = fetchPypi {
    inherit pname version;
 #   sha256 = "sha256-CFbtDgcbrFKEYknegVRSUZbc+jS0OCGN53ZYBAUAFD4=";
  };
  doCheck = false;
}
