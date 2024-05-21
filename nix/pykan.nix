{ buildPythonPackage, fetchPypi }:
buildPythonPackage rec {
  pname = "pykan";
  version = "0.0.2";
  src = fetchPypi {
    inherit pname version;
    sha256 = "sha256-wOcP2mq7gLwxuu70oyNRab/7g0jHilfcK51qMJMnb2c=";
  };
  doCheck = false;
}
