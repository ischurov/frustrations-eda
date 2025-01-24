{ lib
, poetry2nix
, python3
, fetchPypi
}:

let
  pname = "neptune";
  version = "1.10.4";
in
poetry2nix.mkPoetryApplication {
  inherit pname version;
  src = fetchPypi {
    inherit pname version;
    sha256 = "sha256-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"; # Replace with actual hash
  };

  # Specify the Python version you want to use
  python = python3;

  # Optionally override dependencies
  overrides = poetry2nix.defaultPoetryOverrides.extend (self: super: {
    # Add any necessary overrides here
  });

  # Disable tests if needed
  doCheck = false;

  meta = with lib; {
    description = "Neptune.ai Python client library";
    homepage = "https://neptune.ai/";
    license = licenses.asl20;
    maintainers = with maintainers; [ ]; # Add maintainers if applicable
  };
}