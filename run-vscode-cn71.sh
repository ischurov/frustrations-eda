#!/bin/bash

set -x

# export PATH="/home/ischurov/.local/bin:$PATH"
cd /home/ischurov/tcm10/frustrations-eda

conda deactivate
nix develop --command code tunnel --accept-server-license-terms
