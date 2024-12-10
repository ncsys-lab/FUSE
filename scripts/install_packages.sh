#!/bin/bash
#
# NOTE: Meant to be used by Docker build script!
#
set -euo pipefail

apt-get update

curl --proto '=https' --tlsv1.2 -sSf -L https://install.determinate.systems/nix/pr/1145 | sh -s -- install linux --init none --no-confirm --extra-conf "
    extra-substituters = https://openlane.cachix.org
    extra-trusted-public-keys = openlane.cachix.org-1:qqdwh+QMNGmZAuyeQJTH9ErW57OWSvdtuwfBKdS254E=
"

mkdir plots

. /nix/var/nix/profiles/default/etc/profile.d/nix-daemon.sh

git clone https://github.com/efabless/openlane2
cd openlane2 ; nix-shell --command "echo Done building OpenLane!" ; cd -
