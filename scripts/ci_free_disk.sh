#!/usr/bin/env bash
# Copyright (c) anneal developers
# Free disk on a GitHub ubuntu-latest runner before an opt-level 2 test build.
# The image keeps Android, .NET, Conda, browsers, and the toolcache on a disk
# that only guarantees about 14 GiB free. Those trees are unused by the test
# jobs. Python stays; setup-python may just have installed it.
set -euo pipefail

df -h /
sudo rm -rf \
  /usr/local/lib/android \
  /usr/share/dotnet \
  /usr/share/miniconda \
  /usr/local/share/vcpkg \
  /usr/local/.ghcup \
  /opt/ghc \
  /usr/share/swift \
  /usr/local/share/powershell \
  /usr/local/share/chromium \
  /usr/local/share/chromedriver-linux64 \
  /usr/local/share/edge_driver \
  /usr/local/share/gecko_driver \
  /opt/google \
  /opt/microsoft \
  /opt/az \
  /usr/lib/jvm \
  /home/linuxbrew
if [[ -d /opt/hostedtoolcache ]]; then
  find /opt/hostedtoolcache -mindepth 1 -maxdepth 1 ! -name Python \
    -exec sudo rm -rf {} +
fi
if command -v docker >/dev/null 2>&1; then
  sudo docker image prune --all --force || true
fi
sudo apt-get clean
df -h /
