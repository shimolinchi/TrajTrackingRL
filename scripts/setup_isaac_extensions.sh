#!/usr/bin/env bash
# User configuration: optional importer dependency for Isaac Sim 5.1 / Python 3.11.
cache_directory="tmp/isaac_extensions"
# End user configuration
set -euo pipefail
[[ $# == 0 ]] || exit 2
root="$(cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$root"
mkdir -p tmp "$cache_directory"
archive="tmp/omni.kit.pip_archive.zip"
url="https://d4i3qtqj3r0z5.cloudfront.net/omni.kit.pip_archive-d38fa9ecd1fb6df4.zip"
expected="3a0d0623be2349a2861f566dc8d5bdb5040c11b0977bb255c6d522b4be7bd682"
if [[ ! -f "$archive" ]] || [[ "$(sha256sum "$archive" | cut -d ' ' -f 1)" != "$expected" ]]; then
  curl --fail --location --retry 2 --continue-at - "$url" -o "$archive"
fi
printf '%s  %s\n' "$expected" "$archive" | sha256sum --check -
unzip -oq "$archive" -d "$cache_directory/omni.kit.pip_archive"
echo "Set TRAJTRACKING_ISAAC_EXTENSION_ROOT to $root/$cache_directory in configs/local.env."
