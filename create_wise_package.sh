#!/usr/bin/env bash
set -euo pipefail

# Usage: ./package.sh <version>
# Example: ./package.sh 1.0.1   -> version becomes 1.0.1.202610280835

if [ "$#" -ne 1 ]; then
  echo "Usage:$0 <version>" >&2
  echo "Example: $0 1.0.1" >&2
  exit 1
fi
VERSION_ARG="$1"

ORIG_DIR="$PWD"                                   # where we started (we cd later)
FOLDER_NAME="$(basename "$PWD")"                  # e.g. "GrIS"
DATE_HOUR="$(date +%Y%m%d%H%M)"                   # 2026-10-28 08:35 -> 202610280835

# Git identifiers: short commit hash + optional trailing "issue_<n>" branch segment
COMMIT_SHORT="$(git rev-parse --short HEAD)"      # e.g. "a1b2c3d"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"       # e.g. "master" or "feature/x/issue_123"

# Branches ending in "issue_<n>" get the token in the file name and the bare
# number appended to the version; any other branch (e.g. master/main) omits it.
VERSION="${VERSION_ARG}.${DATE_HOUR}"             # "1.0.1.202610280835"
BRANCH_LABEL=""
if [[ "$BRANCH" =~ issue_[0-9]+$ ]]; then
  BRANCH_LAST="${BASH_REMATCH[0]}"                # e.g. "issue_123"
  ISSUE_NUM="${BRANCH_LAST#issue_}"               # e.g. "123"
  VERSION="${VERSION}.${ISSUE_NUM}"               # "1.0.1.202610280835.123"
  BRANCH_LABEL="_${BRANCH_LAST}"                  # "_issue_123"
fi

ZIP_NAME="${FOLDER_NAME}_${VERSION}${BRANCH_LABEL}_${COMMIT_SHORT}.zip"
#   master: "GrIS_1.0.1.202610280835_d34f766.zip"
#   issue:  "GrIS_1.0.1.202610280835.123_issue_123_a1b2c3d.zip"

TMP_DIR="$(mktemp -d)"                            # unique temp dir, e.g. /tmp/tmp.XXXX
trap 'rm -rf "$TMP_DIR"' EXIT                     # safety net: clean up even on failure

# 1) copy current folder contents (incl. dotfiles) into the temp dir
cp -r "$PWD/." "$TMP_DIR/"

# 2) set the version inside the temp copy of package.json
cd "$TMP_DIR"
if [ ! -f package.json ]; then
  echo "ERROR: package.json not found in $TMP_DIR" >&2
  exit 1
fi
sed -i -E "s/(\"version\"[[:space:]]*:[[:space:]]*\")[^\"]*(\")/\1${VERSION}\2/" package.json

# 3) build the archive from the temp copy (the command you specified)
zip -r "${ZIP_NAME}" . -x@zip_exclude.lst

# 4) copy the generated zip back to the original folder
cp "${ZIP_NAME}" "$ORIG_DIR/"

# 5) delete the temp copy
rm -rf "$TMP_DIR"

echo "Done: $ORIG_DIR/$ZIP_NAME"