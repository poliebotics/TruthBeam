#!/usr/bin/env bash
# Turn-key publish of the TruthBeam repo. Run this AFTER you have:
#   (1) confirmed your publication gates;
#   (2) GitHub auth available to YOU (use your own account / a PAT);
#   (3) noted that the target slug (poliebotics/truthbeam) holds the older proof-of-concept —
#       the DECIDED plan (2026-06-10) is to overwrite it deliberately; the prototype is
#       summarised as lineage in the PolieBotics landing repo's truth_beam.md.
#
# This repo is published ALL RIGHTS RESERVED (no open-source license, no patent license) and with
# the 64s video intentionally excluded (published separately). History is clean: no video, no keys,
# no private host/paths.
set -euo pipefail
cd "$(dirname "$0")"

ORG_REPO="${1:-poliebotics/truthbeam}"   # override: ./PUSH.sh youruser/yourrepo

echo "About to publish $(git rev-list --count HEAD) commits / $(git ls-files | wc -l) files to github.com/$ORG_REPO"
echo "LICENSE = all-rights-reserved (no patent license). Confirm your publication gates before proceeding."
read -r -p "Proceed? [y/N] " ok; [ "$ok" = y ] || { echo "aborted"; exit 1; }

if command -v gh >/dev/null 2>&1; then
  gh auth status >/dev/null 2>&1 || gh auth login
  if gh repo view "$ORG_REPO" >/dev/null 2>&1; then
    echo "Repo $ORG_REPO already exists — this will OVERWRITE its history (force-push)."
    read -r -p "Force-push over the existing repo? [y/N] " ow; [ "$ow" = y ] || { echo "aborted"; exit 1; }
    git remote get-url origin >/dev/null 2>&1 || git remote add origin "https://github.com/$ORG_REPO.git"
    git branch -M main && git push --force -u origin main
    gh repo edit "$ORG_REPO" --description "TruthBeam: blockchain-anchored projector-camera recording with a diffusion-residual verifier (all rights reserved)"
  else
    gh repo create "$ORG_REPO" --public --source=. --remote=origin --push \
       --description "TruthBeam: blockchain-anchored projector-camera recording with a diffusion-residual verifier (all rights reserved)"
  fi
else
  echo "gh not installed. Create an EMPTY repo named '$ORG_REPO' on github.com, then:"
  echo "  git remote add origin git@github.com:$ORG_REPO.git   # (or https://… with a PAT)"
  echo "  git branch -M main && git push -u origin main"
  echo "Run those once the remote exists."
fi
echo "After push: tag a release (e.g. v1.0-whitepaper) so Zenodo mints a DOI (see docs/DATA_MODEL_PUBLISHING_PLAN.md)."
