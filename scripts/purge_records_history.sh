#!/usr/bin/env bash
# Purge the 'Records' file (name + phone pairs) from ALL git history and force-push.
# Irreversible for collaborators: everyone must re-clone afterwards.
# Prereq: pip install git-filter-repo   (https://github.com/newren/git-filter-repo)
set -euo pipefail
cd "$(dirname "$0")/.."
git filter-repo --invert-paths --path Records --force
git remote add origin git@github.com:ParmpalSinghGill/Hospital-Management.git 2>/dev/null || true
git push origin --force --all
git push origin --force --tags
echo "Done. Now ask GitHub Support to purge cached views of the old commits, and record the incident (MD/hipaa/02 §8)."
