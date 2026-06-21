#!/usr/bin/env bash
#
# verify_quick.sh — the seconds-fast complement to verify_all.sh
#
# This is a lightweight, NO-PIP smoke test. Where verify_all.sh does the full
# cryptographic re-derivation (drand beacon, BLAKE3-XOF tiles, AUROC, BLS, etc.)
# and needs `pip install numpy scikit-learn py_ecc blake3` plus several minutes,
# this script proves two cheap, high-signal facts in seconds using ONLY curl +
# sha256sum/shasum + python3 stdlib:
#
#   (1) The published PIGMIE patent PDF still hashes to its pinned SHA-256.
#   (2) The machine-readable claims.json is fetchable and is valid JSON.
#
# A PASS here does NOT replace verify_all.sh; it is a fast tripwire that the
# anchored documents have not changed and the release endpoints are live.
#
# Exit status: 0 if BOTH checks PASS, 1 otherwise.

set -eu

# ---- configuration ----------------------------------------------------------
PDF_URL="https://data.poliebotics.com/reality_kernel/pdfs/PIGMIE_Filing1_Description_v0_38.pdf"
PDF_EXPECTED_SHA="f0a635b7a0e152060cd1cefcf7f6f5eba554af554c7e1c2d4774ec4402485326"
CLAIMS_URL="https://data.truthbeam.com/release/claims.json"
UA="truthbeam-verify-quick/1.0 (+https://truthbeam.com)"

TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

# Track overall result without tripping `set -e`.
RC=0

# ---- helper: compute sha256 portably ----------------------------------------
sha256_of() {
    # $1 = path to file; prints the hex digest only
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{print $1}'
    else
        echo "ERROR: neither sha256sum nor shasum found" >&2
        return 2
    fi
}

echo "== verify_quick.sh (seconds-fast complement to verify_all.sh) =="
echo

# ---- check 1: patent PDF SHA-256 --------------------------------------------
echo "[1/2] PIGMIE patent PDF SHA-256"
echo "      url: $PDF_URL"
PDF_FILE="$TMPDIR/patent.pdf"
if curl -fsSL -A "$UA" -o "$PDF_FILE" "$PDF_URL"; then
    GOT_SHA="$(sha256_of "$PDF_FILE")"
    echo "      expected: $PDF_EXPECTED_SHA"
    echo "      actual:   $GOT_SHA"
    if [ "$GOT_SHA" = "$PDF_EXPECTED_SHA" ]; then
        echo "      PASS: PDF hash matches pinned value"
    else
        echo "      FAIL: PDF hash MISMATCH"
        RC=1
    fi
else
    echo "      FAIL: could not download PDF"
    RC=1
fi
echo

# ---- check 2: claims.json is valid JSON -------------------------------------
echo "[2/2] claims.json fetch + JSON validity"
echo "      url: $CLAIMS_URL"
CLAIMS_FILE="$TMPDIR/claims.json"
if curl -fsSL -A "$UA" -o "$CLAIMS_FILE" "$CLAIMS_URL"; then
    if python3 -m json.tool "$CLAIMS_FILE" >/dev/null 2>&1; then
        echo "      PASS: claims.json is valid JSON"
    else
        echo "      FAIL: claims.json is NOT valid JSON"
        RC=1
    fi
else
    echo "      FAIL: could not download claims.json"
    RC=1
fi
echo

# ---- overall result ---------------------------------------------------------
if [ "$RC" -eq 0 ]; then
    echo "== OVERALL: PASS =="
else
    echo "== OVERALL: FAIL =="
fi

exit "$RC"
