#!/usr/bin/env bash
#
# Load the 5-model set into DJL Serving for the OVMS vs DJL head-to-head benchmark.
# Models match the OpenVINO catalog in docs/openvino/scripts/setup-models.sh so rows
# in the benchmark CSV can be diffed 1:1.
#
# Models are loaded smallest first so a failure on the big models (expected for
# bge-m3 on a 16 GB GPU) doesn't block the rest of the sweep.
#
# Usage:
#   ./load-5-models.sh                         # default http://localhost:8090
#   ./load-5-models.sh http://localhost:8090
#   DJL_ENGINE=PyTorch ./load-5-models.sh      # override engine if needed
#
# Required:
#   curl, jq, DJL Serving reachable at $BASE_URL with /ping responding 200.
#
set -uo pipefail

BASE_URL="${1:-http://localhost:8090}"
ENGINE="${DJL_ENGINE:-PyTorch}"

# Format: "hf_id|djl_served_name"
# djl_served_name is what appears in /models and the URL for /predictions/<name>.
# Kept close to the OVMS pipeline names (minus the _pipeline suffix) so labels line up.
MODELS=(
  "sentence-transformers/all-MiniLM-L6-v2|all-MiniLM-L6-v2"
  "intfloat/e5-small-v2|e5-small-v2"
  "sentence-transformers/all-mpnet-base-v2|all-mpnet-base-v2"
  "intfloat/e5-large-v2|e5-large-v2"
  "BAAI/bge-m3|bge-m3"
)

loaded=()
failed=()
load_times_ms=()

echo "==> Waiting for DJL Serving at $BASE_URL..."
for i in {1..60}; do
  if curl -s -f "$BASE_URL/ping" >/dev/null 2>&1; then
    echo "==> DJL Serving is up."
    break
  fi
  sleep 1
  if [[ $i -eq 60 ]]; then
    echo "ERROR: DJL Serving did not respond at $BASE_URL/ping within 60s" >&2
    exit 1
  fi
done

url_encode() {
  python3 -c "import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1], safe=''))" "$1"
}

for entry in "${MODELS[@]}"; do
  IFS='|' read -r hf_id served_name <<< "$entry"

  echo
  echo "============================================================"
  echo "  Loading $hf_id as '$served_name'"
  echo "============================================================"

  # Skip if already loaded — idempotent re-runs
  if curl -s "$BASE_URL/models/$(url_encode "$served_name")" 2>/dev/null | jq -e '.models[0].status == "Healthy"' >/dev/null 2>&1; then
    echo "  already loaded, skipping"
    loaded+=("$served_name")
    load_times_ms+=("cached")
    continue
  fi

  model_url="djl://ai.djl.huggingface.pytorch/${hf_id}"
  encoded_url=$(url_encode "$model_url")
  encoded_name=$(url_encode "$served_name")

  start_ms=$(date +%s%3N)
  response=$(curl -s -w "\n%{http_code}" -X POST \
      "$BASE_URL/models?url=${encoded_url}&model_name=${encoded_name}&engine=${ENGINE}&synchronous=true")
  end_ms=$(date +%s%3N)
  elapsed_ms=$((end_ms - start_ms))

  http_code=$(echo "$response" | tail -1)
  body=$(echo "$response" | head -n -1)

  if [[ "$http_code" =~ ^2 ]]; then
    echo "  loaded in ${elapsed_ms} ms"
    loaded+=("$served_name")
    load_times_ms+=("$elapsed_ms")
  else
    echo "  FAILED with HTTP $http_code after ${elapsed_ms} ms"
    echo "  body: $(echo "$body" | head -c 500)"
    failed+=("$served_name")
  fi
done

echo
echo "============================================================"
echo "  Summary"
echo "============================================================"
echo "Loaded (${#loaded[@]}):"
for i in "${!loaded[@]}"; do
  printf "  %-25s %s ms\n" "${loaded[$i]}" "${load_times_ms[$i]}"
done
if [[ ${#failed[@]} -gt 0 ]]; then
  echo "Failed (${#failed[@]}):"
  for m in "${failed[@]}"; do
    echo "  $m"
  done
fi

echo
echo "Live status (curl $BASE_URL/models | jq '.models | length'):"
curl -s "$BASE_URL/models" | jq '.models | length' 2>/dev/null || echo "  (jq parse failed — see raw response above)"

# Exit non-zero only if NOTHING loaded. Partial success is acceptable per plan.
if [[ ${#loaded[@]} -eq 0 ]]; then
  exit 1
fi
