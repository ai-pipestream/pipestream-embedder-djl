#!/usr/bin/env bash
#
# Sweep every model currently loaded and READY on a DJL Serving endpoint, append
# rows to a single CSV using the same schema as the OVMS benchmarks. Discovers
# which models to run from GET /models so it stays in sync with whatever
# load-5-models.sh actually managed to load.
#
# Usage:
#   ./run-djl-benchmark.sh                                 # defaults below
#   DJL_URL=http://djl-serving-host:8080 ./run-djl-benchmark.sh
#
set -euo pipefail

DJL_URL="${DJL_URL:-http://localhost:8080}"
CORPUS_LIMIT="${CORPUS_LIMIT:-100}"
BATCH="${BATCH:-32}"
WARMUP="${WARMUP:-3}"
OUTPUT="${OUTPUT:-build/bench/sweep-5model-djl.csv}"
NOTES="${NOTES:-djl-benchmark}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
GRADLEW="$REPO_ROOT/gradlew"

# Dimensions per model — DJL has no metadata RPC, so we carry the dims table here.
# Keep aligned with the MODELS array in load-5-models.sh.
# Format: "served_name|dim|label_prefix"
MODEL_DIMS=(
  "all-MiniLM-L6-v2|384|minilm"
  "e5-small-v2|384|e5_small"
  "all-mpnet-base-v2|768|mpnet"
  "e5-large-v2|1024|e5_large"
  "bge-m3|1024|bge_m3"
)

if ! curl -s -f "$DJL_URL/ping" >/dev/null 2>&1; then
  echo "ERROR: DJL Serving not reachable at $DJL_URL/ping" >&2
  exit 1
fi

# Pull live model list once; filter the MODEL_DIMS table against it.
live_models=$(curl -s "$DJL_URL/models" | jq -r '.models[]?.modelName' 2>/dev/null || echo "")
if [[ -z "$live_models" ]]; then
  echo "ERROR: no models returned by $DJL_URL/models" >&2
  exit 1
fi

echo "==> Sweep config:"
echo "      url:         $DJL_URL"
echo "      batch:       $BATCH (corpus limit $CORPUS_LIMIT, warmup $WARMUP)"
echo "      output:      $REPO_ROOT/embedder-test-harness/$OUTPUT"
echo "      notes:       $NOTES"
echo "==> Live models on endpoint:"
echo "$live_models" | sed 's/^/      /'
echo

for entry in "${MODEL_DIMS[@]}"; do
  IFS='|' read -r model dim label_prefix <<< "$entry"

  if ! echo "$live_models" | grep -qx "$model"; then
    echo "==> $model not loaded on $DJL_URL, skipping"
    continue
  fi

  label="${label_prefix}-djl-batch${BATCH}"
  echo "==> $model (dim=$dim) → label='$label'"

  ( cd "$REPO_ROOT" && "$GRADLEW" :embedder-test-harness:runDjl --no-daemon --quiet --console=plain \
      --args="--url $DJL_URL --model $model --dim $dim \
              --batch $BATCH --warmup $WARMUP \
              --corpus-limit $CORPUS_LIMIT \
              --label $label \
              --notes $NOTES \
              --output $OUTPUT" ) | grep -E "throughput|latency|provider=|FAIL" || true
done

echo
echo "==> Done. Results appended to $REPO_ROOT/embedder-test-harness/$OUTPUT"
