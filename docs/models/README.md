# DJL Serving custom Python handlers

These handlers replace DJL's shipped `djl://` traced models because three of
the five traced models we tried are broken on modern CUDA (NVIDIA 4080 Super,
CUDA 12.x):

- `intfloat/e5-small-v2`, `intfloat/e5-large-v2` — nvrtc fusion compile
  fails with `extra text after expected end of number` on the constant
  `-3.402823466385289e+38.f`. Affects any DJL traced model that uses fused
  attention with softmax masking via `-MAX_FLOAT`.
- `sentence-transformers/all-mpnet-base-v2` — traced `position_bias` lookup
  runs on CPU while the rest of the model is on GPU, throwing
  `Expected all tensors to be on the same device, but found at least two
  devices, cuda:0 and cpu!`.

Only `sentence-transformers/all-MiniLM-L6-v2` actually works through DJL's
PyTorch traced path on this GPU; that one is loaded as a normal `djl://`
model by `scripts/load-5-models.sh`.

`BAAI/bge-m3` is in here for a different reason: DJL's default HF tokenizer
rejects it because bge-m3 ships custom Python code for its XLM-RoBERTa-XL
tokenizer, and transformers 5.x refuses `torch.load()` on the `.bin` weights
without torch ≥ 2.6 (CVE-2025-32434). The container ships torch 2.5.1, so
we force `use_safetensors=True` in every handler — it sidesteps the CVE
check and is faster anyway.

## Layout

```
models/
  all-MiniLM-L6-v2/     # placeholder; loaded from djl:// zoo in load-5-models.sh
  e5-small-v2/
    model.py
    serving.properties
  e5-large-v2/...
  all-mpnet-base-v2/...
  bge-m3/...
```

## Deploying

The handlers get copied into the running DJL container at `/tmp/<name>/`
and registered via the management API. `scripts/load-5-models.sh` knows
which models take the Python path vs the traced path.

## serving.properties knobs

- `engine=Python` — switches DJL to the Python backend instead of PyTorch
- `option.model_id=<hf-id>` — sentence-transformers loads this on first call
- `job_queue_size=2000` — default is too small under the ~600 ms per-request
  DJL dispatch overhead we measured; 2000 covers `c=128` client bursts
- `batch_size=1` + `max_batch_delay=0` — we pre-batch client-side (32
  sentences per HTTP request), so DJL's server-side batching adds latency
  without adding throughput
