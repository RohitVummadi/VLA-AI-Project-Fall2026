# Student A — Week 3 initial implementation

Status: CPU contract checks only. Palmetto access is pending. No GPU inference,
measured runtime/memory, 10-clip output, trajectory plots, or GPU-hours result exists yet.

Inspected upstream commit: `36aeb4c5938cbc2eb2aed33b22434773da4ab639`.
These inspected files live in the separate NVlabs/alpamayo1.5 repository:
`src/alpamayo1_5/test_inference.py`, `src/alpamayo1_5/models/alpamayo1_5.py`,
`src/alpamayo1_5/load_physical_aiavdataset.py`, and its README.

## Code inspection findings

(a) The standard rollout sets `num_return_sequences=num_traj_samples` and pairs
reasoning with trajectories in `[batch, num_traj_sets, K]` order. We fix sets=1.
Independent stochastic draws do not guarantee distinct reasoning strings.

(b) Two-level K reasoning × J trajectory sampling is not implemented here.
The current API multiplies diffusion batch size by sets but does not explicitly
repeat the reasoning cache accordingly, and reshapes K text outputs into sets×K.
Support for sets>1 needs a targeted GPU experiment and likely a model change.

(c) Although its docstring mentions logprob, the standard function returns only
positions/rotations, plus text when requested. Internal `output_logits=True`
does not expose scores. `log_prob` is therefore JSON null. Extracting reasoning
scores requires defining token boundaries, padding masks and raw vs sampling-policy
probabilities, then modifying the inference implementation.

(d) Temperature, top-p, top-k, K and external random seeds are supported. The
script seeds Python, NumPy, PyTorch and CUDA before each clip's measured call;
this does not promise bitwise reproducibility across GPU/software configurations.

(e) The upstream README comparison table and RL FAQ explicitly describe 1.5
as RL post-trained. This is documentation evidence, not a new training experiment.

## JSONL contract and measurement scope

One row per sample: `clip_id, sample_id` (zero based), `model, temperature, seed,
reasoning_text, trajectory` (64×3 xyz meters in ego frame at t0, 10 Hz),
`log_prob` (null), `runtime_s, perturbation` (null for clean inputs).
Additional fields preserve K, top-p/top-k, timestamp, commit/dirty state, model
revision when available, GPU/software information, generation limit and warmup/graph settings.

`runtime_s` is synchronized wall time for the entire K-sample inference call,
repeated on each of that clip's K rows; do not sum repeated rows for GPU-hours.
It excludes model/data loading, preprocessing, warmup, CPU transfer and JSON writing.
Memory is peak PyTorch allocated/reserved bytes on the current GPU, including
resident model/input/graph buffers; it is not total device usage from nvidia-smi.
Peak counters reset after per-clip warmup. Allocator reservations and CUDA graph
captures may persist; benchmark each K and graph setting in a fresh process.
Cold graph captures can still occur when a measured shape differs from warmup;
inspect the graph counters before claiming a speedup. No logprob extraction overhead
is added by this script beyond the stock model implementation.

Existing output files are never overwritten. A failed run can leave a partial
JSONL; validate the expected clip×K count before using it in the report.

## Once Palmetto is available

1. Transfer this team checkout/commit to Palmetto and obtain one suitable CUDA GPU.
   Clone the upstream repository separately, pin the inspected revision and follow
   its README setup (Python 3.12, uv environment, dependencies, HF login):

```sh
git clone https://github.com/NVlabs/alpamayo1.5.git
cd alpamayo1.5
git checkout 36aeb4c5938cbc2eb2aed33b22434773da4ab639
uv venv
source .venv/bin/activate
uv sync --active
hf auth login
python src/alpamayo1_5/test_inference.py
```

   Keep that environment activated and change back to the team repository before
   running the commands below. Its installed Alpamayo package supplies imports;
   the team repository does not vendor the model. If Flash Attention setup fails,
   follow the upstream troubleshooting section.
2. Obtain the team's actual fixed 10 clip IDs and save one per line in
   `configs/ten_clips.txt`. The supplied example file has only the upstream smoke-test clip.
3. Validate the manifest, then benchmark fresh processes:

```sh
python sampling/sample.py --clip-ids configs/ten_clips.txt --k 16 --output results/check.jsonl --validate-only
for k in 1 4 8 16; do
  python sampling/sample.py --clip-ids configs/ten_clips.txt \
    --model nvidia/Alpamayo-1.5-10B --k "$k" --temperature 0.6 \
    --top-p 0.98 --seed 42 --warmup 1 --output "results/ten_clips_k${k}.jsonl"
done
```

Repeat with `--cuda-graph` and different output filenames. Measure variability
with additional fresh-process repeats before reporting a graph speedup.
If a GPU runs out of memory, report the failure and GPU model; do not substitute
published numbers. K=16 may need more capacity than K=1.

4. Check 10×K rows, finite 64×3 trajectories, and reasoning alignment. Plot 2–3
   real examples; give Student C at least 50 real reasoning samples from K=16.
5. Aggregate one runtime/memory record per clip per K, reporting warmup and graph
   settings. Estimate inference-only GPU-hours as
   `1300 * mean_K16_runtime_s / 3600` for one GPU; separately account for startup,
   data loading, warmup, retries and experiments when estimating allocation needs.
6. Verify two-level sampling experimentally, implement token scoring if needed,
   and complete the separately requested Alpamayo 1 smoke test. Add results and this
   script's producing commit to the team's private repository and combined report.

| K | Runtime/clip | Peak GPU memory |
|---|---|---|
| 1 | Pending Palmetto | Pending |
| 4 | Pending Palmetto | Pending |
| 8 | Pending Palmetto | Pending |
| 16 | Pending Palmetto | Pending |

CPU checks: `python3 -m unittest discover -s sampling -p 'test_*.py'`.

Store generated JSONL in the shared Palmetto directory per the team README.
Replace `results/` in the examples with an absolute shared run directory.
The `code_commit` field records the team script commit; the upstream source
revision is recorded separately as `alpamayo_source_commit`.
