"""Student A: one JSONL row per reasoning/trajectory pair; requires a CUDA GPU."""

import argparse
import json
import math
from pathlib import Path
import random
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clip-ids', '--clip_ids', type=Path, required=True,
                        help='Text file: one clip ID per line; blank lines and # comments allowed')
    parser.add_argument('--model', default='nvidia/Alpamayo-1.5-10B')
    parser.add_argument('--k', '--K', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=0.6)
    parser.add_argument('--top-p', type=float, default=0.98)
    parser.add_argument('--top-k', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--t0-us', type=int, default=5_100_000)
    parser.add_argument('--max-generation-length', type=int, default=256)
    parser.add_argument('--warmup', type=int, default=1,
                        help='Untimed inference calls per clip before measurement')
    parser.add_argument('--cuda-graph', action='store_true')
    parser.add_argument('--validate-only', action='store_true',
                        help='Validate arguments and clip list without loading GPU dependencies')
    args = parser.parse_args(argv)
    if args.k < 1 or args.warmup < 0 or args.max_generation_length < 1:
        parser.error('K and generation length must be positive; warmup must be nonnegative')
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error('temperature must be finite and positive')
    if not 0 < args.top_p <= 1 or (args.top_k is not None and args.top_k < 0):
        parser.error('top-p must be in (0, 1]; top-k must be nonnegative')
    if args.t0_us <= 1_600_000 or not 0 <= args.seed < 2**32:
        parser.error('t0-us must exceed history length; seed must be in [0, 2**32)')
    return args


def read_clip_ids(path):
    clips = [line.split('#', 1)[0].strip() for line in path.read_text().splitlines()]
    clips = [clip for clip in clips if clip]
    if not clips or len(clips) != len(set(clips)):
        raise ValueError('Clip list must be nonempty and contain no duplicates')
    return clips


def make_rows(args, clip_id, trajectories, reasoning, metrics):
    if len(trajectories) != args.k or len(reasoning) != args.k:
        raise ValueError('Expected exactly K paired trajectories and reasoning strings')
    for sample_id, (trajectory, text) in enumerate(zip(trajectories, reasoning)):
        if len(trajectory) != 64 or any(len(point) != 3 for point in trajectory):
            raise ValueError('Expected 64 x 3 ego-frame trajectory')
        if not isinstance(text, str):
            raise ValueError('Expected reasoning text')
        yield dict(clip_id=clip_id, sample_id=sample_id, model=args.model,
                   temperature=args.temperature, seed=args.seed, reasoning_text=text,
                   trajectory=trajectory, log_prob=None, runtime_s=metrics['runtime_s'],
                   perturbation=None, k=args.k, top_p=args.top_p, top_k=args.top_k,
                   t0_us=args.t0_us, **{key: value for key, value in metrics.items()
                                      if key != 'runtime_s'})


def main(argv=None):
    args = parse_args(argv)
    clips = read_clip_ids(args.clip_ids)
    if args.validate_only:
        print(f'Validated {len(clips)} clip IDs; K={args.k}. No inference or results written.')
        return
    # Lazy imports keep CLI validation usable on a laptop without CUDA dependencies.
    import numpy as np
    import torch
    from alpamayo1_5 import helper
    from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required. Palmetto measurements remain pending.')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True))
    model = Alpamayo1_5.from_pretrained(args.model, dtype=torch.bfloat16).to('cuda').eval()
    processor = helper.get_processor(model.tokenizer)
    if args.cuda_graph:
        model.enable_diffusion_expert_cuda_graph(max_batch_size=args.k, max_graphs=4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidentally replacing a previous experiment.
    with args.output.open('x') as output:
        for clip_id in clips:
            data = load_physical_aiavdataset(clip_id, t0_us=args.t0_us)
            messages = helper.create_message(frames=data['image_frames'].flatten(0, 1),
                                             camera_indices=data['camera_indices'])
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False,
                continue_final_message=True, return_dict=True, return_tensors='pt')
            model_inputs = helper.to_device(dict(tokenized_data=inputs,
                ego_history_xyz=data['ego_history_xyz'], ego_history_rot=data['ego_history_rot']),
                'cuda')

            def infer():
                return model.sample_trajectories_from_data_with_vlm_rollout(
                    data=model_inputs, top_p=args.top_p, top_k=args.top_k,
                    temperature=args.temperature, num_traj_samples=args.k, num_traj_sets=1,
                    max_generation_length=args.max_generation_length, return_extra=True)

            def seed():
                random.seed(args.seed)
                np.random.seed(args.seed)
                torch.manual_seed(args.seed)
                torch.cuda.manual_seed_all(args.seed)

            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                seed()
                for _ in range(args.warmup):
                    warm = infer()
                    del warm
                seed()  # Warmup must not advance the measured sampling RNG stream.
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                xyz, rot, extra = infer()
                torch.cuda.synchronize()
                runtime = time.perf_counter() - started
                metrics = dict(runtime_s=runtime,
                    peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
                    gpu_name=torch.cuda.get_device_name(), code_commit=commit, code_dirty=dirty,
                    alpamayo_source_commit='36aeb4c5938cbc2eb2aed33b22434773da4ab639',
                    model_revision=getattr(model.config, '_commit_hash', None),
                    torch_version=torch.__version__, cuda_version=torch.version.cuda,
                    warmup_calls=args.warmup, cuda_graph=args.cuda_graph,
                    cuda_graph_stats=model.diffusion_expert_cuda_graph_stats,
                    max_generation_length=args.max_generation_length,
                    runtime_scope='K-sample inference call; excludes load/preprocess/serialization')
                trajectories = xyz[0, 0].float().cpu().tolist()
                reasoning = extra['cot'][0, 0].tolist()
                for row in make_rows(args, clip_id, trajectories, reasoning, metrics):
                    output.write(json.dumps(row, allow_nan=False) + '\n')
                output.flush()
                print(f'{clip_id}: K={args.k}, {runtime:.3f}s, '
                      f"peak allocated={metrics['peak_gpu_allocated_bytes'] / 2**30:.3f} GiB")
                del xyz, rot, extra, model_inputs, inputs


if __name__ == '__main__':
    main()
