

import argparse
import time

import torch
import yaml

from models.msdr_net import build_model
from models.baselines import build_baseline, BASELINE_NAMES

WARMUP = 50
TIMED_RUNS = 200


def count_parameters_and_macs(model, input_size: int = 224):
    from fvcore.nn import FlopCountAnalysis
    params = sum(p.numel() for p in model.parameters())
    dummy = torch.randn(1, 3, input_size, input_size)
    flops = FlopCountAnalysis(model, dummy)
    return params, int(flops.total())  # MACs under the 1 MAC = 1 FLOP convention


def _time_gpu(model, batch_size: int, device) -> float:
    model = model.to(device).eval()
    dummy = torch.randn(batch_size, 3, 224, 224, device=device)
    with torch.no_grad():
        for _ in range(WARMUP):
            model(dummy)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(TIMED_RUNS):
            model(dummy)
        end.record()
        torch.cuda.synchronize()
    return start.elapsed_time(end) / TIMED_RUNS / batch_size  # ms per image


def _time_cpu(model, batch_size: int = 1) -> float:
    model = model.cpu().eval()
    dummy = torch.randn(batch_size, 3, 224, 224)
    with torch.no_grad():
        for _ in range(WARMUP):
            model(dummy)
        t0 = time.perf_counter()
        for _ in range(TIMED_RUNS):
            model(dummy)
        elapsed = time.perf_counter() - t0
    return elapsed / TIMED_RUNS / batch_size * 1000.0  # ms per image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default_config.yaml")
    parser.add_argument("--model", required=True,
                        help="msdr_net[_full|_no_eca|_single|_single_eca] or a baseline name")
    args = parser.parse_args()

    with open(args.config) as f:
        yaml.safe_load(f)  # config reserved for protocol consistency

    if args.model in BASELINE_NAMES:
        model = build_baseline(args.model, pretrained=False)
    else:
        model = build_model(args.model)

    params, macs = count_parameters_and_macs(model)
    print(f"Parameters: {params / 1e6:.1f}M")
    print(f"Computational cost: {macs / 1e9:.2f} GMACs (1 MAC = 1 FLOP, input 224 x 224)")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU batch-averaged throughput (batch 16, FP32): "
              f"{_time_gpu(model, 16, device):.2f} ms/image")
        print(f"GPU single-instance latency (batch 1, FP32): "
              f"{_time_gpu(model, 1, device):.2f} ms/image")
    else:
        print("CUDA not available; skipping GPU timing.")
    print(f"CPU single-instance latency (batch 1, wall clock): "
          f"{_time_cpu(model, 1):.1f} ms/image")


if __name__ == "__main__":
    main()
