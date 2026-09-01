#!/usr/bin/env python3.8
"""
export_onnx.py — Export SB3 PPO policy checkpoints to ONNX.

Run inside the simulation container (Python 3.8 + SB3 + torch are available):
  python3.8 /root/rospug_research/scripts/export_onnx.py

Output (written alongside each .zip):
  checkpoints/policy_A_500k.onnx
  checkpoints/policy_B_500k.onnx
  checkpoints/policy_C_500k.onnx

ONNX model contract:
  Input  "obs"    — float32[batch, 26]  — raw observation vector (no normalization)
  Output "action" — float32[batch, 12]  — unclipped action means; caller clips to ±GAIT_RESIDUAL
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from stable_baselines3 import PPO

_ROOT          = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
CHECKPOINT_DIR = os.path.join(_ROOT, 'checkpoints')

CHECKPOINTS = [
    ('policy_A_500k.zip', 'policy_A_500k.onnx'),
    ('policy_B_500k.zip', 'policy_B_500k.onnx'),
    ('policy_C_500k.zip', 'policy_C_500k.onnx'),
]

OBS_DIM    = 26
ACTION_DIM = 12


class _ActorOnly(nn.Module):
    """Wraps only the PPO actor path for ONNX export.

    SB3 PPO actor path: obs → pi_features_extractor → mlp_extractor.forward_actor → action_net.
    For MlpPolicy with Box obs the features extractor is FlattenExtractor (identity for 1-D input).
    No observation normalization is present — these policies were trained without VecNormalize.
    """

    def __init__(self, policy):
        super().__init__()
        self._feat = policy.pi_features_extractor
        self._mlp  = policy.mlp_extractor
        self._act  = policy.action_net

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        feat = self._feat(obs)
        lat  = self._mlp.forward_actor(feat)
        return self._act(lat)


def export_policy(zip_path: str, onnx_path: str) -> None:
    print(f"\n[export] Loading  {zip_path}")
    model  = PPO.load(zip_path, device='cpu')
    policy = model.policy
    policy.eval()

    actor = _ActorOnly(policy)
    actor.eval()

    dummy = torch.zeros(1, OBS_DIM, dtype=torch.float32)

    # Sanity check: actor output must be finite and right shape
    with torch.no_grad():
        out = actor(dummy)
    assert out.shape == (1, ACTION_DIM), f"Unexpected shape: {out.shape}"
    assert torch.isfinite(out).all(), "Actor output contains NaN/Inf"

    torch.onnx.export(
        actor,
        dummy,
        onnx_path,
        input_names=['obs'],
        output_names=['action'],
        dynamic_axes={'obs': {0: 'batch'}, 'action': {0: 'batch'}},
        opset_version=11,
        do_constant_folding=True,
        verbose=False,
    )
    print(f"[export] Saved    {onnx_path}")

    # Verify the exported ONNX model with onnxruntime
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        obs_np = np.zeros((1, OBS_DIM), dtype=np.float32)
        action_np = sess.run(['action'], {'obs': obs_np})[0]
        assert action_np.shape == (1, ACTION_DIM)
        # Compare to torch output for the same input
        with torch.no_grad():
            ref = actor(torch.from_numpy(obs_np)).numpy()
        max_diff = float(np.abs(action_np - ref).max())
        print(f"[verify] onnxruntime vs torch max diff: {max_diff:.2e}  {'OK' if max_diff < 1e-5 else 'WARNING'}")
    except ImportError:
        print("[verify] onnxruntime not installed — skipping runtime verification")


def main() -> None:
    for zip_name, onnx_name in CHECKPOINTS:
        zip_path  = os.path.join(CHECKPOINT_DIR, zip_name)
        onnx_path = os.path.join(CHECKPOINT_DIR, onnx_name)

        if not os.path.exists(zip_path):
            print(f"[skip] Not found: {zip_path}")
            continue

        export_policy(zip_path, onnx_path)

    print("\n[export] Done.")


if __name__ == '__main__':
    main()
