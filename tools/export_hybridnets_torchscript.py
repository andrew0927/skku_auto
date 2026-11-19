#!/usr/bin/env python3
"""Convert the official HybridNets weights (hybridnets.pth) to the TorchScript file the car uses.

usage: python3 tools/export_hybridnets_torchscript.py <HybridNets repo> <hybridnets.pth> \
           [models/hybridnets_384x512/hybridnets_384x512.torchscript.pt]
Needs the repo https://github.com/datvuthanh/HybridNets and its packages (efficientnet_pytorch,
timm 0.6, pretrainedmodels, prefetch_generator, webcolors, tqdm) only for this conversion; the car
then loads the single .pt file with torch.jit.load (no repo code). Input 384x512 (640x480 resized),
detection head only: outputs (regression, classification).
"""
import sys
from pathlib import Path

import torch

repo, weights = Path(sys.argv[1]).resolve(), sys.argv[2]
out = sys.argv[3] if len(sys.argv) > 3 else str(
    Path(__file__).resolve().parent.parent / 'models/hybridnets_384x512/hybridnets_384x512.torchscript.pt')
sys.path.insert(0, str(repo))
from backbone import HybridNetsBackbone  # noqa: E402

net = HybridNetsBackbone(num_classes=1, compound_coef=3, ratios=[(0.62, 1.58), (1.0, 1.0), (1.58, 0.62)],
                         scales=[2 ** 0, 2 ** 0.70, 2 ** 1.32], seg_classes=2, backbone_name=None,
                         seg_mode='multiclass', onnx_export=True)
net.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
net.eval()


class Detection(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        regression, classification, _ = self.net(x)
        return regression, classification


x = torch.randn(1, 3, 384, 512)
with torch.inference_mode():
    model = Detection(net).eval()
    traced = torch.jit.freeze(torch.jit.trace(model, x, check_trace=False))
    traced.save(out)
    diff = max((a - b).abs().max().item() for a, b in zip(model(x), torch.jit.load(out)(x)))
print(f'saved {out} (max difference to the original model {diff:.2e})')
