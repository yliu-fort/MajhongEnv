from __future__ import annotations
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import gymnasium as gym
import torch
import numpy as np

import torch
import torch.nn as nn

try:
    import timm  # for ViT and other backbones
except ImportError:
    timm = None

from mahjong_features import RiichiResNetFeatures, NUM_FEATURES, NUM_TILES
from .random_discard_agent import RandomDiscardAgent

def tid136_to_t34(tid: int) -> int:
    return tid // 4

def _resize_batch(x, size=224):                     # [B,C,H,W]
    return torch.nn.functional.interpolate(x, (size, 65), mode="nearest")

class VisualClassifier(nn.Module):
    def __init__(self, backbone: str = "resnet18", in_chans: int = 3, num_classes: int = 10, pretrained: bool = True):
        super().__init__()
        self.backbone_name = backbone.lower()

        try:
            self.model = timm.create_model(self.backbone_name, pretrained=pretrained, num_classes=num_classes, in_chans=in_chans)
        except ValueError:
            raise ValueError(f"Unsupported backbone: {backbone}")

    def forward(self, x):
        return self.model(x)


#TODO: add GPU computation support
class VisualAgent:
    def __init__(self, env: gym.Env, backbone: str = "resnet18", device = 'cpu'):
        self.env = env
        self.model = VisualClassifier(backbone, in_chans = NUM_FEATURES, num_classes = NUM_TILES, pretrained = False)
        self.extractor = RiichiResNetFeatures()
        self._alt_model = RandomDiscardAgent(env)
        self._ema = True
        self._device = device
 
    def train(self, total_timesteps=100000):
        pass
 
    def save_model(self, path="resnet_mahjong"):
        pass
 
    def load_model(self, path="resnet18"):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if self._ema and ckpt["ema"]: 
            ema_weights = {
            k: v.clone().detach()
            for k, v in ckpt["ema"]["shadow"].items()
            }
            self.model.load_state_dict(ema_weights, strict=False)
        else:
            self.model.load_state_dict(ckpt["model"], strict=True)

    
    def predict(self, observation):
        # 如果当前状态是和牌状态，直接返回和牌动作
        if self.env and (self.env.phase == "tsumo" or self.env.phase == "ron"):
            return (0, True)
        
        # No options yet
        if self.env and (self.env.phase == "riichi"):
            return self._alt_model.predict(observation)[0], True if self.env.num_riichi < 3 else False

        if self.env and (self.env.phase == "discard"):
            # 推理时获取动作
            with torch.no_grad():
                out = self.extractor(observation[0])
                x = out["x"][None,:,:,0]
                x = _resize_batch(x)
                self.model.eval()
                logits = self.model(x).detach().numpy().squeeze()
                logits += -1e9*(1-np.array(out["legal_mask"])) # mask to valid logits
                pred = int(logits.argmax()) # tile-34

                # Check if pred falls in valid action_masks
                action_masks = self.env.action_masks() # 0 - 13 position in hand
                for i, x in enumerate(self.env.hands[observation[1]['who']]):
                    if tid136_to_t34(x) == pred and action_masks[i] == True:
                        return (i, False)

        # if preds not in action_masks, return a random choice from action_masks.
        return self._alt_model.predict(observation)[0], False