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

from mahjong_features import RiichiResNetFeatures, NUM_ACTIONS, NUM_FEATURES, NUM_TILES, get_action_from_index, get_action_index
from .random_discard_agent import RandomDiscardAgent

def tid136_to_t34(tid: int) -> int:
    return tid // 4

def _resize_batch(x, size=224):                     # [B,C,H,W]
    return torch.nn.functional.interpolate(x, (size, 65), mode="nearest")


def _select_device(device: str | torch.device | None) -> torch.device:
    """Resolve the preferred compute device with GPU fallback."""
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    if not isinstance(device, torch.device):
        try:
            device = torch.device(device)
        except (TypeError, ValueError):
            device = torch.device("cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    if device.type == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if not (mps_backend and mps_backend.is_available()):
            device = torch.device("cpu")
    print(device)
    return device

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

class VisualAgent:
    def __init__(self, env: gym.Env, backbone: str = "resnet18", device = None):
        self.env = env
        self._device = _select_device(device)
        self.model = VisualClassifier(backbone, in_chans = NUM_FEATURES, num_classes = NUM_ACTIONS, pretrained = False)
        self.model.to(self._device)
        self.extractor = RiichiResNetFeatures()
        self._alt_model = RandomDiscardAgent(env)
        self._ema = True

    def train(self, total_timesteps=100000):
        pass
 
    def save_model(self, path="resnet_mahjong"):
        pass
 
    def load_model(self, path="resnet18"):
        ckpt = torch.load(path, map_location=self._device, weights_only=False)
        if self._ema and ckpt["ema"]: 
            ema_weights = {
            k: v.clone().detach()
            for k, v in ckpt["ema"]["shadow"].items()
            }
            self.model.load_state_dict(ema_weights, strict=False)
        else:
            self.model.load_state_dict(ckpt["model"], strict=True)
        self.model.to(self._device)

    
    def predict(self, observation):
        legal_mask = np.asarray(self.env.action_masks())
        if sum(legal_mask) == 0:
            return 252
        if sum(legal_mask) == 1:
            return np.where(legal_mask==1)[0]
        
        # 如果当前状态是和牌状态，直接返回和牌动作
        if self.env and (self.env.phase in ["tsumo", "ron", "ryuukyoku"]):
            return get_action_index(None, self.env.phase)
        
        if self.env and (self.env.phase in ["kan", "chakan", "ankan"]):
            return 252

        if self.env and (self.env.phase in ["chi", "pon", "kan", "chakan", "ankan"]):
            # 推理时获取动作
            with torch.no_grad():
                out = self.extractor(observation[0])
                x = out["x"][None,:,:,:1].to(self._device, non_blocking=True)
                x = _resize_batch(x)
                self.model.eval()
                logits = self.model(x).detach()
                if logits.device.type != "cpu":
                    logits = logits.cpu()
                logits = logits.numpy().squeeze()
                legal_mask = np.asarray(self.env.action_masks())[:NUM_ACTIONS]
                logits += -1e9*(1-legal_mask) # mask to valid logits
                pred = int(logits.argmax()) # 253-dim
                return pred
                

        if self.env and (self.env.phase == "riichi"):
            # 推理时获取动作
            with torch.no_grad():
                observation[0].riichi_sticks += 1 # Workaround to handle inconsistency between generated data and rl env.
                out = self.extractor(observation[0])
                observation[0].riichi_sticks -= 1
                x = out["x"][None,:,:,:1].to(self._device, non_blocking=True)
                x = _resize_batch(x)
                self.model.eval()
                logits = self.model(x).detach()
                if logits.device.type != "cpu":
                    logits = logits.cpu()
                logits = logits.numpy().squeeze()
                legal_mask = np.asarray(self.env.action_masks())[:NUM_ACTIONS]
                logits += -1e9*(1-legal_mask) # mask to valid logits
                pred = int(logits.argmax()) # 253-dim
                
                return pred
                
                
        if self.env and (self.env.phase == "discard"):
            # 推理时获取动作
            with torch.no_grad():
                out = self.extractor(observation[0])
                x = out["x"][None,:,:,:1].to(self._device, non_blocking=True)
                x = _resize_batch(x)
                self.model.eval()
                logits = self.model(x).detach()
                if logits.device.type != "cpu":
                    logits = logits.cpu()
                logits = logits.numpy().squeeze()
                legal_mask = np.asarray(self.env.action_masks())[:NUM_ACTIONS]
                logits += -1e9*(1-legal_mask) # mask to valid logits
                pred = int(logits.argmax()) # 253-dim

                return pred
                        
        # if preds not in action_masks, return a random choice from action_masks.
        return self._alt_model.predict(observation)