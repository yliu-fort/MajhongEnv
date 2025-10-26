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

from mahjong_features import RiichiResNetFeatures, NUM_ACTIONS, NUM_FEATURES, NUM_TILES, get_action_from_index, get_action_index, get_action_type_from_index
from .random_discard_agent import RandomDiscardAgent
from my_types import Response, ActionSketch, Seat, ActionType

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
        #who = observation[1]["who"]
        legal_mask = np.asarray(observation.legal_actions_mask)
        # DISABLE KAN/ANKAN/CHAKAN
        legal_mask[147:249]=False
        valid_action_list = [i for i in list(range(NUM_ACTIONS)) if legal_mask[i]]
        allowed_action_type = set([get_action_type_from_index(i) for i, a in enumerate(legal_mask) if a])

        if sum(legal_mask) == 0:
            return None
        elif sum(legal_mask) == 1:
            action_id = valid_action_list[0]
            return ActionSketch(action_type=get_action_type_from_index(action_id), payload={"action_id": action_id})

        if (any([_ in allowed_action_type for _ in [ActionType.TSUMO,]])):
            return ActionSketch(action_type=ActionType.TSUMO, payload={"action_id": 251})

        if (any([_ in allowed_action_type for _ in [ActionType.RON,]])):
            return ActionSketch(action_type=ActionType.RON, payload={"action_id": 250})

        # 推理时获取动作
        with torch.no_grad():
            out = self.extractor(observation)
            x = out["x"][None,:,:,:1].to(self._device, non_blocking=True)
            x = _resize_batch(x)
            self.model.eval()
            logits = self.model(x).detach()
            if logits.device.type != "cpu":
                logits = logits.cpu()
            logits = logits.numpy().squeeze()
            #legal_mask = np.asarray(self.env.action_masks())
            masked_logits = logits-1e9*(1-legal_mask) # mask to valid logits
            pred = int(masked_logits.argmax()) # 262-dim
            pred1 = None
            
            if sum(legal_mask[34:68]) > 0:
                logits_0 = np.asarray([x if i >= 34 and i < 68 else -1e9 for i, x in enumerate(masked_logits)])
                logits_0[253] = logits[253]
                pred1 = int(logits_0.argmax()) # 262-dim
            if pred1 is not None and pred1 != 253:
                pred = pred1 # 262-dim
            
            return ActionSketch(action_type=get_action_type_from_index(pred), payload={"action_id": pred})
                    
        # if preds not in action_masks, return a random choice from action_masks.
        return self._alt_model.predict(observation)


    def policy_distribution(self, observation, top_k: int = 5):
        """Return the policy distribution and top-k pairs for an observation."""
        _, distribution, top_actions = self.predict_with_distribution(
            observation, top_k=top_k, enable_all_actions=True
        )
        return distribution, top_actions

    def predict_with_distribution(self, observation, top_k: int = 5, enable_all_actions=True):
        legal_mask_full = np.asarray(observation.legal_actions_mask)
        legal_mask_full[252]=False # TODO: temporary fix
        legal_sum = int(legal_mask_full.sum())
        distribution = np.zeros(NUM_ACTIONS, dtype=np.float32)

        if legal_sum == 0:
            fallback = 252
            if fallback < NUM_ACTIONS:
                distribution[fallback] = 1.0
            return fallback, distribution, [(fallback, 1.0)]

        if legal_sum == 1:
            action = int(np.where(legal_mask_full == 1)[0].item())
            if action < NUM_ACTIONS:
                distribution[action] = 1.0
            return action, distribution, [(action, 1.0)]

        # 如果当前状态是和牌状态，直接返回和牌动作
        if enable_all_actions:
            valid_phases = {"draw", "kan_draw", "tsumo", "ron", "ryuukyoku", "kan", "chakan", "ankan", \
                            "discard", "riichi", "chi", "pon", "kan", "chakan", "ankan", "?"}
        else:
            valid_phases = {"draw", "kan_draw", "discard"}

        if self.env and (self.env.phase in valid_phases):
            with torch.no_grad():
                out = self.extractor(observation)
                x = out["x"][None, :, :, :1].to(self._device, non_blocking=True)
                x = _resize_batch(x)
                self.model.eval()
                logits = self.model(x).detach()
                if logits.device.type != "cpu":
                    logits = logits.cpu()
                logits = logits.view(-1)
                legal_mask_np = np.asarray(legal_mask_full, dtype=bool)

                legal_mask_tensor = torch.from_numpy(legal_mask_np)
                masked_logits = torch.full_like(logits, float("-inf"))
                masked_logits[legal_mask_tensor] = logits[legal_mask_tensor]

                valid_logits = masked_logits[legal_mask_tensor]
                max_logit = torch.max(valid_logits)
                stable_logits = valid_logits - max_logit
                exp_logits = torch.exp(stable_logits)
                denom = torch.sum(exp_logits)

                probs = torch.zeros_like(logits, dtype=torch.float32)
                probs[legal_mask_tensor] = exp_logits / denom
                distribution = probs.numpy()

                legal_indices = np.where(legal_mask_np)[0]
                sorted_idx = legal_indices[np.argsort(distribution[legal_indices])[::-1]]

                top = [
                    (int(idx), float(distribution[idx]))
                    for idx in sorted_idx[: max(1, top_k)]
                ]
                pred = int(sorted_idx[0])
                return pred, distribution, top