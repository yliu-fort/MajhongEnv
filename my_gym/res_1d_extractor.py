import numpy as np
import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torchvision.models import resnet18


# --------- 1D ResNet 基本残差块 ----------
class BasicBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, groups=8, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1   = nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch)
        #self.bn1   = nn.BatchNorm1d(out_ch)
        self.act   = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2   = nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch)
        #self.bn2   = nn.BatchNorm1d(out_ch)
        self.drop  = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # 如果通道或步幅不一致，用 1x1 投影做 downsample
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                #nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch),
                nn.BatchNorm1d(out_ch)
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.gn2(out)
        out = self.drop(out)
        out = out + self.downsample(identity)
        #out += identity
        out = self.act(out)
        return out

# --------- 1D ResNet 提取器（支持 Box 或 Dict['observation']） ----------

class ResNet1DExtractor(BaseFeaturesExtractor):
    """
    1D ResNet 风格特征提取器（输入 (B,C,H)；若 channels_last=True 则输入 (B,H,C)）
    """
    def __init__(
        self,
        observation_space: spaces.Space,
        features_dim: int = 512,
        base_channels: int = 256,
        blocks_per_stage=(2, 2, 1, 1),
        channels_per_stage=None,
        stem_stride: int = 1,
        use_maxpool: bool = False,
        dropout: float = 0.0,
        groups: int = 64,
        channels_last: bool = False,
    ):
        super().__init__(observation_space, features_dim)
        self.channels_last = channels_last

        # 解析空间
        obs_space = observation_space["observation"] if isinstance(observation_space, spaces.Dict) else observation_space
        assert isinstance(obs_space, spaces.Box)
        assert len(obs_space.shape) == 2, f"期望 2D Box (C,H) 或 (H,C)，收到 {obs_space.shape}"

        a, b = int(obs_space.shape[0]), int(obs_space.shape[1])
        if channels_last:
            self.in_ch, self.length = b, a   # 观测是 (H,C)
        else:
            self.in_ch, self.length = a, b   # 观测是 (C,H)

        if channels_per_stage is None:
            scale = base_channels // 64
            channels_per_stage = [64*scale, 128*scale, 128*scale, 256*scale]

        # stem
        self.stem = nn.Sequential(
            nn.Conv1d(self.in_ch, channels_per_stage[0], kernel_size=1, stride=stem_stride, padding=0, bias=False),
            nn.GroupNorm(num_groups=min(groups, channels_per_stage[0]), num_channels=channels_per_stage[0]),
            #nn.BatchNorm1d(channels_per_stage[0]),
            nn.SiLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1) if use_maxpool else nn.Identity(),
        )

        # stages
        stages = []
        ch_in = channels_per_stage[0]
        for i, (num_blocks, ch_out) in enumerate(zip(blocks_per_stage, channels_per_stage)):
            for b in range(num_blocks):
                stride = 2 if (b == 0 and i > 0) else 1
                stages.append(BasicBlock1D(ch_in, ch_out, stride=stride, groups=groups, dropout=dropout))
                ch_in = ch_out
        self.backbone = nn.Sequential(*stages)

        self.pool = nn.AdaptiveAvgPool1d(2)
        self.flatten = nn.Flatten()
        #self.proj = nn.Linear(ch_in, features_dim)
        self._features_dim = features_dim

    def forward(self, obs):
        x = obs["observation"] if isinstance(obs, dict) else obs
        x = x.float()  # 期望 (B,C,H) 或 (B,H,C)

        # 整理到 (B,C,H)，严格按 in_channels 匹配，不再用“大小比较”的启发式
        if x.ndim == 2:
            # (B,H) -> 这里无法区分通道，只能视作单通道
            x = x.unsqueeze(1)  # (B,1,H)
            if self.in_ch != 1:
                raise RuntimeError(f"收到 (B,H) 但 in_channels={self.in_ch}，请提供 (B,C,H) 或设置 channels_last")
        elif x.ndim == 3:
            if self.channels_last:
                # 期望 (B,H,C) -> (B,C,H)
                if x.shape[2] != self.in_ch and x.shape[1] == self.in_ch:
                    # 已经是 (B,C,H)，无需转
                    pass
                else:
                    x = x.transpose(1, 2).contiguous()
            else:
                # 期望 (B,C,H)
                if x.shape[1] != self.in_ch and x.shape[2] == self.in_ch:
                    x = x.transpose(1, 2).contiguous()
                elif x.shape[1] != self.in_ch:
                    raise RuntimeError(
                        f"输入通道 {x.shape[1]} 与期望 {self.in_ch} 不符；"
                        f"若你的数据是 (B,H,C)，请设置 channels_last=True"
                    )
        else:
            raise RuntimeError(f"期望张量维度为 2 或 3，收到 {x.ndim}")

        z = self.stem(x)
        z = self.backbone(z)
        z = self.pool(z).squeeze(-1)
        z = self.flatten(z)
        #z = self.proj(z)
        return z



class ResNet18Extractor(BaseFeaturesExtractor):
    """
    用 ResNet-18 提取 Dict 观测中 'observation' 的特征，输出 features_dim 向量。
    忽略 'action_mask'（MaskablePPO 会单独读取掩码）。
    """
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 512, pretrained: bool = False):
        # 先调用父类：声明最后的特征维度
        super().__init__(observation_space, features_dim)
        assert isinstance(observation_space, spaces.Dict), "请用 MultiInputPolicy + Dict 观测"
        obs_space = observation_space["observation"]
        assert isinstance(obs_space, spaces.Box), "observation 必须是 Box"

        # 推断通道数/高/宽（支持 (H,W) 或 (C,H,W)）
        if len(obs_space.shape) == 2:
            c, h, w = obs_space.shape[0], 224, 65
        elif len(obs_space.shape) == 3:
            # SB3 期望图像是 CHW
            c, h, w = obs_space.shape
        else:
            raise ValueError(f"不支持的 observation 形状: {obs_space.shape}")

        # 构建 ResNet-18
        self.backbone = resnet18(weights=None if not pretrained else "IMAGENET1K_V1")
        # 调整第一层通道数
        if self.backbone.conv1.in_channels != c:
            self.backbone.conv1 = nn.Conv2d(c, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # 拆掉最终分类头，保留全局池化后的 512 维向量
        in_feat = self.backbone.fc.in_features  # 一般是 512
        self.backbone.fc = nn.Identity()

        # 投影到期望的 features_dim（共享给 actor/critic）
        #self.proj = nn.Linear(in_feat, features_dim)

        # 可选：做简单的归一化层（根据你数据的范围调整）
        self.register_buffer("_mean", th.tensor(0.0), persistent=False)
        self.register_buffer("_std", th.tensor(1.0), persistent=False)

    def forward(self, obs: dict) -> th.Tensor:
        x = obs["observation"].float()  # (B, H, W) 或 (B, C, H, W)
        # 统一成 (B, C, H, W)
        if x.ndim == 3:             # (B, H, W)
            x = x.unsqueeze(-1)      # -> (B, H, W, 1)
            x = ResNet18Extractor._resize_batch(x)

        # 如有需要，可在此做归一化（你的数据若已在[0,1]可不处理）
        # x = (x - self._mean) / (self._std + 1e-8)

        z = self.backbone(x)        # (B, 512)
        #z = self.proj(z)            # (B, features_dim)
        return z
    
    @staticmethod
    def _nearest_idx(in_size, out_size):
        # PyTorch (align_corners=False) 映射：in = (out+0.5)*(in/out) - 0.5，再取最近整数
        scale = in_size / out_size
        out_coords = np.arange(out_size, dtype=np.float64)
        in_coords = (out_coords + 0.5) * scale - 0.5
        idx = np.clip(np.round(in_coords), 0, in_size - 1).astype(np.int64)
        return idx

    @staticmethod
    def _resize_batch(x, size = (224, 65)):
        """x: (..., H, W) 任意前置批与通道维；size=(new_h, new_w)"""
        new_h, new_w = size
        H, W = x.shape[-2:]
        iy = ResNet18Extractor._nearest_idx(H, new_h)              # (new_h,)
        ix = ResNet18Extractor._nearest_idx(W, new_w)              # (new_w,)
        # 利用广播采样
        return x[..., iy[:, None], ix[None, :]]


def visualize_resnet1d(input_shape, **extractor_kwargs):
    """
    Print per-layer tensor shapes for ResNet1DExtractor given an input shape.
    input_shape: (channels, length) or length-only tuple/int.
    extractor_kwargs: forwarded to ResNet1DExtractor constructor.
    """
    if isinstance(input_shape, int):
        input_shape = (1, int(input_shape))
    elif len(input_shape) == 1:
        input_shape = (1, int(input_shape[0]))
    elif len(input_shape) != 2:
        raise ValueError(f"Expected input_shape (C, L) or (L,), got {input_shape}")

    obs_space = spaces.Box(low=-1.0, high=1.0, shape=input_shape, dtype=np.float32)
    extractor = ResNet1DExtractor(observation_space=obs_space, **extractor_kwargs).eval()

    hooks = []
    records = []

    def _to_shape(obj):
        if isinstance(obj, th.Tensor):
            return tuple(obj.shape)
        if isinstance(obj, (list, tuple)):
            return [ _to_shape(item) for item in obj ]
        return type(obj).__name__

    def _make_hook(name, param_count):
        def hook(module, inputs, outputs):
            in_shape = _to_shape(inputs[0] if len(inputs) == 1 else inputs)
            out_shape = _to_shape(outputs)
            records.append((name, in_shape, out_shape, param_count))
        return hook

    layer_params = []
    for name, module in extractor.named_modules():
        if not name:
            continue
        if any(module.children()):
            continue
        params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        layer_params.append((name, params))
        hooks.append(module.register_forward_hook(_make_hook(name, params)))

    with th.no_grad():
        dummy = th.zeros((1,) + input_shape)
        extractor(dummy)

    for hook in hooks:
        hook.remove()

    header = f"{'Layer':30} {'Input shape':20} {'Output shape':20} {'# Params':10}"
    print(header)
    print("-" * len(header))
    for name, in_shape, out_shape, param_count in records:
        print(f"{name:30} {str(in_shape):20} {str(out_shape):20} {param_count:10}")
    print(f"Total trainable parameters = {sum(x[-1] for x in records)}")

    return records


def visualize_resnet18(input_shape, batch_size: int = 1, **extractor_kwargs):
    """
    Print per-layer tensor shapes for ResNet18Extractor given an input shape.
    input_shape: (channels, height, width) or (height, width).
    extractor_kwargs: forwarded to ResNet18Extractor constructor.
    """
    if isinstance(input_shape, int):
        input_shape = (1, int(input_shape), int(input_shape))
    elif len(input_shape) == 2:
        input_shape = (1, int(input_shape[0]), int(input_shape[1]))
    elif len(input_shape) == 3:
        input_shape = tuple(int(dim) for dim in input_shape)
    else:
        raise ValueError(f"Expected input_shape (C,H,W) or (H,W), got {input_shape}")

    obs_space = spaces.Dict({
        "observation": spaces.Box(low=0.0, high=1.0, shape=input_shape, dtype=np.float32)
    })
    extractor = ResNet18Extractor(observation_space=obs_space, **extractor_kwargs).eval()

    hooks = []
    records = []

    def _to_shape(obj):
        if isinstance(obj, th.Tensor):
            return tuple(obj.shape)
        if isinstance(obj, (list, tuple)):
            return [_to_shape(item) for item in obj]
        return type(obj).__name__

    def _make_hook(name):
        def hook(module, inputs, outputs):
            in_shape = _to_shape(inputs[0] if len(inputs) == 1 else inputs)
            out_shape = _to_shape(outputs)
            records.append((name, in_shape, out_shape))
        return hook

    for name, module in extractor.named_modules():
        if not name:
            continue
        if any(module.children()):
            continue
        hooks.append(module.register_forward_hook(_make_hook(name)))

    with th.no_grad():
        dummy = th.zeros((batch_size,) + input_shape, dtype=th.float32)
        extractor({"observation": dummy})

    for hook in hooks:
        hook.remove()

    header = f"{'Layer':30} {'Input shape':20} {'Output shape':20}"
    print(header)
    print("-" * len(header))
    for name, in_shape, out_shape in records:
        print(f"{name:30} {str(in_shape):20} {str(out_shape):20}")

    return records


def print_resnet1d_output_stats(input_shape=(163, 34), batch_size: int = 32, **extractor_kwargs):
    """
    Instantiate ResNet1DExtractor, push a random batch through it, and print mean/variance stats.
    """
    if isinstance(input_shape, int):
        input_shape = (1, int(input_shape))
    elif len(input_shape) == 1:
        input_shape = (1, int(input_shape[0]))
    elif len(input_shape) != 2:
        raise ValueError(f"Expected input_shape (C, L) or (L,), got {input_shape}")

    obs_space = spaces.Box(low=-1.0, high=1.0, shape=input_shape, dtype=np.float32)
    extractor = ResNet1DExtractor(observation_space=obs_space, **extractor_kwargs).eval()

    with th.no_grad():
        sample = th.randn(batch_size, extractor.in_ch, extractor.length, dtype=th.float32)
        if extractor.channels_last:
            sample = sample.transpose(1, 2).contiguous()

        output = extractor(sample)
        mean = output.mean().item()
        var = output.var(unbiased=False).item()

    print(f"ResNet1D output mean: {mean:.6f}")
    print(f"ResNet1D output variance: {var:.6f}")
    return mean, var

if __name__ == "__main__":
    visualize_resnet1d((163, 34))
    #visualize_resnet18((163, 224, 65))
    print_resnet1d_output_stats()
