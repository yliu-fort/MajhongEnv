import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# --------- 1D ResNet 基本残差块 ----------
class BasicBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, groups=8, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1   = nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch)
        self.act   = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2   = nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch)
        self.drop  = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # 如果通道或步幅不一致，用 1x1 投影做 downsample
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch),
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
        out = self.act(out)
        return out

# --------- 1D ResNet 提取器（支持 Box 或 Dict['observation']） ----------
class ResNet1DExtractor(BaseFeaturesExtractor):
    """
    1D ResNet 风格特征提取器（输入形状: (B, C, H)）。
    - 默认 4 个 stage、共 6 个残差块（12 个卷积层 + stem）→ 轻量稳定。
    - 使用 GroupNorm，适合 RL 小 batch。
    """
    def __init__(
        self,
        observation_space: spaces.Space,
        features_dim: int = 256,
        base_channels: int = 64,
        blocks_per_stage=(2, 2, 1, 1),  # 每个 stage 的残差块数：总卷积层数=2*sum(blocks)
        channels_per_stage=None,        # 若不指定，按 [64, 128, 128, 256] * (base_channels/64)
        stem_stride: int = 2,
        use_maxpool: bool = False,
        dropout: float = 0.0,
        groups: int = 8,
    ):
        super().__init__(observation_space, features_dim)

        # ---- 解析输入空间：支持 Dict['observation'] 或直接 Box ----
        if isinstance(observation_space, spaces.Dict):
            obs_space = observation_space["observation"]
        else:
            obs_space = observation_space
        assert isinstance(obs_space, spaces.Box), "ResNet1DExtractor 需要 Box 或 Dict[Box] 观测"
        assert len(obs_space.shape) == 2 or len(obs_space.shape) == 3, \
            f"期望形状 (C,H) 或 (B,C,H) 中的 (C,H)，收到 {obs_space.shape}"

        # 推断通道/长度
        if len(obs_space.shape) == 2:
            in_ch, length = int(obs_space.shape[0]), int(obs_space.shape[1])
        else:
            # SB3 传入的是每步单样本空间，因此这里通常不会出现 (B, C, H)
            in_ch, length = int(obs_space.shape[-2]), int(obs_space.shape[-1])

        # 默认每个 stage 的通道
        if channels_per_stage is None:
            scale = base_channels // 64
            channels_per_stage = [64*scale, 128*scale, 128*scale, 256*scale]

        # ---- stem ----
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, channels_per_stage[0], kernel_size=7, stride=stem_stride, padding=3, bias=False),
            nn.GroupNorm(num_groups=min(groups, channels_per_stage[0]), num_channels=channels_per_stage[0]),
            nn.SiLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1) if use_maxpool else nn.Identity(),
        )

        # ---- stages ----
        stages = []
        ch_in = channels_per_stage[0]
        for i, (num_blocks, ch_out) in enumerate(zip(blocks_per_stage, channels_per_stage)):
            for b in range(num_blocks):
                stride = 2 if (b == 0 and i > 0) else 1  # 除第一组外，每组首块下采样
                stages.append(BasicBlock1D(ch_in, ch_out, stride=1, groups=groups, dropout=dropout))
                ch_in = ch_out
        self.backbone = nn.Sequential(*stages)

        # ---- 头部：全局池化 + 投影到 features_dim ----
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(ch_in, features_dim)

        # 记录输出维度（SB3 需要）
        self._features_dim = features_dim

    def forward(self, obs):
        """
        obs: tensor 或 dict，若为 dict 则使用 obs['observation']，形状必须是 (B,C,H)
        """
        x = obs["observation"] if isinstance(obs, dict) else obs
        x = x.float()
        # 若输入是 (B,H) 或 (B,H,C) 之类，尝试整理为 (B,C,H)
        if x.ndim == 2:  # (B, H) -> (B, 1, H)
            x = x.unsqueeze(1)
        elif x.ndim == 3 and x.shape[2] > x.shape[1]:
            # 可能传成 (B, H, C)，转为 (B, C, H)
            x = x.transpose(1, 2).contiguous()

        z = self.stem(x)
        z = self.backbone(z)
        z = self.pool(z).squeeze(-1)   # (B, C_out)
        z = self.proj(z)               # (B, features_dim)
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
        elif x.ndim == 4 and x.shape[1] not in (1, 3):  # 可能是 (B, H, W, C)
            x = x.permute(0, 3, 1, 2).contiguous()

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