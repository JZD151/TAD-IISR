import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class AsymmetricConvBranch(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.conv_h = nn.Conv2d(dim, dim, kernel_size=(1, 7), padding=(0, 3), groups=dim)
        self.conv_w = nn.Conv2d(dim, dim, kernel_size=(7, 1), padding=(3, 0), groups=dim)
        self.act = nn.GELU()
        self.bn = LayerNorm(dim, data_format="channels_first")

    def forward(self, x):
        feat_h = self.conv_h(x)
        feat_w = self.conv_w(x)
        return self.bn(self.act(feat_h + feat_w))

class LargeKernelBranch(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.large_conv = nn.Conv2d(dim, dim, kernel_size=5, padding=6, dilation=3, groups=dim)
        self.point_conv = nn.Conv2d(dim, dim, 1)
        self.act = nn.GELU()
        self.bn = LayerNorm(dim, data_format="channels_first")

    def forward(self, x):
        out = self.large_conv(x)
        out = self.point_conv(out)
        return self.bn(self.act(out))

class SKFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        reduction_dim = max(dim // 4, 16)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, reduction_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduction_dim, 2 * dim, bias=False)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2):
        batch_size = x1.size(0)
        
        feat_sum = x1 + x2
        s = self.avg_pool(feat_sum).view(batch_size, -1)
        
        z = self.fc(s)
        z = z.view(batch_size, 2, self.dim)
        attn = self.softmax(z)
        
        out = x1 * attn[:, 0].view(batch_size, self.dim, 1, 1) + \
              x2 * attn[:, 1].view(batch_size, self.dim, 1, 1)
        return out

class PhysicsDisentangledBlock(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.stripe_branch = AsymmetricConvBranch(dim)
        self.turb_branch = LargeKernelBranch(dim)
        self.fusion = SKFusion(dim)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        res = x
        x_stripe = self.stripe_branch(x)
        x_turb = self.turb_branch(x)
        
        x_out = self.fusion(x_stripe, x_turb)
        return res + self.proj(x_out)

class DownsampleBlock(nn.Module):
    
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.down = nn.Sequential(
            LayerNorm(in_dim, data_format="channels_first"),
            nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2)
        )
    def forward(self, x):
        return self.down(x)

class idem(nn.Module):
    
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        
        dims = [64, 128, 256]
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=3, padding=1),
            LayerNorm(dims[0], data_format="channels_first")
        )
        
        self.stage1 = nn.Sequential(
            PhysicsDisentangledBlock(dims[0]),
            DownsampleBlock(dims[0], dims[1])
        )
        
        self.stage2 = nn.Sequential(
            PhysicsDisentangledBlock(dims[1]),
            PhysicsDisentangledBlock(dims[1]),
            DownsampleBlock(dims[1], dims[2])
        )
        
        self.stage3 = nn.Sequential(
            PhysicsDisentangledBlock(dims[2]),
            PhysicsDisentangledBlock(dims[2]),
            PhysicsDisentangledBlock(dims[2]),
            DownsampleBlock(dims[2], out_channels)
        )
        
        self.final_process = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.final_process(x)
        return x