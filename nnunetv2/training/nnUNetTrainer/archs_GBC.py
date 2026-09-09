# AD-GBC + Rolling-UNet
import torch
import torch.nn.functional as F
try:
    from .GBC_utils import *
except (ImportError, ValueError):
    from GBC_utils import *
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math

__all__ = ['GBC_Rolling_Unet_S', 'GBC_Rolling_Unet_M', 'GBC_Rolling_Unet_L', 'Rolling_Unet_L']


class GranularBall(nn.Module):
    def __init__(self, in_ch, num_balls=32, proj_dim=None, use_residual=True, use_diag_cov=True, tau=1.0):
        super().__init__()
        self.in_ch = in_ch
        self.proj_dim = proj_dim or in_ch
        self.num_balls = num_balls
        self.use_residual = use_residual
        self.use_diag_cov = use_diag_cov
        self.tau = tau

        self.centers = nn.Parameter(torch.randn(num_balls, self.proj_dim) * 0.01)
        # ★ Added: Radius/Diagonal covariance (softplus positive)
        # diag_cov 쓰면 k개 ball마다 proj_dim 개의 radii 사용 (num_balls, proj_dim) => Anisotropic ball
        # diag_cov안쓰면 k개 ball마다 중심 하나로 퉁쳐 (num_balls, 1)                 => Hypersphere
        if use_diag_cov:
            self.log_sigma = nn.Parameter(torch.zeros(num_balls, self.proj_dim))  # diag std = Anisotropic
        else:
            self.log_radius = nn.Parameter(torch.zeros(num_balls, 1))  # scalar std = Hypersphere

        if self.proj_dim != in_ch:
            self.proj_in = nn.Conv2d(in_ch, self.proj_dim, 1, bias=True)
            self.bn_in = nn.BatchNorm2d(self.proj_dim)
            self.proj_out = nn.Conv2d(self.proj_dim, in_ch, 1, bias=False)
            self.bn_out = nn.BatchNorm2d(in_ch)
        else:
            self.proj_in = self.proj_out = None

        self.refine = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, tau=1.0):
        B, C, H, W = x.shape
        z = self.bn_in(self.proj_in(x)) if self.proj_in is not None else x  # 차원 안 맞으면 1x1Conv
        d = z.shape[1]  # 1x1Conv한 후에 Dim = d(안했으면 C겠지)
        z_flat = z.view(B, d, H * W).permute(0, 2, 1)  # (B,N,d)

        # 논문에서 Forward 부분 정리한 대로 따라감.
        # 차원 다르면 1x1 Conv해서 차원 맞춘 input feature map z flatten(BxNxD)
        # d_{i,k} = ||(z_i-c_k) [element-wise div] sigma||^2_2
        # c : GB anchor center, sigma : GB anchor radii
        dif = z_flat.unsqueeze(2) - self.centers.unsqueeze(0).unsqueeze(0)  # (B,N,K,d)

        # Softplus = log(1+exp(x))
        # sigma>0 보장하고 미분 편하게.
        if self.use_diag_cov:
            sigma = (F.softplus(self.log_sigma) + 1e-6).unsqueeze(0).unsqueeze(0)  # (1,1,K,d)
        else:
            sigma = (F.softplus(self.log_radius) + 1e-6).unsqueeze(0).unsqueeze(0)  # (1,1,K,1)
        dif_scaled = dif / sigma  # Broadcasted
        dist2 = (dif_scaled ** 2).sum(-1)  # (B,N,K)

        # 논문에서는 alpha_{i,k}
        # Ball k에 소속될 pixel z_i의 soft weight.
        att = F.softmax(-dist2 / max(1e-6, tau), dim=-1)  # Soft membership = Fuzzy assignment weights

        # alpha center(att된 z들)가 논문에서는 Broadcast(Ball -> Set)
        # 근데 왜 Aggregation 부분 없냐고 십탱
        # @논문 :
        # Aggregation : c'_k = \sum_{i=1}^N  alpha_{i,k} z_i        N은 z(픽셀)갯수
        #    (BxKxN)x(BxNxd) = (BxKxd)
        #           여기서는 alpha_{i,k}가 z_i에 곱.
        # Broadcast   : hat{z_i} = \sum_{k=1}^K alpha_{i,k} c'_k    K는 ball 갯수
        #    (BxNxK)x(BxKxd) = (BxNxd)
        #           여기선 alpha_{i,k}가 c'_k에 곱 - Agg 부분은 사실 alpha.T다.
        # 총 두 번 곱해져야 함 - 모든 픽셀, 모든 ball 따라 hat{z_i} = (alpha) x (alpha.T) x (z)
        # 이거 Unflatten하고 채널맞추기 1x1Conv(Optional)
        # 근데 왜 코드에서는 그대로 zeros 파라미터 centers에 한 번만 matmul하지?
        # self.centers, Sigma는 Param이니 ball 모양 찾아가며 계속 업데이트. 그거로 최종 Feature Y = f_refine (X+hat{X})만 목표

        # 논문 방향 = 데이터들의 특징점들이 모여 구(Anisotropic)를 형성한다 -> 걔네를 cluster한다 -> 각 클러스터 이용해 feature refine
        # 실제 구현 = 데이터들 보다보면 feature space 마다 클러스터 보이겠지 ->               -> 각 클러스터 이용해 feature refine
        # 그러니까 일단 zeros 파라미터로 클러스터 뿌려놓고 역전파로 업데이트해가면서 찾아가는 과정

        # ★★정리★★
        # Aggregation : Pixel feature에서 Cluster 만들기
        # Broadcast   : Cluster로 pixel정보 강화
        # In the wild에서는 이미 데이터들 보다보면 클러스터 돼있을거라 가정, 굳이 픽셀들 이용해서 cluster 구하지 않는다(걔네도 몰라)
        # 클러스터 있을거라 가정하고 Parameters로 뿌려놓고 걔네 학습해가며 사용.
        # => 실제 구현에서는 굳이 Aggregation 필요 없다. 추정해놓은 클러스터에서 Broadcasting만 사용
        recon_flat = torch.matmul(att, self.centers)  # (B,N,d)
        recon = recon_flat.permute(0, 2, 1).view(B, d, H, W)

        if self.proj_out is not None:
            recon = self.bn_out(self.proj_out(recon))

        out = recon + x if self.use_residual else recon
        return self.refine(out), att, sigma, dif


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)


class Lo2(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., shift_size=5):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(in_features, hidden_features)
        self.fc3 = nn.Linear(in_features, hidden_features)
        self.fc4 = nn.Linear(in_features, hidden_features)
        self.fc5 = nn.Linear(in_features * 2, hidden_features)
        self.fc6 = nn.Linear(hidden_features * 2, out_features)
        self.drop = nn.Dropout(drop)
        self.dwconv = DWConv(hidden_features)
        self.act1 = act_layer()
        self.act2 = nn.ReLU()
        self.norm1 = nn.LayerNorm(hidden_features * 2)
        self.norm2 = nn.BatchNorm2d(hidden_features)
        self.shift_size = shift_size
        self.pad = shift_size // 2
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape

        ### DOR-MLP
        ### OR-MLP
        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xs = torch.chunk(xn, C, 1)
        x_shift = [torch.roll(x_c, shift, 2) for x_c, shift in zip(xs, range(0, C))]
        x_cat = torch.cat(x_shift, 1)
        x_s = x_cat.reshape(B, C, H * W).contiguous()
        x_shift_r = x_s.transpose(1, 2)
        x_shift_r = self.fc1(x_shift_r)
        x_shift_r = self.act1(x_shift_r)
        x_shift_r = self.drop(x_shift_r)
        xn = x_shift_r.transpose(1, 2).view(B, C, H, W).contiguous()
        xs = torch.chunk(xn, C, 1)
        x_shift = [torch.roll(x_c, shift, 3) for x_c, shift in zip(xs, range(0, C))]
        x_cat = torch.cat(x_shift, 1)
        x_s = x_cat.reshape(B, C, H * W).contiguous()
        x_shift_c = x_s.transpose(1, 2)
        x_shift_c = self.fc2(x_shift_c)
        x_1 = self.drop(x_shift_c)

        ### OR-MLP
        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xs = torch.chunk(xn, C, 1)
        x_shift = [torch.roll(x_c, -shift, 3) for x_c, shift in zip(xs, range(0, C))]
        x_cat = torch.cat(x_shift, 1)
        x_s = x_cat.reshape(B, C, H * W).contiguous()
        x_shift_c = x_s.transpose(1, 2)
        x_shift_c = self.fc3(x_shift_c)
        x_shift_c = self.act1(x_shift_c)
        x_shift_c = self.drop(x_shift_c)
        xn = x_shift_c.transpose(1, 2).view(B, C, H, W).contiguous()
        xs = torch.chunk(xn, C, 1)
        x_shift = [torch.roll(x_c, shift, 2) for x_c, shift in zip(xs, range(0, C))]
        x_cat = torch.cat(x_shift, 1)
        x_s = x_cat.reshape(B, C, H * W).contiguous()
        x_shift_r = x_s.transpose(1, 2)
        x_shift_r = self.fc4(x_shift_r)
        x_2 = self.drop(x_shift_r)

        x_1 = torch.add(x_1, x)
        x_2 = torch.add(x_2, x)
        x1 = torch.cat([x_1, x_2], dim=2)
        x1 = self.norm1(x1)
        x1 = self.fc5(x1)
        x1 = self.drop(x1)
        x1 = torch.add(x1, x)
        x2 = x.transpose(1, 2).view(B, C, H, W)

        ### DSC
        x2 = self.dwconv(x2, H, W)
        x2 = self.act2(x2)
        x2 = self.norm2(x2)
        x2 = x2.flatten(2).transpose(1, 2)

        x3 = torch.cat([x1, x2], dim=2)
        x3 = self.fc6(x3)
        x3 = self.drop(x3)
        return x3


class Lo2Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Lo2(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.drop_path(self.mlp(x, H, W))
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.point_conv = nn.Conv2d(dim, dim, 1, 1, 0, bias=True, groups=1)

    def forward(self, x, H, W):
        x = self.dwconv(x)
        x = self.point_conv(x)
        return x


class Feature_Incentive_Block(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)
        self.act = nn.GELU()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.act(x)
        x = self.norm(x)
        return x, H, W


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class D_DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(D_DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class GBC_Rolling_Unet_S(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, img_size=224,
                 embed_dims=[16, 32, 64, 128, 256],
                 num_heads=[1, 2, 4, 8], qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1],
                 gbc_num_balls=32, gbc_proj_dim=None, use_diag_cov=True, tau=1.0, **kwargs):
        super().__init__()

        self.embed_dims = embed_dims
        self.conv1 = DoubleConv(input_channels, embed_dims[0])
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(embed_dims[0], embed_dims[1])
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(embed_dims[1], embed_dims[2])
        self.pool3 = nn.MaxPool2d(2)

        self.pool4 = nn.MaxPool2d(2)

        self.FIBlock1 = Feature_Incentive_Block(img_size=img_size // 4, patch_size=3, stride=1,
                                                in_chans=embed_dims[2],
                                                embed_dim=embed_dims[3])
        self.FIBlock2 = Feature_Incentive_Block(img_size=img_size // 8, patch_size=3, stride=1,
                                                in_chans=embed_dims[3],
                                                embed_dim=embed_dims[4])
        self.FIBlock3 = Feature_Incentive_Block(img_size=img_size // 8, patch_size=3, stride=1,
                                                in_chans=embed_dims[4],
                                                embed_dim=embed_dims[3])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.block1 = nn.ModuleList([Lo2Block(
            dim=embed_dims[3], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])
        self.block2 = nn.ModuleList([Lo2Block(
            dim=embed_dims[4], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate + 0.1, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])
        self.block3 = nn.ModuleList([Lo2Block(
            dim=embed_dims[3], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.norm1 = norm_layer(embed_dims[3])
        self.norm2 = norm_layer(embed_dims[4])
        self.norm3 = norm_layer(embed_dims[3])

        self.FIBlock4 = nn.Conv2d(embed_dims[3], embed_dims[2], 3, stride=1, padding=1)
        self.dbn4 = nn.BatchNorm2d(embed_dims[2])

        self.decoder3 = D_DoubleConv(embed_dims[2], embed_dims[1])
        self.decoder2 = D_DoubleConv(embed_dims[1], embed_dims[0])
        self.decoder1 = D_DoubleConv(embed_dims[0], 8)

        self.final = nn.Conv2d(8, num_classes, kernel_size=1)

        proj_dim_actual = gbc_proj_dim if gbc_proj_dim and gbc_proj_dim > 0 else embed_dims[2]
        self.gbc = GranularBall(in_ch=embed_dims[2], num_balls=gbc_num_balls, proj_dim=proj_dim_actual,
                                use_diag_cov=use_diag_cov, use_residual=True, tau=tau)

    def forward(self, x):
        B = x.shape[0]

        ### Conv Stage
        out = self.conv1(x)
        t1 = out
        out = self.pool1(out)
        out = self.conv2(out)
        t2 = out
        out = self.pool2(out)
        out = self.conv3(out)
        t3 = out

        t3_gbc, att_t3, _, dif_t3 = self.gbc(t3)
        out = self.pool3(t3_gbc)

        ### Stage 4
        out, H, W = self.FIBlock1(out)
        for i, blk in enumerate(self.block1):
            out = blk(out, H, W)
        out = self.norm1(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out

        out = self.pool4(out)

        ### Bottleneck
        out, H, W = self.FIBlock2(out)
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)
        out = self.norm2(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out, H, W = self.FIBlock3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.interpolate(out, scale_factor=(2, 2), mode='bilinear')

        ### Stage 4
        out = torch.add(out, t4)
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.block3):
            out = blk(out, H * 2, W * 2)
        out = self.norm3(out)
        out = out.reshape(B, H * 2, W * 2, -1).permute(0, 3, 1, 2).contiguous()
        out = F.interpolate(F.relu(self.dbn4(self.FIBlock4(out))), scale_factor=(2, 2), mode='bilinear')

        ### Conv Stage
        # out = torch.add(out, t3)
        out, att_out, _, dif_out = self.gbc(out)
        out = torch.add(out, t3_gbc)  # 与经过GBC处理的t3进行跳跃连接

        out = F.interpolate(self.decoder3(out), scale_factor=(2, 2), mode='bilinear')
        out = torch.add(out, t2)
        out = F.interpolate(self.decoder2(out), scale_factor=(2, 2), mode='bilinear')
        out = torch.add(out, t1)
        out = self.decoder1(out)

        out = self.final(out)

        if self.training:
            # 将所有计算loss所需的中间变量打包
            loss_intermediates = {
                "att_1": att_t3, "dif_1": dif_t3,
                "att_2": att_out, "dif_2": dif_out
            }
            return out, loss_intermediates
        else:
            # 在评估/推理时，只返回分割结果
            return out

        return out


class GBC_Rolling_Unet_M(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, img_size=224,
                 embed_dims=[32, 64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1],
                 gbc_num_balls=32, gbc_proj_dim=None, use_diag_cov=True, tau=1.0, **kwargs):
        super().__init__()

        self.embed_dims = embed_dims
        self.conv1 = DoubleConv(input_channels, embed_dims[0])
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(embed_dims[0], embed_dims[1])
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(embed_dims[1], embed_dims[2])
        self.pool3 = nn.MaxPool2d(2)

        self.pool4 = nn.MaxPool2d(2)

        self.FIBlock1 = Feature_Incentive_Block(img_size=img_size // 4, patch_size=3, stride=1,
                                                in_chans=embed_dims[2],
                                                embed_dim=embed_dims[3])
        self.FIBlock2 = Feature_Incentive_Block(img_size=img_size // 8, patch_size=3, stride=1,
                                                in_chans=embed_dims[3],
                                                embed_dim=embed_dims[4])
        self.FIBlock3 = Feature_Incentive_Block(img_size=img_size // 8, patch_size=3, stride=1,
                                                in_chans=embed_dims[4],
                                                embed_dim=embed_dims[3])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.block1 = nn.ModuleList([Lo2Block(
            dim=embed_dims[3], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])
        self.block2 = nn.ModuleList([Lo2Block(
            dim=embed_dims[4], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate + 0.2, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])
        self.block3 = nn.ModuleList([Lo2Block(
            dim=embed_dims[3], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.norm1 = norm_layer(embed_dims[3])
        self.norm2 = norm_layer(embed_dims[4])
        self.norm3 = norm_layer(embed_dims[3])

        self.FIBlock4 = nn.Conv2d(embed_dims[3], embed_dims[2], 3, stride=1, padding=1)
        self.dbn4 = nn.BatchNorm2d(embed_dims[2])

        self.decoder3 = D_DoubleConv(embed_dims[2], embed_dims[1])
        self.decoder2 = D_DoubleConv(embed_dims[1], embed_dims[0])
        self.decoder1 = D_DoubleConv(embed_dims[0], 16)

        self.final = nn.Conv2d(16, num_classes, kernel_size=1)

        proj_dim_actual = gbc_proj_dim if gbc_proj_dim and gbc_proj_dim > 0 else embed_dims[2]
        self.gbc = GranularBall(in_ch=embed_dims[2], num_balls=gbc_num_balls, proj_dim=proj_dim_actual,
                                use_diag_cov=use_diag_cov, use_residual=True, tau=tau)

    def forward(self, x):
        B = x.shape[0]

        ### Conv Stage
        out = self.conv1(x)
        t1 = out
        out = self.pool1(out)
        out = self.conv2(out)
        t2 = out
        out = self.pool2(out)
        out = self.conv3(out)
        t3 = out
        # out = self.pool3(out)
        t3_gbc, att_t3, _, dif_t3 = self.gbc(t3)
        out = self.pool3(t3_gbc)

        ### Stage 4
        out, H, W = self.FIBlock1(out)
        for i, blk in enumerate(self.block1):
            out = blk(out, H, W)
        out = self.norm1(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out
        out = self.pool4(out)

        ### Bottleneck
        out, H, W = self.FIBlock2(out)
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)
        out = self.norm2(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out, H, W = self.FIBlock3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.interpolate(out, scale_factor=(2, 2), mode='bilinear')

        ### Stage 4
        out = torch.add(out, t4)
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.block3):
            out = blk(out, H * 2, W * 2)
        out = self.norm3(out)
        out = out.reshape(B, H * 2, W * 2, -1).permute(0, 3, 1, 2).contiguous()
        out = F.interpolate(F.relu(self.dbn4(self.FIBlock4(out))), scale_factor=(2, 2), mode='bilinear')

        ### Conv Stage
        # out = torch.add(out, t3)
        out, att_out, _, dif_out = self.gbc(out)
        out = torch.add(out, t3_gbc)  # 与经过GBC处理的t3进行跳跃连接

        out = F.interpolate(self.decoder3(out), scale_factor=(2, 2), mode='bilinear')
        out = torch.add(out, t2)
        out = F.interpolate(self.decoder2(out), scale_factor=(2, 2), mode='bilinear')
        out = torch.add(out, t1)
        out = self.decoder1(out)

        out = self.final(out)

        if self.training:
            # 将所有计算loss所需的中间变量打包
            loss_intermediates = {
                "att_1": att_t3, "dif_1": dif_t3,
                "att_2": att_out, "dif_2": dif_out
            }
            return out, loss_intermediates
        else:
            # 在评估/推理时，只返回分割结果
            return out

        return out


class GBC_Rolling_Unet_L(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, img_size=224,
                 embed_dims=[64, 128, 256, 512, 1024],
                 num_heads=[1, 2, 4, 8], qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1],
                 gbc_num_balls=32, gbc_proj_dim=None, use_diag_cov=True, tau=1.0, **kwargs):
        super().__init__()

        self.embed_dims = embed_dims
        self.conv1 = DoubleConv(input_channels, embed_dims[0])
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(embed_dims[0], embed_dims[1])
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(embed_dims[1], embed_dims[2])
        self.pool3 = nn.MaxPool2d(2)

        self.pool4 = nn.MaxPool2d(2)

        self.FIBlock1 = Feature_Incentive_Block(img_size=img_size // 4, patch_size=3, stride=1,
                                                in_chans=embed_dims[2],
                                                embed_dim=embed_dims[3])
        self.FIBlock2 = Feature_Incentive_Block(img_size=img_size // 8, patch_size=3, stride=1,
                                                in_chans=embed_dims[3],
                                                embed_dim=embed_dims[4])
        self.FIBlock3 = Feature_Incentive_Block(img_size=img_size // 8, patch_size=3, stride=1,
                                                in_chans=embed_dims[4],
                                                embed_dim=embed_dims[3])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.block1 = nn.ModuleList([Lo2Block(
            dim=embed_dims[3], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])
        self.block2 = nn.ModuleList([Lo2Block(
            dim=embed_dims[4], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate + 0.3, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])
        self.block3 = nn.ModuleList([Lo2Block(
            dim=embed_dims[3], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.norm1 = norm_layer(embed_dims[3])
        self.norm2 = norm_layer(embed_dims[4])
        self.norm3 = norm_layer(embed_dims[3])

        self.FIBlock4 = nn.Conv2d(embed_dims[3], embed_dims[2], 3, stride=1, padding=1)
        self.dbn4 = nn.BatchNorm2d(embed_dims[2])

        self.decoder3 = D_DoubleConv(embed_dims[2], embed_dims[1])
        self.decoder2 = D_DoubleConv(embed_dims[1], embed_dims[0])
        self.decoder1 = D_DoubleConv(embed_dims[0], 32)

        self.final = nn.Conv2d(32, num_classes, kernel_size=1)

        proj_dim_actual = gbc_proj_dim if gbc_proj_dim and gbc_proj_dim > 0 else embed_dims[2]
        self.gbc = GranularBall(in_ch=embed_dims[2], num_balls=gbc_num_balls, proj_dim=proj_dim_actual,
                                use_diag_cov=use_diag_cov, use_residual=True, tau=tau)

    def forward(self, x):
        B = x.shape[0]

        ### Conv Stage
        out = self.conv1(x)
        t1 = out
        out = self.pool1(out)
        out = self.conv2(out)
        t2 = out
        out = self.pool2(out)
        out = self.conv3(out)
        t3 = out
        # out = self.pool3(out)
        t3_gbc, att_t3, _, dif_t3 = self.gbc(t3)
        out = self.pool3(t3_gbc)

        ### Stage 4
        out, H, W = self.FIBlock1(out)
        for i, blk in enumerate(self.block1):
            out = blk(out, H, W)
        out = self.norm1(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out
        out = self.pool4(out)

        ### Bottleneck
        out, H, W = self.FIBlock2(out)
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)
        out = self.norm2(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out, H, W = self.FIBlock3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.interpolate(out, scale_factor=(2, 2), mode='bilinear')

        ### Stage 4
        out = torch.add(out, t4)
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.block3):
            out = blk(out, H * 2, W * 2)
        out = self.norm3(out)
        out = out.reshape(B, H * 2, W * 2, -1).permute(0, 3, 1, 2).contiguous()
        out = F.interpolate(F.relu(self.dbn4(self.FIBlock4(out))), scale_factor=(2, 2), mode='bilinear')

        ### Conv Stage
        # out = torch.add(out, t3)
        out, att_out, _, dif_out = self.gbc(out)
        out = torch.add(out, t3_gbc)  # 与经过GBC处理的t3进行跳跃连接
        out = F.interpolate(self.decoder3(out), scale_factor=(2, 2), mode='bilinear')
        out = torch.add(out, t2)
        out = F.interpolate(self.decoder2(out), scale_factor=(2, 2), mode='bilinear')
        out = torch.add(out, t1)
        out = self.decoder1(out)

        out = self.final(out)

        if self.training:
            # 将所有计算loss所需的中间变量打包
            loss_intermediates = {
                "att_1": att_t3, "dif_1": dif_t3,
                "att_2": att_out, "dif_2": dif_out
            }
            return out, loss_intermediates
        else:
            # 在评估/推理时，只返回分割结果
            return out

        return out

class Rolling_Unet_L(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, img_size=224,
                 embed_dims=[64, 128, 256, 512, 1024],
                 num_heads=[1, 2, 4, 8], qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1], **kwargs):
        super().__init__()

        self.conv1 = DoubleConv(input_channels, embed_dims[0])
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(embed_dims[0], embed_dims[1])
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(embed_dims[1], embed_dims[2])
        self.pool3 = nn.MaxPool2d(2)

        self.pool4 = nn.MaxPool2d(2)

        self.FIBlock1 = Feature_Incentive_Block(img_size=img_size // 4, patch_size=3, stride=1,
                                                    in_chans=embed_dims[2],
                                                    embed_dim=embed_dims[3])
        self.FIBlock2 = Feature_Incentive_Block(img_size=img_size // 8, patch_size=3, stride=1,
                                                    in_chans=embed_dims[3],
                                                    embed_dim=embed_dims[4])
        self.FIBlock3 = Feature_Incentive_Block(img_size=img_size // 8, patch_size=3, stride=1,
                                                    in_chans=embed_dims[4],
                                                    embed_dim=embed_dims[3])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.block1 = nn.ModuleList([Lo2Block(
            dim=embed_dims[3], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])
        self.block2 = nn.ModuleList([Lo2Block(
            dim=embed_dims[4], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate + 0.3, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])
        self.block3 = nn.ModuleList([Lo2Block(
            dim=embed_dims[3], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.norm1 = norm_layer(embed_dims[3])
        self.norm2 = norm_layer(embed_dims[4])
        self.norm3 = norm_layer(embed_dims[3])

        self.FIBlock4 = nn.Conv2d(embed_dims[3], embed_dims[2], 3, stride=1, padding=1)
        self.dbn4 = nn.BatchNorm2d(embed_dims[2])

        self.decoder3 = D_DoubleConv(embed_dims[2], embed_dims[1])
        self.decoder2 = D_DoubleConv(embed_dims[1], embed_dims[0])
        self.decoder1 = D_DoubleConv(embed_dims[0], 32)

        self.final = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        B = x.shape[0]

        ### Conv Stage
        out = self.conv1(x)
        t1 = out
        out = self.pool1(out)
        out = self.conv2(out)
        t2 = out
        out = self.pool2(out)
        out = self.conv3(out)
        t3 = out
        out = self.pool3(out)

        ### Stage 4
        out, H, W = self.FIBlock1(out)
        for i, blk in enumerate(self.block1):
            out = blk(out, H, W)
        out = self.norm1(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out
        out = self.pool4(out)

        ### Bottleneck
        out, H, W = self.FIBlock2(out)
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)
        out = self.norm2(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out, H, W = self.FIBlock3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.interpolate(out, scale_factor=(2, 2), mode='bilinear')

        ### Stage 4
        out = torch.add(out, t4)
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.block3):
            out = blk(out, H * 2, W * 2)
        out = self.norm3(out)
        out = out.reshape(B, H * 2, W * 2, -1).permute(0, 3, 1, 2).contiguous()
        out = F.interpolate(F.relu(self.dbn4(self.FIBlock4(out))), scale_factor=(2, 2), mode='bilinear')

        ### Conv Stage
        out = torch.add(out, t3)
        out = F.interpolate(self.decoder3(out), scale_factor=(2, 2), mode='bilinear')
        out = torch.add(out, t2)
        out = F.interpolate(self.decoder2(out), scale_factor=(2, 2), mode='bilinear')
        out = torch.add(out, t1)
        out = self.decoder1(out)

        out = self.final(out)

        return out