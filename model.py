"""
本模型主要参考pvt和pvtv2原始论文及源码

English translation:
This model mainly refers to the pvt and pvtv2 original papers and source code

pvtv2 source code：https://github.com/whai362/PVT/blob/v2/classification/pvt_v2.py
"""
import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class patch_embed(nn.Module):
    def __init__(self, in_channels=6, patch_size=4, embed_dim=32):
        super(patch_embed, self).__init__()
        self.embed = nn.Conv2d(in_channels=in_channels,
                               out_channels=embed_dim,
                               kernel_size=2 * patch_size - 1,
                               stride=patch_size,
                               padding=patch_size - 1)
        self.layer_norm = nn.LayerNorm(normalized_shape=embed_dim)

    def forward(self, x):
        # [B, embed_dim, H/patch_size, W/patch_size]
        x = self.embed(x)
        # [B, embed_dim, H * W / patch_size^2]
        x = x.flatten(2)
        # [B, H * W / patch_size^2, embed_dim]
        x = x.transpose(1, 2)
        x = self.layer_norm(x)

        return x


class SRA(nn.Module):
    def __init__(self, embed_dim=32, sr_ratio=8, num_head=1, drop_rate=0., atten_drop_rate=0.):
        super(SRA, self).__init__()
        self.embed_dim = embed_dim
        self.sr_ratio = sr_ratio
        self.num_head = num_head
        head_dim = embed_dim // num_head
        self.scale = head_dim ** -0.5

        self.kv = nn.Linear(embed_dim, embed_dim * 2)
        self.q = nn.Linear(embed_dim, embed_dim)

        self.atten_drop = nn.Dropout(atten_drop_rate)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(drop_rate)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(embed_dim, embed_dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, H, W):
        B, N, C = x.shape # N = H * W
        q = self.q(x).reshape(B, N, self.num_head, C // self.num_head).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            x = x.permute(0, 2, 1).reshape(B, C, H, W)
            x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
            x = self.layer_norm(x)
        kv = self.kv(x).reshape(B, -1, 2, self.num_head, C // self.num_head).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attenion = (q @ k.transpose(-2, -1)) * self.scale
        attenion = attenion.softmax(dim=-1)
        attenion = self.atten_drop(attenion)
        x = (attenion @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    本函数及droppath类均来源于ViT的复现代码
    English: This function and the droppath class are derived from the duplicate code of ViT
    duplicate code of ViT: https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/blob/master/pytorch_classification/vision_transformer/vit_model.py
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor

    return output


class droppath(nn.Module):
    def __init__(self, drop_prob=None):
        super(droppath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class dwconv(nn.Module):
    def __init__(self, dim=768):
        super(dwconv, self).__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.conv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class feed_forward(nn.Module):
    def __init__(self, input_size, hidden_size, drop_rate):
        super(feed_forward, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dwconv = dwconv(hidden_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_size, input_size)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class block(nn.Module):
    def __init__(self, H, W,
                 embed_dim=32,
                 sr_ratio=1,
                 mlp_ratio=8,
                 num_head=1,
                 drop_rate=0.,
                 atten_drop_rate=0.,
                 drop_path_rate=0.):
        super(block, self).__init__()
        self.H = H
        self.W = W
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.sra = SRA(embed_dim=embed_dim,
                       sr_ratio=sr_ratio,
                       num_head=num_head,
                       drop_rate=drop_rate,
                       atten_drop_rate=atten_drop_rate
                       )
        self.droppath = droppath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.fc = feed_forward(input_size=embed_dim, hidden_size=embed_dim * mlp_ratio, drop_rate=drop_rate)

    def forward(self, x):
        # [B, H * W / patch_size^2, embed_dim]
        x_res = self.layer_norm(x)
        x_res = self.sra(x_res, self.H, self.W)
        x_res = self.droppath(x_res)
        x = x + x_res

        x_res = self.layer_norm(x)
        x_res = self.fc(x_res, self.H, self.W)
        x_res = self.droppath(x_res)
        x = x + x_res

        return x


class stage(nn.Module):
    def __init__(self,
                 H, W,
                 input_channels=6,
                 patch_size=4,
                 embed_dim=32,
                 block_num=8,
                 sr_ratio=8,
                 mlp_ratio=4,
                 num_head=1,
                 drop_rate=0.,
                 atten_drop_rate=0.,
                 drop_path_rate=0.
                 ):
        super(stage, self).__init__()
        self.embed = patch_embed(in_channels=input_channels, patch_size=patch_size, embed_dim=embed_dim)
        self.blocks = nn.ModuleList([
            block(
                H // patch_size, W // patch_size,
                embed_dim=embed_dim,
                sr_ratio=sr_ratio,
                mlp_ratio=mlp_ratio,
                num_head=num_head,
                drop_rate=drop_rate,
                atten_drop_rate=atten_drop_rate,
                drop_path_rate=drop_path_rate
            )
        for i in range(block_num)])
        self.H = H // patch_size
        self.W = W // patch_size

    def forward(self, x):
        x = self.embed(x)
        for blk in self.blocks:
            x = blk(x)
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, self.H, self.W)
        return x


class encoder(nn.Module):
    def __init__(self,
                 H, W,
                 stage_num=4,
                 input_channels=6,
                 patch_size=(4, 2, 2, 2),
                 embed_dim=(32, 64, 160, 256),
                 block_num=(2, 2, 2, 2),
                 sr_ratio=(8, 4, 2, 1),
                 mlp_ratio=(8, 8, 4, 4),
                 num_head=(1, 2, 5, 8),
                 drop_rate=0.3,
                 atten_drop_rate=0.3,
                 drop_path_rate=0.3
                 ):
        super(encoder, self).__init__()
        self.stage_num = stage_num
        self.stages = nn.ModuleList()
        for i in range(stage_num):
            self.stages.append(stage(
                H=H, W=W,
                input_channels=input_channels,
                patch_size=patch_size[i],
                embed_dim=embed_dim[i],
                block_num=block_num[i],
                sr_ratio=sr_ratio[i],
                mlp_ratio=mlp_ratio[i],
                num_head=num_head[i],
                drop_rate=drop_rate,
                atten_drop_rate=atten_drop_rate,
                drop_path_rate=drop_path_rate
            ))
            input_channels = embed_dim[i]
            H //= patch_size[i]
            W //= patch_size[i]

    def forward(self, x):
        origin = [] # record the output of every stage, and the last one is the result of encoding
        for model in self.stages:
            x = model(x)
            origin.append(x)
        return origin


class deconv(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size, last=False):
        """
        Every deconv layer have three parts of option:
        1. deconv
        2. residual splicing
        3. 2*conv (not change the image size)
        """
        super(deconv, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=patch_size+1, stride=patch_size, padding=1, output_padding=1)
        self.last = last
        if not last:
            self.conv1 = nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels, kernel_size=7, stride=1, padding=3) # Residual splicing
        else:
            self.conv1 = nn.Conv2d(in_channels=out_channels + 6, out_channels=out_channels, kernel_size=7, stride=1, padding=3) # Splicing the original version of the previous frame and the latter frame
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.act = nn.ReLU()

    def forward(self, x, output=None):
        x = self.deconv(x)
        # if not self.last:
        x = self.act(x)
        x = torch.cat([x, output], dim=1)
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        return x


class decoder(nn.Module):
    def __init__(self, deconv_num, embed_dim, patch_size):
        super(decoder, self).__init__()
        self.deconv_num = deconv_num
        self.embed_dim = embed_dim
        self.deconvs = nn.ModuleList()
        for i in range(len(embed_dim) - 1):
            if i < len(embed_dim) - 2:
                self.deconvs.append(deconv(embed_dim[-i-1], embed_dim[-i-2], patch_size[-i - 1]))
            else:
                self.deconvs.append(deconv(embed_dim[-i-1], embed_dim[-i-2], patch_size[-i - 1], last=True))

    def forward(self, outputs, x_origin):
        """
        :param outputs: list of the output after each stage (output[-1] means the final version of encoder)
        :param x_origin: the original version of the previous frame and the latter frame
        :return: size = [B, layer_before_predict, H, W]
        """
        x = outputs[-1]
        for i in range(len(outputs) - 1):
            x = self.deconvs[i](x, outputs[- i - 2])
        x = self.deconvs[-1](x, x_origin)
        return x


class VFImodel(nn.Module):
    def __init__(self,
                 H, W,
                 stage_num=4,
                 input_channels=6,
                 patch_size=(4, 2, 2, 2),
                 embed_dim=(32, 64, 160, 256),
                 block_num=(2, 2, 2, 2),
                 sr_ratio=(8, 4, 2, 1),
                 mlp_ratio=(8, 8, 4, 4),
                 num_head=(1, 2, 5, 8),
                 drop_rate=0.3,
                 atten_drop_rate=0.3,
                 drop_path_rate=0.3,
                 layer_before_predict=16
                 ):
        """
        :param H: Height of the video.
        :param W: Width of the video
        :param stage_num: The number of floors in the pyramid.
        :param input_channels: 3 + 3, namely the sum of the number of channels in two color images.
        :param patch_size: The reduction in the width and length of each floor of the pyramid.*
        :param embed_dim: The number of channels in each floor of the pyramid.*
        :param block_num: The encoding times in each floor of the pyramid.*
        :param sr_ratio: The downsampling factor in the attention module of SRA in each floor of the pyramid.*
        :param mlp_ratio: The scaling factor of each layer in the feed forward in each floor of the pyramid.*
        :param num_head: Numbers of heads of self-attention in each floor of the pyramid.*
        :param drop_rate: Dropout rate.
        :param atten_drop_rate: Attention dropout rate.
        :param drop_path_rate:  Droppath rate.
        :param layer_before_predict: After upsampling with the deconvolution operation, add an intermediate layer for transition.
        *: Refer to the PVT original paper and code.
        """
        super(VFImodel, self).__init__()
        self.encoding = encoder(H, W,
                 stage_num=stage_num,
                 input_channels=input_channels,
                 patch_size=patch_size,
                 embed_dim=embed_dim,
                 block_num=block_num,
                 sr_ratio=sr_ratio,
                 mlp_ratio=mlp_ratio,
                 num_head=num_head,
                 drop_rate=drop_rate,
                 atten_drop_rate=atten_drop_rate,
                 drop_path_rate=drop_path_rate)
        self.decoding = decoder(deconv_num=stage_num, embed_dim=[layer_before_predict] + list(embed_dim), patch_size=patch_size)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(layer_before_predict, 3, kernel_size=7, padding=3, stride=1)

    def forward(self, x1, x2):
        """
        Using the previous frame and the latter frame to predict the intermediate frame.
        :param x1: size = [B, 3, H, W]
        :param x2: size = [B, 3, H, W]
        :return: predict answer, size = [B, 3, H, W]
        """
        # x = torch.cat([self.relu(x1-x2), self.relu(x2-x1)], dim=1)
        x = torch.cat([x1, x2], dim=1)
        y = self.encoding(x)
        y = self.decoding(y, x)
        ans = self.conv(y)
        ans = self.sigmoid(ans)
        # y1, y2 = torch.split(y, split_size_or_sections=3, dim=1)
        # y1 = self.sigmoid(y1)
        # y2 = self.tanh(y2)
        # ans = y1 * x1 + (torch.ones(x2.shape).to(device) - y1) * x2 + y2
        return ans


if __name__ == '__main__':
    """
    长和宽需要是patchsize的累乘的整数倍（如果不是，需要调整decoder）
    每次的embed_dim要等于num_head的倍数
    
    English translation:
    The length and width need to be multiples of the patch size (if not, the decoder needs to be adjusted).
    The embed_dim at each step should be a multiple of the number of heads.
    """
    device = torch.device('cpu')
    print('begin')
    model = VFImodel(256, 448)
    model.eval()
    x1 = torch.rand(10, 3, 256, 448)
    x2 = torch.rand(10, 3, 256, 448)
    ans = model(x1, x2)
    # for y in ans:
    #     print(y.shape)
    print(ans.shape)