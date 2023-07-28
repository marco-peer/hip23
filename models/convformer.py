
import math

import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from timm.models.layers.helpers import to_2tuple

class Scale(nn.Module):
    """
    Scale vector by element multiplications.
    """
    def __init__(self, dim, init_value=1.0, trainable=True):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x):
        return x * self.scale
        

class StarReLU(nn.Module):
    """
    StarReLU: s * relu(x) ** 2 + b
    """
    def __init__(self, scale_value=1.0, bias_value=0.0,
        scale_learnable=True, bias_learnable=True, 
        mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
            requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
            requires_grad=bias_learnable)
        
    def forward(self, x):
        return self.scale * self.relu(x)**2 + self.bias

class SepConv(nn.Module):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """
    def __init__(self, dim, expansion_ratio=2,
        act1_layer=StarReLU, act2_layer=nn.Identity,
        bias=False, kernel_size=7, padding=3,
        **kwargs, ):
        super().__init__()
        med_channels = int(expansion_ratio * dim)
        self.pwconv1 = nn.Linear(dim, med_channels, bias=bias)
        self.act1 = act1_layer()
        self.dwconv = nn.Conv2d(
            med_channels, med_channels, kernel_size=kernel_size,
            padding=padding, groups=med_channels, bias=bias) # depthwise conv
        self.act2 = act2_layer()
        self.pwconv2 = nn.Linear(med_channels, dim, bias=bias)

    def forward(self, x):
        x = self.pwconv1(x)
        x = self.act1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.act2(x)
        x = self.pwconv2(x)
        return x

class Mlp(nn.Module):
    """ MLP as used in MetaFormer models, eg Transformer, MLP-Mixer, PoolFormer, MetaFormer baslines and related networks.
    Mostly copied from timm.
    """
    def __init__(self, dim, mlp_ratio=4, out_features=None, act_layer=StarReLU, drop=0., bias=False, **kwargs):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class MetaFormerBlock(nn.Module):
    """
    Implementation of one MetaFormer block.
    """
    def __init__(self, dim,
                 token_mixer=nn.Identity, mlp=Mlp,
                 norm_layer=nn.LayerNorm,
                 drop=0., drop_path=0.,
                 layer_scale_init_value=1e-5, res_scale_init_value=None
                 ):

        super().__init__()

        self.norm1 = norm_layer(dim)
        self.token_mixer = token_mixer(dim=dim, drop=drop)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale1 = Scale(dim=dim, init_value=layer_scale_init_value) \
            if layer_scale_init_value else nn.Identity()
        self.res_scale1 = Scale(dim=dim, init_value=res_scale_init_value) \
            if res_scale_init_value else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp(dim=dim, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale2 = Scale(dim=dim, init_value=layer_scale_init_value) \
            if layer_scale_init_value else nn.Identity()
        self.res_scale2 = Scale(dim=dim, init_value=res_scale_init_value) \
            if res_scale_init_value else nn.Identity()
        
    def forward(self, x):
        x = self.res_scale1(x) + \
            self.layer_scale1(
                self.drop_path1(
                    self.token_mixer(self.norm1(x))
                )
            )
        x = self.res_scale2(x) + \
            self.layer_scale2(
                self.drop_path2(
                    self.mlp(self.norm2(x))
                )
            )
        return x


class MixPool(nn.Module):
    def __init__(self,
                 in_channels=1024,
                 in_h=20,
                 in_w=20,
                 out_channels=512,
                 mix_depth=4,
                 mlp_ratio=1,
                 out_rows=2,
                 ) -> None:
        super().__init__()

        self.in_h = in_h # height of input feature maps
        self.in_w = in_w # width of input feature maps
        self.in_channels = in_channels # depth of input feature maps
        
        self.out_channels = out_channels # depth wise projection dimension
        self.out_rows = out_rows # row wise projection dimesion

        self.mix_depth = mix_depth # L the number of stacked FeatureMixers
        self.mlp_ratio = mlp_ratio # ratio of the mid projection layer in the mixer block

        hw = in_h*in_w
        self.mix = nn.Sequential(*[
            MetaFormerBlock(in_channels, token_mixer=SepConv)
            for _ in range(self.mix_depth)
        ])
        self.channel_proj = nn.Linear(in_channels, out_channels)
        self.row_proj = nn.Linear(hw, out_rows)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.mix(x)
        x = x.permute(0, 3, 1, 2).flatten(2)

        x = x.permute(0, 2, 1)
        x = self.channel_proj(x)
        x = x.permute(0, 2, 1)
        x = self.row_proj(x)
        x = F.normalize(x.flatten(1), p=2, dim=-1)
        return x


class ResNet(nn.Module):
    def __init__(self,
                 model_name='resnet50',
                 pretrained=True,
                 layers_to_freeze=2,
                 layers_to_crop=[],
                 ):
        """Class representing the resnet backbone used in the pipeline
        we consider resnet network as a list of 5 blocks (from 0 to 4),
        layer 0 is the first conv+bn and the other layers (1 to 4) are the rest of the residual blocks
        we don't take into account the global pooling and the last fc
        Args:
            model_name (str, optional): The architecture of the resnet backbone to instanciate. Defaults to 'resnet50'.
            pretrained (bool, optional): Whether pretrained or not. Defaults to True.
            layers_to_freeze (int, optional): The number of residual blocks to freeze (starting from 0) . Defaults to 2.
            layers_to_crop (list, optional): Which residual layers to crop, for example [3,4] will crop the third and fourth res blocks. Defaults to [].
        Raises:
            NotImplementedError: if the model_name corresponds to an unknown architecture. 
        """
        super().__init__()
        self.model_name = model_name.lower()
        self.layers_to_freeze = layers_to_freeze

        if pretrained:
            # the new naming of pretrained weights, you can change to V2 if desired.
            weights = 'IMAGENET1K_V1'
        else:
            weights = None

        if 'swsl' in model_name or 'ssl' in model_name:
            # These are the semi supervised and weakly semi supervised weights from Facebook
            self.model = torch.hub.load(
                'facebookresearch/semi-supervised-ImageNet1K-models', model_name)
        else:
            if 'resnext50' in model_name:
                self.model = torchvision.models.resnext50_32x4d(weights=weights)
            elif 'resnet50' in model_name:
                self.model = torchvision.models.resnet50(weights=weights)
            elif '101' in model_name:
                self.model = torchvision.models.resnet101(weights=weights)
            elif '152' in model_name:
                self.model = torchvision.models.resnet152(weights=weights)
            elif '34' in model_name:
                self.model = torchvision.models.resnet34(weights=weights)
            elif '18' in model_name:
                # self.model = torchvision.models.resnet18(pretrained=False)
                self.model = torchvision.models.resnet18(weights=weights)
            elif 'wide_resnet50_2' in model_name:
                self.model = torchvision.models.wide_resnet50_2(weights=weights)
            else:
                raise NotImplementedError(
                    'Backbone architecture not recognized!')

        # freeze only if the model is pretrained
        if pretrained:
            if layers_to_freeze >= 0:
                self.model.conv1.requires_grad_(False)
                self.model.bn1.requires_grad_(False)
            if layers_to_freeze >= 1:
                self.model.layer1.requires_grad_(False)
            if layers_to_freeze >= 2:
                self.model.layer2.requires_grad_(False)
            if layers_to_freeze >= 3:
                self.model.layer3.requires_grad_(False)

        # remove the avgpool and most importantly the fc layer
        self.model.avgpool = None
        self.model.fc = None

        if 4 in layers_to_crop:
            self.model.layer4 = None
        if 3 in layers_to_crop:
            self.model.layer3 = None

        out_channels = 2048
        if '34' in model_name or '18' in model_name:
            out_channels = 512
            
        self.out_channels = out_channels // 2 if self.model.layer4 is None else out_channels
        self.out_channels = self.out_channels // 2 if self.model.layer3 is None else self.out_channels

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        if self.model.layer3 is not None:
            x = self.model.layer3(x)
        if self.model.layer4 is not None:
            x = self.model.layer4(x)
        return x

class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()        
        self.backbone = ResNet(model_name='resnet18', layers_to_freeze=-1, layers_to_crop=[])
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        return F.normalize(self.avgpool(self.backbone(x)).view(x.shape[0], -1))
    


class ResNet34(nn.Module):
    def __init__(self):
        super().__init__()        
        self.model = ResNet(model_name='resnet34', layers_to_freeze=-1, layers_to_crop=[])
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        return F.normalize(self.avgpool(self.model(x)).view(x.shape[0], -1))
    
class ResNet34Mixer(nn.Module):
    def __init__(self, img_size):
        super().__init__()        
        self.backbone = ResNet(model_name='resnet34', layers_to_freeze=-1, layers_to_crop=[])
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        out, h, w = self.backbone(torch.rand(1,3,*img_size)).shape[1:]

        self.mix = nn.Sequential(*[
            MetaFormerBlock(out, token_mixer=SepConv)
            for _ in range(4)
        ])
    def forward(self, x):
        return F.normalize(self.avgpool(self.backbone(x)).view(x.shape[0], -1))

    def forward(self, x):
        x = self.backbone(x)
        x = x.permute(0, 2, 3, 1)
        x = self.mix(x)
        x = x.permute(0, 3, 1, 2)
        x = F.normalize(self.avgpool(x).view(x.shape[0], -1))
        return x
    


class ModelWrapper(torch.nn.Module):

    def __init__(self, model, num_classes, img_size, mode='retrieval', lookup = None):
        super().__init__()
        self.model = model
        self.mode = mode
        self.num_classes = num_classes
        self.lookup = lookup

        output_dim = model(torch.rand(1,3, *img_size)).view(1, -1).shape[1]
        self.classifier = torch.nn.Linear(output_dim, num_classes, bias=False)
        self.softmax = torch.nn.Softmax(dim=1)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x):
        emb = self.model(x)

        # classification: always return classes
        if self.mode == 'classification':
            return self.softmax(self.classifier(self.dropout(emb)))
        
        # retrieval: train with classification and return embedding
        if self.mode == 'retrieval' and self.training:
            return self.softmax(self.classifier(self.dropout(emb)))
        else:
            return emb

class Model(nn.Module):

    def __init__(self, img_size=(512, 512), backbone='resnet34', out_channels=512, mix_depth=4, out_rows=2):
        super().__init__()
        self.backbone = ResNet(model_name=backbone, layers_to_freeze=-1, layers_to_crop=[])

        print(f' Using {mix_depth} mixer layers')
        print(f' Using {out_rows} out_rows')
        print(f' Using {out_channels} out_channels')
        print(f' Using {backbone} as backbone')

        out, h, w = self.backbone(torch.rand(1,3,*img_size)).shape[1:]
        print(f"Mixing features of size {out, h, w}")
        self.agg = MixPool(in_channels=out, in_h=h, in_w=w, mix_depth=mix_depth, out_channels=out_channels, out_rows=out_rows)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.agg(x)
        return F.normalize(x)