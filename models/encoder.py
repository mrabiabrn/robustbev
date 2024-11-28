    
import torch
import torch.nn as nn
import timm
from .lora import LoRA_ViT_timm


class Encoder_dinov2(nn.Module):
    def __init__(self, C, encoder_args):
        super().__init__()

        encoder_type = encoder_args['encoder_type']
        res = encoder_args['resolution']
        use_lora = encoder_args['use_lora']
        lora_rank = encoder_args['lora_rank']
        use_qkv = encoder_args['use_qkv']
        finetune = encoder_args['finetune']


        if encoder_type == "dinov2_s":
            self.model = timm.create_model("vit_small_patch14_dinov2.lvd142m", img_size=res, pretrained=True, num_classes=0)
            self.model_dim = 384
        
        elif encoder_type == "dinov2_b":
            self.model = timm.create_model("vit_base_patch14_dinov2.lvd142m", img_size=res, pretrained=True, num_classes=0)
            self.model_dim = 768
        
        elif encoder_type == "dinov2_l":
            self.model = timm.create_model("vit_large_patch14_dinov2.lvd142m", img_size=res, pretrained=True, num_classes=0)
            self.model_dim = 1024

        elif encoder_type == "dinov2_g":
            self.model = timm.create_model("vit_giant_patch14_dinov2.lvd142m", img_size=res, pretrained=True, num_classes=0)
            self.model_dim = 1536

        self.patch_size = 14
        self.Hf, self.Wf = res[0]//self.patch_size, res[1]//self.patch_size

        self.use_lora = use_lora
        self.finetune = finetune

        # Adaptation Setting
        if use_lora:
            self.model = LoRA_ViT_timm(self.model, r=lora_rank, use_qkv=use_qkv) 
        # Finetuning
        elif finetune:
            for param in self.model.parameters():
                param.requires_grad = True
            self.model.train()
        # Frozen
        else:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

        self.token_projection = nn.Linear(self.model_dim, C)
        self.C = C
    
    def forward(self, x):
    

        if self.use_lora:
            tokens = self.model(x)                  # (B * S, P, D_model)
        elif self.finetune:
            tokens = self.model.forward_features(x)[:, 1:]                   # (B * S, P, D_model
        elif self.use_vit_adaptation:
            tokens = self.model(x)[1]             # (B * S, D_model, h, w)
            tokens = tokens.permute(0, 2, 3, 1)    # (B * S, h, w, D_model)
        else:
            self.model.eval()
            with torch.no_grad():
                tokens = self.model.forward_features(x)[:, 1:]     
        
        tokens = self.token_projection(tokens)                                # (B * N_cam, P, D)

        tokens = tokens.reshape(-1,6,self.Hf,self.Wf,self.C)                        # B, S, h, w, D
        tokens = tokens.reshape(-1, self.Hf, self.Wf, self.C)                   # B, S, h*w, D
        tokens = tokens.permute(0,3,1,2)                                       # B*S, D, h, w

        return tokens
    



"""
    Adapted from https://github.com/aharley/simple_bev
"""


class UpsamplingConcat(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_to_upsample, x):
        x_to_upsample = self.upsample(x_to_upsample)
        x_to_upsample = torch.cat([x, x_to_upsample], dim=1)
        return self.conv(x_to_upsample)



import torchvision
class Encoder_res101(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.C = C
        resnet = torchvision.models.resnet101(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-4])
        self.layer3 = resnet.layer3

        self.depth_layer = nn.Conv2d(512, self.C, kernel_size=1, padding=0)
        self.upsampling_layer = UpsamplingConcat(1536, 512)

    def forward(self, x):
        x1 = self.backbone(x)
        x2 = self.layer3(x1)
        x = self.upsampling_layer(x2, x1)
        x = self.depth_layer(x)

        return x

    