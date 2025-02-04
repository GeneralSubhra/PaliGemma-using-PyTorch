import torch
from torch import nn
from timm.models.vision_transformer import VisionTransformer
from timm.models.layers import PatchEmbed
from timm.models.registry import register_model
import timm

class SimpleVisionTransformer(VisionTransformer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init=''):
        super().__init__(img_size, patch_size, in_chans, num_classes, embed_dim, depth,
                         num_heads, mlp_ratio, qkv_bias, representation_size, distilled,
                         drop_rate, attn_drop_rate, drop_path_rate, embed_layer, norm_layer,
                         act_layer, weight_init)

    def forward_features(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        return x[:, 0]  # Extract CLS token only

    def forward(self, x):
        x = self.forward_features(x)
        logits = self.head(x)  # Standard classification head
        return logits

@register_model
def vit_base_patch16_224_cls(pretrained=False, **kwargs):
    model = SimpleVisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    if pretrained:
        vit_model = timm.create_model('vit_base_patch16_224', pretrained=True, **kwargs)
        model.load_state_dict(vit_model.state_dict(), strict=False)
    return model
