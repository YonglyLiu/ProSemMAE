import os
from models.TransVW.ynet3d import UNet3D  # TransVW
from models.ModelsGenesis.unet3d import UNet3D as ModelsGenesis  # ModelsGenesis
from monai.networks.nets.swin_unetr import SwinUNETR # SwinUNETR
from models.GVSL.UNet_GVSL import UNet3D_GVSL  # GVSL
from models.PCRLv2 import PCRLv23d
from models.UniMiSS.MiT import MiTnet
from models.vox2vec.nn import FPN3d, FPNLinearHead
from models.vox2vec.eval.end_to_end import EndToEnd
from models.UNETR.unetr import UNETR
from models.HySparK.hybird3 import build_hybird

import torch



def load_UNet3D(num_classes):
    model = UNet3D(n_class=num_classes)
    model.cuda()
    return model

def load_unetr(num_classes):
    model = UNETR(
        in_channels=1,
        out_channels=num_classes,
        img_size=(96, 96, 96),
        feature_size=32,
        hidden_size=1536,
        mlp_dim=1536 * 4,
        num_heads=16,
        pos_embed="perceptron",
        norm_name="instance",
        conv_block=True,
        res_block=True,
        dropout_rate=0.0,
        qkv_bias=True
        )
    model.cuda()
    return model
def load_TransVW(num_classes):
    model = UNet3D(n_class=num_classes)
    weight_dir = ''
    checkpoint = torch.load(weight_dir, weights_only=False)
    state_dict = checkpoint['state_dict']
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    delete = [key for key in state_dict if "projection_head" in key]
    for key in delete: del state_dict[key]
    delete = [key for key in state_dict if "prototypes" in key]
    for key in delete: del state_dict[key]
    for key in state_dict.keys():
        if key in model.state_dict().keys():
            model.state_dict()[key].copy_(state_dict[key])
            print("Copying {} <---- {}".format(key, key))
        elif key.replace("classficationNet.", "") in model.state_dict().keys():
            model.state_dict()[key.replace("classficationNet.", "")].copy_(state_dict[key])
            print("Copying {} <---- {}".format(key.replace("classficationNet.", ""), key))
        else:
            print("Key {} is not found".format(key))
    model.cuda()
    return model

def load_ModelsGenesis(num_classes):
    model = ModelsGenesis(n_class=num_classes)
    weight_dir = ''
    checkpoint = torch.load(weight_dir, weights_only=False)
    state_dict = checkpoint['state_dict']
    unParalled_state_dict = {}
    for key in state_dict.keys():
        if "out_tr.final_conv" not in key:
            unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
    print(unParalled_state_dict.keys())
    model.load_state_dict(unParalled_state_dict, strict=False)
    model.cuda()
    return model


def load_SwinUNETR(num_classes):
    model = SwinUNETR(
        img_size=(96, 96, 96),
        in_channels=1,
        out_channels=num_classes,
        feature_size=48,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=True,
    )

    model.cuda()
    return model

def load_GVSL(num_classes):
    model = UNet3D_GVSL(n_classes=num_classes, pretrained_dir='').cuda()
    return model
    
def load_prlv2(num_classes):
    # pip install segmentation-models-pytorch
    model = PCRLv23d(n_class=num_classes)
    weight_path = ''
    model_dict = torch.load(weight_path, weights_only=False)['state_dict']
    model_dict.pop("out_tr.final_conv.weight")
    model_dict.pop("out_tr.final_conv.bias")
    model.load_state_dict(model_dict, strict=False)
    model.cuda()
    return model


def load_UniMiss(num_classes):
    model = MiTnet( in_chans=1, pretrain=True, pretrain_path="", activation_cfg='LeakyReLU',
           norm_cfg='IN', img_size=[96, 96, 96], num_classes=num_classes)
    model.cuda()
    return model

def load_VoCo(num_classes):
    model = SwinUNETR(
        img_size=(96, 96, 96),
        in_channels=1,
        out_channels=num_classes,
        feature_size=48,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=True,
        )
    model_dict = torch.load("", map_location=torch.device('cpu'))
    state_dict = model_dict
    if "module." in list(state_dict.keys())[0]:
        print("Tag 'module.' found in state dict - fixing!")
        for key in list(state_dict.keys()):
            state_dict[key.replace("module.", "")] = state_dict.pop(key)
    if "swin_vit" in list(state_dict.keys())[0]:
        print("Tag 'swin_vit' found in state dict - fixing!")
        for key in list(state_dict.keys()):
            state_dict[key.replace("swin_vit", "swinViT")] = state_dict.pop(key)
    model.load_state_dict(state_dict, strict=False)
    print("Using pretrained voco self-supervised Swin UNETR backbone weights !")
    model.cuda()
    return model


def load_vox2vec(num_classes):
    backbone = FPN3d(1, 16, 6)
    flag = backbone.load_state_dict(torch.load(""))
    head = FPNLinearHead(16, 6, num_classes)
    model = EndToEnd(backbone, head, patch_size=(96, 96, 96)).cuda()
    if flag:
        print("Using pretrained vox2vec weights !")
    return model


def load_hyspark(num_classes):
    model = build_hybird(in_channel=1, n_classes=num_classes, img_size=96).cuda()
    model_dict = torch.load("")
    flag = model.load_state_dict(model_dict, strict=False)  
    if flag:
        print("SparK use pretrained weights Successful")
    model.cuda()
    return model
 


def load_hyspark_brats(num_classes):

    # ============================================================
    # 1. Build BraTS model
    #    BraTS: T1 / T1ce / T2 / FLAIR -> 4 channels
    # ============================================================
    model = build_hybird(
        in_channel=4,
        n_classes=num_classes,
        img_size=96
    )

    print("\n========== HySparK ==========")
    print("BraTS input channels: 4")
    print("Stem:", model.embeddings.stem)
    print(
        "Stem shape:",
        model.embeddings.stem.weight.shape
    )

    # ============================================================
    # 2. Load CT pretrained checkpoint
    # ============================================================
    ckpt_path = ""

    model_dict = torch.load(
        ckpt_path,
        map_location="cpu"
    )

    # Compatible with checkpoint wrappers
    if "state_dict" in model_dict:
        model_dict = model_dict["state_dict"]

    elif "model" in model_dict:
        model_dict = model_dict["model"]

    # Remove possible "module." prefix
    model_dict = {
        k[7:] if k.startswith("module.") else k: v
        for k, v in model_dict.items()
    }

    # ============================================================
    # 3. Convert CT stem: 1 channel -> BraTS 4 channels
    # ============================================================
    stem_key = "embeddings.stem.weight"

    if stem_key in model_dict:

        pretrained_weight = model_dict[stem_key]
        target_weight = model.state_dict()[stem_key]

        print(
            "Pretrained stem shape:",
            pretrained_weight.shape
        )

        print(
            "Target stem shape:",
            target_weight.shape
        )

        if (
            pretrained_weight.ndim == 5
            and pretrained_weight.shape[1] == 1
            and target_weight.shape[1] == 4
        ):

            print(
                "Expanding CT stem: "
                "[32,1,3,3,3] -> [32,4,3,3,3]"
            )

            # Copy CT filters to all four MRI modalities
            pretrained_weight = pretrained_weight.repeat(
                1, 4, 1, 1, 1
            )

            # Keep approximately the same activation scale
            pretrained_weight = pretrained_weight / 4.0

            model_dict[stem_key] = pretrained_weight

    # ============================================================
    # 4. Remove any remaining incompatible parameters
    # ============================================================
    current_dict = model.state_dict()

    compatible_dict = {}

    for k, v in model_dict.items():

        if k not in current_dict:
            print(
                f"[Skip: not found] {k}"
            )
            continue

        if v.shape != current_dict[k].shape:
            print(
                f"[Skip: shape mismatch] {k}: "
                f"{tuple(v.shape)} -> "
                f"{tuple(current_dict[k].shape)}"
            )
            continue

        compatible_dict[k] = v

    # ============================================================
    # 5. Load pretrained parameters
    # ============================================================
    msg = model.load_state_dict(
        compatible_dict,
        strict=False
    )

    print("\n========== Loading Report ==========")

    print(
        f"Loaded: {len(compatible_dict)} / "
        f"{len(current_dict)} parameters"
    )

    if len(msg.missing_keys) > 0:
        print("\nMissing keys:")
        for k in msg.missing_keys:
            print("  ", k)

    if len(msg.unexpected_keys) > 0:
        print("\nUnexpected keys:")
        for k in msg.unexpected_keys:
            print("  ", k)

    print("====================================")

    # ============================================================
    # 6. Move to GPU
    # ============================================================
    model = model.cuda()

    return model
 

def load_UNETR(num_classes):
    model = UNETR(
        in_channels=1,
        out_channels=num_classes,
        img_size=(96, 96, 96),
        feature_size=32,
        hidden_size=1536,
        mlp_dim=1536 * 4,
        num_heads=16,
        pos_embed="perceptron",
        norm_name="instance",
        conv_block=True,
        res_block=True,
        dropout_rate=0.0,
        qkv_bias=True
        )
    model.cuda()
    return model

def load_hi_end_mae_10k(num_classes):
    model = UNETR(
        in_channels=1,
        out_channels=num_classes,
        img_size=(96, 96, 96),
        feature_size=32,
        hidden_size=1536,
        mlp_dim=1536 * 4,
        num_heads=16,
        pos_embed="perceptron",
        norm_name="instance",
        conv_block=True,
        res_block=True,
        dropout_rate=0.0,
        qkv_bias=True
        )
    model_dict = torch.load("")['state_dict']

    pretrained_dict = {}
    pretrained_dict["vit.patch_embedding.patch_embeddings.1.weight"] = model_dict["module.encoder.patch_embed.proj.weight"].flatten(1)
    pretrained_dict["vit.patch_embedding.patch_embeddings.1.bias"] = model_dict["module.encoder.patch_embed.proj.bias"]
    pretrained_dict["vit.patch_embedding.position_embeddings"] = model_dict["module.encoder_pos_embed"]
    for k, v in model_dict.items():
       if "encoder." in k and "cls_token" not in k and "patch_embed" not in k:
           pretrained_dict[k.replace("module.encoder", "vit").replace("fc", "linear").replace("proj", "out_proj")] = v
    model.load_state_dict(pretrained_dict, strict=False)
    print("Using pretrained self-supervised UNETR backbone weights !")
    model.cuda()
    return model

def load_ProSemMAE(num_classes):
    model = UNETR(
        in_channels=1,
        out_channels=num_classes,
        img_size=(96, 96, 96),
        feature_size=32,
        hidden_size=1536,
        mlp_dim=1536 * 4,
        num_heads=16,
        pos_embed="perceptron",
        norm_name="instance",
        conv_block=True,
        res_block=True,
        dropout_rate=0.0,
        qkv_bias=True
        )
    ckpt = torch.load("", map_location="cpu")
    model_dict = ckpt["state_dict"]


    pretrained_dict = {}

    # patch embedding
    pretrained_dict[
        "vit.patch_embedding.patch_embeddings.1.weight"
    ] = model_dict["module.encoder.patch_embed.proj.weight"].flatten(1)

    pretrained_dict[
        "vit.patch_embedding.patch_embeddings.1.bias"
    ] = model_dict["module.encoder.patch_embed.proj.bias"]

    pretrained_dict[
        "vit.patch_embedding.position_embeddings"
    ] = model_dict["module.encoder_pos_embed"]

    # transformer blocks
    for k, v in model_dict.items():
        if "encoder." in k and "cls_token" not in k and "patch_embed" not in k:
            new_k = (
                k.replace("module.encoder", "vit")
                 .replace("fc", "linear")
                 .replace("proj", "out_proj")
            )
            pretrained_dict[new_k] = v


    model.load_state_dict(pretrained_dict, strict=False)
    print("Using pretrained ProSemMAE UNETR backbone weights!")

    model.cuda()
    return model
