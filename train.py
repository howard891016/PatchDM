from templates import *
from templates_latent import *
import sys
import os

import argparse

# --- 動態路徑設定開始 ---
import os, sys

# 1. 取得目前 train.py 所在的資料夾
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. 找到它的上層資料夾（因為 LDM_Patch 跟 PatchDM 是同層）
parent_dir = os.path.dirname(current_dir)

# 3. 組合出 LDM_Patch 的完整路徑
ldm_patch_path = os.path.join(parent_dir, 'LDM_Patch')

# 4. 如果存在，就加入 sys.path
if os.path.exists(ldm_patch_path) and ldm_patch_path not in sys.path:
    print(f"將子專案路徑加入到 sys.path: {ldm_patch_path}")
    sys.path.insert(0, ldm_patch_path)

# --- 動態路徑設定結束 ---



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--batch_size', '-b', type=int, default=96,
                        help='batch size (all gpus)')
    parser.add_argument('--patch_size', '-ps', type=int, default=32,
                        help='model base patch size')
    parser.add_argument('--data_path', '-d', type=str, default="./dataset",
                        help='dataset path')
    parser.add_argument('--name', '-n', type=str, default="train",
                        help='experiment name')
    parser.add_argument('--semantic_enc', action='store_true',
                        help='use semantic encoder')
    parser.add_argument('--semantic_path', type=str, default="",
                        help='semantic code path')
    parser.add_argument('--cfg', action='store_true',
                        help='use cfg')
    # (Howard add) Use LDM arch
    parser.add_argument('--ldm', action='store_true',
                        help='use cfg')
    parser.add_argument('--whole_patch', action='store_true',
                        help='use whole patch')
    parser.add_argument('--use_vae', action='store_true',
                        help='use vae')

    args = parser.parse_args()
    gpus = [0,1]

    conf = train_autoenc()

    conf.batch_size = args.batch_size
    conf.sample_size = len(gpus)
    conf.save_every_samples = 100_000
    conf.patch_size = args.patch_size
    conf.name = args.name
    conf.data_path = args.data_path
    conf.semantic_enc = args.semantic_enc
    conf.semantic_path = args.semantic_path
    conf.cfg = args.cfg
    conf.lr = 1e-4
    conf.min_lr = 1e-6
    conf.lr_decay_gamma = 0.9999
    conf.loss_type = LossType.mse
    conf.whole_patch = args.whole_patch
    conf.use_vae = args.use_vae

    if args.semantic_path: assert not args.semantic_enc, "Semantic Encoder mode should turn off"

    if conf.patch_size == 256:
        conf.net_ch = 128
        conf.net_ch_mult = (1, 1, 2, 2, 4, 4)
        conf.net_enc_channel_mult = (1, 1, 2, 2, 4, 4, 4)
        conf.make_model_conf()
    elif conf.patch_size == 128:
        conf.net_ch = 128
        conf.net_ch_mult = (1, 1, 2, 3, 4)
        conf.net_enc_channel_mult = (1, 1, 2, 3, 4, 4)
        conf.make_model_conf()
    # elif conf.patch_size == 64:
    #     conf.net_ch_mult = (1, 2, 4, 8)
    #     conf.net_ch = 128
    #     # conf.net_ch_mult = (1, 2, 2, 2)
    #     conf.net_num_res_blocks = 4
    #     conf.net_enc_channel_mult = (1, 2, 4, 8, 8)
    #     # conf.net_enc_channel_mult = (1, 2, 2, 2)
    elif conf.patch_size == 64:
        conf.net_ch_mult = (1, 2, 4, 8)
        conf.net_enc_channel_mult = (1, 2, 4, 8, 8)
    elif conf.patch_size == 32:
        conf.net_ch = 128  # 與 64x64 的基礎通道數相同，但可以減少為 32 作為備選
        conf.net_ch_mult = (1, 2, 4)  # 確保層數與輸入大小匹配
        # conf.net_ch_mult = (1, 2, 2, 2)  # 確保層數與輸入大小匹配
        conf.net_num_res_blocks = 4
        # conf.net_enc_channel_mult = (1, 2, 2, 2)  # 避免通道數倍增過大
        conf.net_enc_channel_mult = (1, 2, 4, 8, 8)
        conf.make_model_conf()
    elif conf.patch_size == 16: 
        print("Use patch size 16")
        conf.net_ch = 128  # 與 32x32 的基礎通道數相同，但可以減少為 16 作為備選
        conf.net_ch_mult = (1, 2, 4)
        conf.net_num_res_blocks = 2
        conf.net_attn = (8, 4, 2)
        conf.net_enc_channel_mult = (1, 2, 4, 4)
        conf.make_model_conf()
    else:
        raise NotImplementedError("Patch size not in [32,64,128,256]")

    vae_conf = VaeConfig(
        target="ldm.models.autoencoder.VQModelInterface",
        params={
            "embed_dim": 3,
            "n_embed": 8192,
            "ddconfig": {
                "double_z": False,
                "z_channels": 3,
                "resolution": 256,
                "in_channels": 3,
                "out_ch": 3,
                "ch": 128,
                "ch_mult": [
                    1,
                    2,
                    4
                ],
                "num_res_blocks": 2,
                "attn_resolutions": [],
                "dropout": 0.0
            },
            "lossconfig": {
                "target": "torch.nn.Identity",
            }
        }
    )

    conf.vae = vae_conf

    print("GPUS: " + str(gpus))
    train(conf, gpus=gpus)

    