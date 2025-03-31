from templates import *
from templates_latent import *

import argparse

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
    elif conf.patch_size == 64:
        # conf.net_ch_mult = (1, 2, 4, 8)
        conf.net_ch = 128
        conf.net_ch_mult = (1, 2, 2, 2)
        conf.net_num_res_blocks = 4
        # conf.net_enc_channel_mult = (1, 2, 4, 8, 8)
        conf.net_enc_channel_mult = (1, 2, 2, 2)
    elif conf.patch_size == 32:
        conf.net_ch = 128  # 與 64x64 的基礎通道數相同，但可以減少為 32 作為備選
        conf.net_ch_mult = (1, 2, 4, 8)  # 確保層數與輸入大小匹配
        # conf.net_ch_mult = (1, 2, 2, 2)  # 確保層數與輸入大小匹配
        conf.net_num_res_blocks = 4
        # conf.net_enc_channel_mult = (1, 2, 2, 2)  # 避免通道數倍增過大
        conf.net_enc_channel_mult = (1, 2, 4, 8, 8)
        conf.make_model_conf()
    else:
        raise NotImplementedError("Patch size not in [32,64,128,256]")
    
    print("GPUS: " + str(gpus))
    train(conf, gpus=gpus)

    