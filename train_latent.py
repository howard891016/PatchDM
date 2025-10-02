from templates import *
from templates_latent import *
import os
import torch
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_path', '-m', type=str, default="./checkpoints/nature1024/last.ckpt",
                        help='Patch-DM model path')
    parser.add_argument('--semantic_enc', action='store_true',
                        help='use semantic encoder')
    parser.add_argument('--name', '-n', type=str, default="train",
                        help='experiment name')
    parser.add_argument('--img_num', type=int, default=50000,
                        help='generate image number')
    parser.add_argument('--batch_size', '-b', type=int, default=512,
                        help='batch size (all gpus)')   # TK add
    parser.add_argument('--batch_size_semantic_enc', type=int, default=1,
                        help='evaluation batch size for semantic encoder') # TK add
    parser.add_argument('--data_path', '-d', type=str, default="",
                        help='dataset path')            # TK add
    args = parser.parse_args()

    model_path = args.model_path

    conf = train_autoenc_latent()
    gpus = [1]
    conf.sample_size = 1
    conf.semantic_enc = args.semantic_enc
    conf.eval_num_images = args.img_num
    conf.batch_size = args.batch_size
    conf.batch_size_semantic_enc = args.batch_size_semantic_enc
    conf.data_path = args.data_path
    conf.latent_znormalize = False
    conf.save_every_samples = 100_0000
    conf.pretrain = PretrainConfig(
        name='72M',
        path=model_path,
    )
    conf.name = args.name
    train(conf, gpus=gpus)
