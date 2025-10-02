from templates import *
from templates_latent import *
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--batch_size', '-b', type=int, default=512,
                        help='evaluation batch size')
    parser.add_argument('--patch_size', '-ps', type=int, default=32,
                        help='model base patch size')
    parser.add_argument('--output_dir', '-g', type=str, default="./output_images",
                        help='generate path')
    parser.add_argument('--image_size', '-img', type=str, default="64x64",
                        help='iamge size (HxW)')
    parser.add_argument('--img_num', type=int, default=2,
                        help='generate image number')
    parser.add_argument('--full_path', '-f', type=str, default="",
                        help='full generation path')
    parser.add_argument('--semantic_enc', action='store_true',
                        help='use semantic encoder')
    parser.add_argument('--znormalize', '-norm', action='store_true',
                        help='latent znormalize')
    parser.add_argument('--backbone', type=str, default="unet",
                        help='backbone') # TK add
    parser.add_argument('--disable_latent_diffusion', action='store_true',
                        help='use default semantic code instead of latent diffusion') # TK add
    args = parser.parse_args()

    gpus = [0]
    conf = train_autoenc_latent()
    
    conf.patch_size = args.patch_size
    conf.output_dir = args.output_dir

    if args.disable_latent_diffusion:
        conf.output_dir = conf.output_dir + "_disable_latent"
    else:
        conf.output_dir = conf.output_dir + "_enable_latent"

    conf.seed = 0
    conf.batch_size = 8
    conf.batch_size_eval = args.batch_size
    conf.sample_size = 1
    conf.train_mode = TrainMode.latent_diffusion
    conf.beatgans_gen_type = GenerativeType.ddim
    conf.eval_programs = ['gen(50,100)']
    conf.semantic_enc = args.semantic_enc
    conf.latent_znormalize = args.znormalize
    conf.full_model_path = args.full_path
    conf.image_size = args.image_size
    conf.eval_num_images = args.img_num
    conf.name = ""

    conf.backbone = args.backbone
    conf.disable_latent_diffusion = args.disable_latent_diffusion
    train(conf, gpus=gpus, mode='eval')
