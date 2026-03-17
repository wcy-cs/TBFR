import os, sys
import argparse
from pathlib import Path

from omegaconf import OmegaConf
from sampler import ResShiftSampler

from utils.util_opts import str2bool
from basicsr.utils.download_util import load_file_from_url
import time


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("-i", "--in_path", type=str, default="", help="Input path.")
    parser.add_argument("-o", "--out_path", type=str, default="./results", help="Output path.")
    parser.add_argument("-s", "--steps", type=int, default=15, help="Diffusion length.")
    parser.add_argument("--scale", type=int, default=4, help="Scale factor for SR.")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed.")
    parser.add_argument(
        "--chop_size",
        type=int,
        default=512,
        choices=[512, 256],
        help="Chopping forward.",
    )
    args = parser.parse_args()

    return args


def get_configs(args):
    configs = OmegaConf.load('./configs/blind_face_restoration256.yaml')

    # prepare the checkpoint
    ckpt_dir = Path('./weights')
    if not ckpt_dir.exists():
        ckpt_dir.mkdir()
    ckpt_path= ""
    ckpt_path=Path(ckpt_path)
    vqgan_path = ckpt_dir / f'autoencoder/autoencoder_vq_f4.pth'

    configs.model.ckpt_path = str(ckpt_path)
    configs.diffusion.params.steps = args.steps
    configs.autoencoder.ckpt_path = str(vqgan_path)

    # save folder
    if not Path(args.out_path).exists():
        Path(args.out_path).mkdir(parents=True)

    if args.chop_size==512:
        chop_stride = (512 - 64) * (16 // args.scale)

    elif args.chop_size==256:
        chop_stride = (256 - 32) * (16 // args.scale)
    else:
        raise ValueError("Chop size must be in [512, 256]")
    args.chop_size *= (16 // args.scale)
    autoencoder_scale = 2 ** (len(configs.autoencoder.params.ddconfig.ch_mult) - 1)
    desired_min_size = 16 * (16 // args.scale)

    return configs, chop_stride, desired_min_size


def main():
    args = get_parser()

    configs, chop_stride, desired_min_size = get_configs(args)

    # Define paths
    clip_ckpt_path = ""
    text_path = ""

    resshift_sampler = ResShiftSampler(
        configs,
        clip_ckpt_path=clip_ckpt_path,
        text_path=text_path,
        chop_size=args.chop_size,
        chop_stride=chop_stride,
        chop_bs=1,
        use_fp16=True,
        seed=args.seed,
        desired_min_size=desired_min_size,
    )

    resshift_sampler.inference(args.in_path, args.out_path, bs=1, noise_repeat=False)


if __name__=='__main__':
    main()
