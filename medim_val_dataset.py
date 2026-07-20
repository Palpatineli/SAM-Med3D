# -*- encoding: utf-8 -*-

from typing import NamedTuple
from sys import stdout
import logging
from argparse import ArgumentParser
from pathlib import Path

import medim
from tqdm import tqdm

from utils.infer_utils import validate_paired_img_gt

class Args(NamedTuple):
    image: Path
    groundtruth: Path
    output: Path
    checkpoint: Path

def parse_args() -> Args:
    parser = ArgumentParser()
    parser.add_argument('--image', '-i', help='validation dataset image folder', type=str)
    parser.add_argument('--groundtruth', '-g', help='validation dataset ground truth', type=str)
    parser.add_argument('--output', '-o', help='output folder for prediction results', type=str)
    parser.add_argument('--checkpoint', '-c', help='model checkpoint path', type=str)
    args = parser.parse_args()
    return Args(**{k: Path(v) for k, v in vars(args).items()})

def get_logger(file_path: Path) -> logging.Logger:
    log_format = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger(__file__)
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler(stdout)
    console_handler.setFormatter(log_format)
    file_handler = logging.FileHandler(file_path, mode='a', encoding='utf-8')
    file_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger

if __name__ == "__main__":
    ''' prepare the pre-trained model with local path or huggingface url '''
    args = parse_args()
    model = medim.create_model("SAM-Med3D", pretrained=True, checkpoint_path=str(args.checkpoint))
    img_files = set([x.name.replace('_0000.nii.gz', '') for x in args.image.glob("*.nii.gz")])
    gt_files = set([x.name.replace('.nii.gz', '') for x in args.groundtruth.glob('*.nii.gz')])
    logger = get_logger(Path('medim_val.log'))
    for case_name in tqdm(sorted(list(img_files & gt_files))):
        print(f'[working] {case_name}')
        img_path = args.image.joinpath(f"{case_name}_0000.nii.gz")
        gt_path = args.groundtruth.joinpath(f"{case_name}.nii.gz")
        out_path = args.output.joinpath(f"{case_name}.nii.gz")
        try:
            validate_paired_img_gt(model, img_path, gt_path, out_path, num_clicks=1)
        except RuntimeError as e:
            logger.error(f'[error] {case_name} fail for:\n\t{str(e)}')
