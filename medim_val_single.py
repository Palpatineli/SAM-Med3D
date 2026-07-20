# -*- encoding: utf-8 -*-

from typing import NamedTuple
from argparse import ArgumentParser
from pathlib import Path
import medim

from utils.infer_utils import validate_paired_img_gt
from utils.metric_utils import compute_metrics, print_computed_metrics

class Args(NamedTuple):
    image: Path
    groundtruth: Path
    output: Path
    checkpoint: Path

def parse_args() -> Args:
    parser = ArgumentParser()
    parser.add_argument('--image', '-i', help='validation images', type=str)
    parser.add_argument('--groundtruth', '-g', help='validation ground truth label', type=str)
    parser.add_argument('--output', '-o', help='output label file path prediction results', type=str)
    parser.add_argument('--checkpoint', '-c', help='checkpoint path', type=str)
    args = parser.parse_args()
    return Args(**{k: Path(v) for k, v in vars(args).items()})

if __name__ == "__main__":
    args = parse_args()
    model = medim.create_model("SAM-Med3D", pretrained=True, checkpoint_path=str(args.checkpoint))

    print("Validation start! plz wait for some times.")
    validate_paired_img_gt(model, args.image, args.groundtruth, args.output, num_clicks=1)
    print("Validation finish! plz check your prediction.")

    ''' 4. compute the metrics of your prediction with the ground truth '''
    metrics = compute_metrics(str(args.groundtruth), str(args.output), None, ['dice'])
    print_computed_metrics(metrics)
