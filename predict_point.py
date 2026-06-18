from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchio as tio

from segment_anything.build_sam3D import sam_model_registry3D
from utils.infer_utils import (
    data_postprocess,
    get_roi_from_subject,
    read_arr_from_nifti,
    save_numpy_to_nifti,
)


DEFAULT_CHECKPOINT = "work_dir/union_train/sam_model_latest.pth"


def parse_args():
    parser = ArgumentParser(
        description="Run SAM-Med3D on one NIfTI image from one positive 3D point prompt."
    )
    parser.add_argument("image", type=Path, help="Input image path (*.nii.gz).")
    parser.add_argument("output", type=Path, help="Output label path (*.nii.gz).")
    parser.add_argument(
        "--point",
        type=float,
        nargs=3,
        required=True,
        metavar=("A", "B", "C"),
        help="Prompt point. Default interpretation is voxel z y x.",
    )
    parser.add_argument(
        "--point-space",
        choices=("voxel-zyx", "voxel-xyz", "world"),
        default="voxel-zyx",
        help=(
            "Coordinate system for --point. voxel-zyx matches SimpleITK/numpy "
            "array order; voxel-xyz matches TorchIO image index order; world "
            "uses the TorchIO affine world coordinate system."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(DEFAULT_CHECKPOINT),
        help="Fine-tuned checkpoint path.",
    )
    parser.add_argument(
        "--model-type",
        choices=sorted(sam_model_registry3D.keys()),
        default="vit_b_ori",
        help="SAM-Med3D architecture used for the checkpoint.",
    )
    parser.add_argument(
        "--target-spacing",
        type=float,
        nargs=3,
        default=(1.5, 1.5, 1.5),
        metavar=("SX", "SY", "SZ"),
        help="TorchIO resampling spacing used before inference.",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=128,
        help="Cubic ROI size used before inference.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device, e.g. cuda or cpu.",
    )
    parser.add_argument(
        "--allow-partial-weight",
        action="store_true",
        help="Load checkpoint with strict=False.",
    )
    parser.add_argument(
        "--label-value",
        type=np.uint8,
        default=1,
        help="Foreground label value written in the output mask.",
    )
    return parser.parse_args()


def load_refined_model(checkpoint_path: Path, model_type: str, device: str, strict: bool):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    state_dict = {
        key.removeprefix("module."): value for key, value in state_dict.items()
    }

    model = sam_model_registry3D[model_type](None)
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if missing or unexpected:
        print(f"Loaded with missing keys: {missing}")
        print(f"Loaded with unexpected keys: {unexpected}")
    return model.to(device).eval()


def point_to_torchio_index(image: tio.ScalarImage, point, point_space: str) -> tuple[int, int, int]:
    point = np.asarray(point, dtype=float)
    if point_space == "voxel-zyx":
        index = point[[2, 1, 0]]
    elif point_space == "voxel-xyz":
        index = point
    else:
        homogeneous = np.append(point, 1.0)
        index = np.linalg.inv(image.affine) @ homogeneous
        index = index[:3]

    index = tuple(int(round(value)) for value in index)
    spatial_shape = image.spatial_shape
    if any(i < 0 or i >= size for i, size in zip(index, spatial_shape)):
        raise ValueError(
            f"Point index {index} is outside image spatial shape {spatial_shape}."
        )
    return index


def preprocess_image_and_point(
    image_path: Path,
    point,
    point_space: str,
    meta_info: dict,
    target_spacing: tuple[float, float, float],
    crop_size: int,
):
    image = tio.ScalarImage(image_path)
    point_index = point_to_torchio_index(image, point, point_space)

    point_mask = torch.zeros((1, *image.spatial_shape), dtype=torch.uint8)
    point_mask[(0, *point_index)] = 1

    subject = tio.Subject(
        image=image,
        label=tio.LabelMap(tensor=point_mask, affine=image.affine),
    )
    meta_info["original_subject_affine"] = subject.image.affine.copy()
    meta_info["original_subject_spatial_shape"] = subject.image.spatial_shape

    subject = tio.Resample(target=target_spacing)(subject)
    subject = tio.ToCanonical()(subject)

    crop_shape = (crop_size, crop_size, crop_size)
    crop_transform = tio.CropOrPad(mask_name="label", target_shape=crop_shape)
    norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)
    roi_image, roi_point_mask, meta_info = get_roi_from_subject(
        subject, meta_info, crop_transform, norm_transform
    )

    point_indices = torch.argwhere(roi_point_mask[0, 0] > 0)
    if len(point_indices) == 0:
        raise RuntimeError(
            "The prompt point was lost during preprocessing. Try voxel coordinates "
            "or a point farther from the image boundary."
        )
    roi_point = point_indices.float().mean(dim=0).round().reshape(1, 1, 3)
    return roi_image, roi_point, meta_info


def infer_from_point(model, roi_image, roi_point, device: str):
    with torch.no_grad():
        input_tensor = roi_image.to(device)
        image_embeddings = model.image_encoder(input_tensor)

        points_coords = roi_point.to(device=device, dtype=torch.float32)
        points_labels = torch.ones((1, 1), device=device, dtype=torch.int64)
        low_res_shape = tuple(max(1, size // 4) for size in roi_image.shape[-3:])
        low_res_mask = torch.zeros(
            (1, 1, *low_res_shape), device=device, dtype=torch.float32
        )

        sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points=[points_coords, points_labels],
            boxes=None,
            masks=low_res_mask,
        )
        low_res_masks, _ = model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        masks = F.interpolate(
            low_res_masks,
            size=roi_image.shape[-3:],
            mode="trilinear",
            align_corners=False,
        )
    return (torch.sigmoid(masks).cpu().numpy().squeeze() > 0.5).astype(np.uint8)


def main():
    args = parse_args()
    if not args.image.exists():
        raise FileNotFoundError(f"Input image not found: {args.image}")
    if not args.image.name.endswith(".nii.gz"):
        raise ValueError(f"Input image must be a .nii.gz file: {args.image}")
    if not args.output.name.endswith(".nii.gz"):
        raise ValueError(f"Output path must end with .nii.gz: {args.output}")

    torch.manual_seed(233)
    np.random.seed(233)

    _, meta_info = read_arr_from_nifti(args.image, get_meta_info=True)
    model = load_refined_model(
        args.checkpoint,
        args.model_type,
        args.device,
        strict=not args.allow_partial_weight,
    )
    roi_image, roi_point, meta_info = preprocess_image_and_point(
        args.image,
        args.point,
        args.point_space,
        meta_info,
        tuple(args.target_spacing),
        args.crop_size,
    )
    roi_pred = infer_from_point(model, roi_image, roi_point, args.device)
    pred = data_postprocess(roi_pred, meta_info)
    pred = (pred > 0).astype(np.uint8) * args.label_value

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_numpy_to_nifti(pred, args.output, meta_info)
    print(f"Wrote prediction to {args.output}")


if __name__ == "__main__":
    main()
