from argparse import ArgumentParser
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchio as tio

from segment_anything.build_sam3D import sam_model_registry3D
from utils.infer_utils import (
    data_postprocess,
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
    parser.add_argument(
        "--neighbor-overlap",
        type=float,
        default=0.5,
        help="Fractional overlap between border-triggered neighboring patches.",
    )
    parser.add_argument(
        "--border-width",
        type=int,
        default=1,
        help="Number of voxels at each patch face used to detect border contact.",
    )
    parser.add_argument(
        "--max-patches",
        type=int,
        default=64,
        help="Maximum number of patches to infer. Use 1 for the old single-patch behavior.",
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
    meta_info["canonical_subject_shape"] = subject.spatial_shape
    meta_info["canonical_subject_affine"] = subject.image.affine.copy()
    meta_info["roi_subject_affine"] = subject.image.affine.copy()

    point_indices = torch.argwhere(subject.label.data[0] > 0)
    if len(point_indices) == 0:
        raise RuntimeError(
            "The prompt point was lost during preprocessing. Try voxel coordinates "
            "or a point farther from the image boundary."
        )
    prompt_point = point_indices.float().mean(dim=0).round().to(torch.int64)
    return subject.image.data.clone().detach(), prompt_point, meta_info


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


def extract_patch(
    image_data: torch.Tensor,
    center: tuple[int, int, int],
    patch_size: int,
) -> tuple[
    torch.Tensor,
    tuple[int, int, int],
    tuple[slice, slice, slice],
    tuple[slice, slice, slice],
]:
    spatial_shape = image_data.shape[-3:]
    starts = tuple(center[axis] - patch_size // 2 for axis in range(3))
    ends = tuple(start + patch_size for start in starts)
    src_starts = tuple(max(0, start) for start in starts)
    src_ends = tuple(min(spatial_shape[axis], ends[axis]) for axis in range(3))
    dst_starts = tuple(max(0, -start) for start in starts)
    dst_ends = tuple(
        dst_starts[axis] + src_ends[axis] - src_starts[axis] for axis in range(3)
    )

    patch = torch.zeros((1, patch_size, patch_size, patch_size), dtype=image_data.dtype)
    patch[
        :,
        dst_starts[0]:dst_ends[0],
        dst_starts[1]:dst_ends[1],
        dst_starts[2]:dst_ends[2],
    ] = image_data[
        :,
        src_starts[0]:src_ends[0],
        src_starts[1]:src_ends[1],
        src_starts[2]:src_ends[2],
    ]
    source_slices = tuple(slice(src_starts[axis], src_ends[axis]) for axis in range(3))
    valid_slices = tuple(slice(dst_starts[axis], dst_ends[axis]) for axis in range(3))
    return patch, starts, source_slices, valid_slices


def normalize_patch(patch: torch.Tensor) -> torch.Tensor:
    norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)
    normalized = norm_transform(patch)
    if torch.isnan(normalized).any():
        normalized = torch.nan_to_num(normalized)
    return normalized.unsqueeze(0)


def clamp_center(center: tuple[int, int, int], spatial_shape) -> tuple[int, int, int]:
    return tuple(
        min(max(0, int(center[axis])), spatial_shape[axis] - 1) for axis in range(3)
    )


def border_triggered_neighbors(
    patch_pred: np.ndarray,
    center: tuple[int, int, int],
    starts: tuple[int, int, int],
    spatial_shape,
    patch_size: int,
    step: int,
    border_width: int,
) -> list[tuple[tuple[int, int, int], torch.Tensor]]:
    neighbors = []
    for axis in range(3):
        for direction, border_slice in (
            (-1, slice(0, border_width)),
            (1, slice(patch_size - border_width, patch_size)),
        ):
            slicer = [slice(None), slice(None), slice(None)]
            slicer[axis] = border_slice
            face = patch_pred[tuple(slicer)] > 0
            if not face.any():
                continue

            new_center = list(center)
            new_center[axis] += direction * step
            new_center = clamp_center(tuple(new_center), spatial_shape)
            if new_center == center:
                continue

            face_coords = np.argwhere(face)
            face_origin = [0, 0, 0]
            face_origin[axis] = border_slice.start
            local_seed = np.round(face_coords.mean(axis=0)).astype(int)
            local_seed[axis] += face_origin[axis]
            global_seed = torch.tensor(
                [starts[i] + int(local_seed[i]) for i in range(3)], dtype=torch.int64
            )
            neighbors.append((new_center, global_seed))
    return neighbors


def infer_with_border_expansion(
    model,
    image_data: torch.Tensor,
    initial_point: torch.Tensor,
    crop_size: int,
    device: str,
    max_patches: int,
    neighbor_overlap: float,
    border_width: int,
):
    if not 0 <= neighbor_overlap < 1:
        raise ValueError("--neighbor-overlap must be >= 0 and < 1.")
    if max_patches < 1:
        raise ValueError("--max-patches must be at least 1.")
    if border_width < 1 or border_width > crop_size:
        raise ValueError("--border-width must be between 1 and --crop-size.")

    spatial_shape = image_data.shape[-3:]
    step = max(1, int(round(crop_size * (1.0 - neighbor_overlap))))
    initial_center = clamp_center(tuple(int(v) for v in initial_point.tolist()), spatial_shape)

    stitched = np.zeros(tuple(spatial_shape), dtype=np.uint8)
    queue = deque([(initial_center, initial_point.clone().detach())])
    queued_or_done = {initial_center}
    inferred = 0

    while queue and inferred < max_patches:
        center, prompt_global = queue.popleft()
        patch, starts, source_slices, valid_slices = extract_patch(
            image_data, center, crop_size
        )
        prompt_local = torch.tensor(
            [[[
                min(max(0, int(prompt_global[axis]) - starts[axis]), crop_size - 1)
                for axis in range(3)
            ]]],
            dtype=torch.float32,
        )
        roi_image = normalize_patch(patch)
        patch_pred = infer_from_point(model, roi_image, prompt_local, device)

        stitched[source_slices] = np.maximum(
            stitched[source_slices], patch_pred[valid_slices]
        )
        inferred += 1

        for neighbor_center, neighbor_seed in border_triggered_neighbors(
            patch_pred,
            center,
            starts,
            spatial_shape,
            crop_size,
            step,
            border_width,
        ):
            if neighbor_center in queued_or_done or len(queue) + inferred >= max_patches:
                continue
            queued_or_done.add(neighbor_center)
            queue.append((neighbor_center, neighbor_seed))

    if queue:
        print(
            f"Warning: stopped after {inferred} patches with {len(queue)} queued; "
            "increase --max-patches if the object still reaches patch borders."
        )
    else:
        print(f"Inferred {inferred} patch(es).")
    return stitched


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
    image_data, prompt_point, meta_info = preprocess_image_and_point(
        args.image,
        args.point,
        args.point_space,
        meta_info,
        tuple(args.target_spacing),
    )
    canonical_pred = infer_with_border_expansion(
        model,
        image_data,
        prompt_point,
        args.crop_size,
        args.device,
        args.max_patches,
        args.neighbor_overlap,
        args.border_width,
    )
    pred = data_postprocess(canonical_pred, meta_info)
    pred = (pred > 0).astype(np.uint8) * args.label_value

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_numpy_to_nifti(pred, args.output, meta_info)
    print(f"Wrote prediction to {args.output}")


if __name__ == "__main__":
    main()
