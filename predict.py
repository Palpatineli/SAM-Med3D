# %%
from argparse import ArgumentParser
import copy
import enum
from pathlib import Path
import numpy as np
import torch
import torchio as tio
import medim
from utils.infer_utils import read_arr_from_nifti, data_postprocess, sam_model_infer, get_roi_from_subject, save_numpy_to_nifti
from utils.infer_utils import get_category_list_and_zero_mask, get_subject_and_meta_info, data_preprocess
seed = 233
_ = torch.manual_seed(seed)
np.random.seed(seed)


ckpt_path = "https://huggingface.co/blueyo0/SAM-Med3D/blob/main/sam_med3d_turbo.pth"
model = medim.create_model("SAM-Med3D", pretrained=True, checkpoint_path=ckpt_path)

def parse() -> tuple[Path, Path, Path]:
    parser = ArgumentParser('sammed3d')
    _ = parser.add_argument('input', type=str)
    _ = parser.add_argument('output', type=str)
    _ = parser.add_argument('--prev_mask', '-m', type=str, default='')
    args = parser.parse_args()
    assert (input_path := Path(args.input)).exists()
    assert (output_path := Path(args.output)).is_dir()
    return input_path, output_path, Path(args.prev_mask)

def img_only_preprocess(subject: tio.Subject, meta_info, target_spacing: tuple[float, float, float],
                        crop_size: int = 128):
    meta_info["original_subject_affine"] = subject.image.affine.copy()
    meta_info["original_subject_spatial_shape"] = subject.image.spatial_shape # D, H, W

    # step-1: resample online
    resampler = tio.Resample(target=target_spacing)
    subject_resampled = resampler(subject)

    # step-2: canonicalize
    transform_canonical = tio.ToCanonical()
    subject_canonical = transform_canonical(subject_resampled)

    # step-3: try to crop or pad roi region (with normalization)
    crop_transform = tio.CropOrPad(mask_name='label', target_shape=(crop_size, crop_size, crop_size))
    norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)
    roi_image, roi_label, meta_info = get_roi_from_subject(
        subject_canonical, meta_info, crop_transform, norm_transform
    )
    # roi image/label is (1, 1, D, H, W) after normalization
    return roi_image, None, meta_info

def _remove_channel(file_name: str) -> str:
    assert file_name.endswith('.nii.gz')
    if len(file_name) > 12 and file_name[-12] == '_':
        try:
            _ = int(file_name[-11: -7])
        except ValueError:
            return file_name
        return file_name[:-12] + '.nii.gz'
    return file_name
# %%
class PredictMode(enum.Enum):
    Mask = enum.auto()
    BBox = enum.auto()
    Point = enum.auto()

def refine(input_folder: Path, output_path: Path, prev_mask_folder: Path, mode: PredictMode):
    # input_folder = Path('/mnt/d/data/nnUNet_raw/Dataset550NF1CorStirExpanded/imagesTs/146_0000.nii.gz')
    # prev_mask_folder = Path('/mnt/d/data/predict/550test/550resEncMAnisoNoMirror25/')
    # output_path = Path('/mnt/d/data/predict/samMed3d/550refine')
    target_spacing=(1.5, 1.5, 1.5)
    crop_size=128
    input_paths = list(input_folder.glob('*.nii.gz')) if input_folder.is_dir() else [input_folder]
    if prev_mask_folder.is_file():
        prev_mask_paths = [prev_mask_folder]
    else:
        prev_mask_paths = []
        for img_path in input_paths:
            prev_mask_file = prev_mask_folder.joinpath(_remove_channel(img_path.name))
            if not prev_mask_file.exists():
                raise FileNotFoundError(f"prev_mask_file not found at {prev_mask_file}.")
            prev_mask_paths.append(prev_mask_file)
    output_path.mkdir(exist_ok=True)

    for img_path, prev_mask_path in zip(input_paths, prev_mask_paths):
        # img_path, prev_mask_path = next(iter(zip(input_paths, prev_mask_paths)))
        print(f"[refine prediction] {img_path} {prev_mask_path}")
        exist_categories, final_pred_numpy_original_grid = get_category_list_and_zero_mask(prev_mask_path)
        _, gt_meta_for_saving = read_arr_from_nifti(prev_mask_path, get_meta_info=True)
        subject, meta_info = get_subject_and_meta_info(img_path, prev_mask_path)

        for category_index in exist_categories:
            # category_index = next(iter(exist_categories))
            category_specific_subject = copy.deepcopy(subject)
            category_specific_meta_info = copy.deepcopy(meta_info)
            roi_image, roi_label, meta_info = data_preprocess(category_specific_subject,
                                                              category_specific_meta_info,
                                                              category_index=category_index,
                                                              target_spacing=target_spacing,
                                                              crop_size=crop_size)

            roi_pred_numpy, _ = sam_model_infer(model, roi_image, roi_gt=roi_label, 
                                                num_clicks=1,
                                                prev_low_res_mask=None)

            cls_pred_original_grid = data_postprocess(roi_pred_numpy, meta_info)
            final_pred_numpy_original_grid[cls_pred_original_grid == category_index] = category_index

        # Save the combined prediction which is on the original GT's grid
        save_numpy_to_nifti(final_pred_numpy_original_grid, output_path.joinpath(img_path.name), gt_meta_for_saving)

# %%
def raw_predict(input_path: Path, output_path: Path):
    # input_path = Path('/mnt/d/data/nnUNet_raw/Dataset550NF1CorStirExpanded/imagesTs/088_0000.nii.gz')
    # output_path = Path('/mnt/d/data/predict/samMed3d/550')
    target_spacing=(1.5, 1.5, 1.5)
    crop_size=128
    input_paths = input_path.glob('*.nii.gz') if input_path.is_dir() else [input_path]
    output_path.mkdir(exist_ok=True)

    for img_path in input_paths:
        # img_path = next(iter(input_paths))
        print(f"[raw prediction] {img_path}")
        _, meta_info = read_arr_from_nifti(img_path, get_meta_info=True)
        final_pred_numpy_original_grid = np.zeros(meta_info["original_numpy_shape"], dtype=np.uint8)
        subject = tio.Subject(image=tio.ScalarImage(img_path))

        for category_index in range(1, 3):
            # category_index = next(iter(gt_fg_labels))
            category_specific_subject = copy.deepcopy(subject)
            category_specific_meta_info = copy.deepcopy(meta_info)
            # roi_image is (1,1,D,H,W), roi_label is (1,1,D,H,W)
            # meta_info contains all necessary affines and shapes
            roi_image, _, meta_info = img_only_preprocess(category_specific_subject, category_specific_meta_info, target_spacing, crop_size)

            roi_pred_numpy, _ = sam_model_infer(model, roi_image, num_clicks=1, prev_low_res_mask=None)

            cls_pred_original_grid = data_postprocess(roi_pred_numpy, meta_info)
            final_pred_numpy_original_grid[cls_pred_original_grid == 1] = category_index
        save_numpy_to_nifti(final_pred_numpy_original_grid, output_path.joinpath(_remove_channel(img_path.name)), meta_info)

def _has_file(folder: Path, ext: str) -> bool:
    if not folder.exists() or not folder.is_dir():
        return False
    return any(f for f in folder.iterdir() if f.name.endswith(ext))

def main(input_folder: Path, output_path: Path, prev_mask_folder: Path, num_clicks: int = 1):
    if prev_mask_folder.exists() and (prev_mask_folder.name.endswith('.nii.gz') or _has_file(prev_mask_folder, '.nii.gz')):
        refine(input_folder, output_path, prev_mask_folder, num_clicks)
    else:
        raw_predict(input_folder, output_path)

# %%

if __name__ == "__main__":
    main(*parse())
