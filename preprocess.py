# %%
from pathlib import Path
import torchio as tio
import numpy as np
import numpy.typing as npt

from_root = Path(r'')
to_root = Path(r'')
img_size = 128
# %%
img_folder = from_root.joinpath('imagesTr')
label_folder = from_root.joinpath('labelsTr')
for img_file in img_folder.glob("*_0000.nii.gz"):
    img_id = img_file.name.rsplit('_', 1)[0]
    label_file = label_folder.joinpath(f'{img_id}.nii.gz')
    if not label_file.exists():
        print(f"[label not found] {img_id}")
        continue

# %%
def run_preprocessing(
    from_dir: str, 
    to_dir: str, 
    img_size: tuple[int, int, int] | npt.NDArray[np.int_] = (128, 128, 128), 
    target_spacing: tuple[float, float, float] | npt.NDArray[np.float64] = (1.5, 1.5, 1.5),
    overlap_ratio: float = 0.5
):
    # paths
    base_from = Path(from_dir)
    base_to = Path(to_dir)
    img_from_dir = base_from.joinpath("imagesTr")
    lbl_from_dir = base_from.joinpath("labelsTr")
    img_to_dir = base_to.joinpath("imagesTr")
    lbl_to_dir = base_to.joinpath("labelsTr")
    img_to_dir.mkdir(parents=True, exist_ok=True)
    lbl_to_dir.mkdir(parents=True, exist_ok=True)

    # standard transforms
    target_spacing = np.array(target_spacing)
    preprocessing_transform = tio.Compose([
        tio.ToCanonical(), 
        tio.Resample(target_spacing)  # pyright: ignore[reportArgumentType]
    ])

    # 3. Calculate absolute overlap
    img_size = np.array(img_size)
    patch_overlap = np.ceil(img_size * overlap_ratio)

    print(f"Patch Size: {img_size}\nOverlap (voxels): {patch_overlap}")

    for img_path in img_from_dir.glob('*.nii.gz'):
        filename = img_path.name
        label_path = lbl_from_dir / filename
        if not label_path.exists():
            print(f"[Warning] No matching label found for {filename}. Skipping.")
            continue

        print(f"\nProcessing: {filename}...")
        subject = tio.Subject(
            image=tio.ScalarImage(img_path),
            label=tio.LabelMap(label_path),
            name=filename.replace('.nii.gz', '')
        )

        subject = preprocessing_transform(subject)

        # minimal padding
        padding = np.max(0, img_size - subject.spatial_shape)

        if any(padding > 0):
            padding = np.vstack([padding // 2, padding - padding // 2]).T.ravel()
            subject = tio.Pad(list(padding))(subject)

        grid_sampler = tio.GridSampler(
            subject,
            patch_size=img_size,
            patch_overlap=patch_overlap
        )

        # 6. Extract and Save Patches
        print(f"  Extracting {len(grid_sampler)} patches...")
        for i, patch_subject in enumerate(grid_sampler):
            # Format: original_file_name_001.nii.gz
            patch_id = f"{i:03d}" 
            base_name = patch_subject['name']
            out_img_name = f"{base_name}_{patch_id}.nii.gz"
            out_lbl_name = f"{base_name}_{patch_id}.nii.gz"
            # Save paths
            out_img_path = img_to_dir / out_img_name
            out_lbl_path = lbl_to_dir / out_lbl_name
            # Save images and labels
            patch_subject['image'].save(out_img_path)
            patch_subject['label'].save(out_lbl_path)

    print("\nPreprocessing complete!")

# --- Execute the script ---
if __name__ == "__main__":
    run_preprocessing(
        from_dir="./from",    # Folder containing /imagesTr and /labelsTr
        to_dir="./to",        # Output folder
        img_size=128,         # 128x128x128 patches
        target_spacing=1.5,   # 1.5mm isotropic
        overlap_ratio=0.51    # slightly > 0.5 as requested
    )
