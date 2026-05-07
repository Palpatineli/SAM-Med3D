''' Example-1: manually list all dataset paths '''
img_datas = [
'/mnt/weka/data/nnUNet/sam/stir/nf1/mri_iso/',
]
''' Example-2: use glob to automatically list all dataset paths '''
# import os.path as osp
# from glob import glob
#
# PROJ_DIR = osp.dirname(osp.dirname(__file__))
# img_datas = glob(osp.join(PROJ_DIR, "data", "brain_pre_sam", "*", "*"))
