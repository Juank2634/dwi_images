# %%
import numpy as np
import os
import glob
import nibabel as nib
import bids
import matplotlib.pyplot as plt

# Denoise correction
from dipy.denoise.patch2self import patch2self

# motion correction
from dipy.align import motion_correction 
from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, save_nifti

# %%
def denoise_correction(image, bvals, bvecs):
    # preprocessing of image
    # Load the DWI image
    data, affine = load_nifti(image)
    bvals, bvecs = read_bvals_bvecs(bvals, bvecs)
    # denoise correction 
    denoise_data = patch2self(data, bvals=bvals, model='ols', shift_intensity=True, clip_negative_vals=False, b0_threshold=50)
    return denoise_data, affine

#%%
def motion_correction(image, bvals, bvecs):
    # Load the DWI image
    data, affine = load_nifti(image)
    bvals, bvecs = read_bvals_bvecs(bvals, bvecs)
    # Motion correction
    corrected_data = motion_correction(data, bvals, bvecs)
    return corrected_data, affine