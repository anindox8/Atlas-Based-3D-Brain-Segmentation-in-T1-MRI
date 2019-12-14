# Libraries
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from prime_aux import computeAtlasProb, computeTissueModels, labelPropg, all_labelPropg
from em_aux import dice_metric
from em import segmentEM


# Generate Probabilistic Atlases from Training Data
prob_atlas_CSF, prob_atlas_WM, prob_atlas_GM  = computeAtlasProb(export='return')

# Load Tissue Models
CSF = np.load('../results/atlas/tissueModel_CSF.npy')
WM  = np.load('../results/atlas/tissueModel_WM.npy')
GM  = np.load('../results/atlas/tissueModel_GM.npy')

# Visualize Independent Tissue Models
plt.figure()
plt.hist(CSF, bins=2000, alpha=0.75, label='CSF', color='crimson',      range=(0,2000), density=True)
plt.hist(WM,  bins=2000, alpha=0.75, label='WM',  color='midnightblue', range=(0,2000), density=True)
plt.hist(GM,  bins=2000, alpha=0.75, label='GM',  color='gold',         range=(0,2000), density=True)
plt.title('Independent Tissue Models')
plt.xlabel('Intensity Values')
plt.ylabel('Frequency')
plt.legend(loc='upper right', fontsize='x-large')
plt.show()

# Visualize Functional Tissue Models
iTP_CSF      = np.histogram(CSF, bins=5000, range=(0,5000))
iTP_WM       = np.histogram(WM,  bins=5000, range=(0,5000))
iTP_GM       = np.histogram(GM,  bins=5000, range=(0,5000))

CSF_bins     = np.nan_to_num(iTP_CSF[0]/(iTP_CSF[0] + iTP_WM[0] + iTP_GM[0]))
WM_bins      = np.nan_to_num(iTP_WM[0]/(iTP_CSF[0]  + iTP_WM[0] + iTP_GM[0]))
GM_bins      = np.nan_to_num(iTP_GM[0]/(iTP_CSF[0]  + iTP_WM[0] + iTP_GM[0]))

fTP_CSF      = (CSF_bins, iTP_CSF[1])
fTP_WM       = (WM_bins,  iTP_WM[1])
fTP_GM       = (GM_bins,  iTP_GM[1])

plt.figure()
plt.bar((fTP_CSF[1])[:-1],fTP_CSF[0], width=1, alpha=0.75, label='CSF', color='crimson')
plt.bar((fTP_WM[1])[:-1], fTP_WM[0],  width=1, alpha=0.75, label='WM',  color='midnightblue')
plt.bar((fTP_GM[1])[:-1], fTP_GM[0],  width=1, alpha=0.75, label='GM',  color='gold')
plt.xlabel('Intensity Values')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.show()


index = '038'

# Segmentation via Training Image-Based Probabilistic Atlas
predicted_mask00 = labelPropg(CSF="../results/testing_results/transformed_labels/CSF/"+index+"/result.mhd", WM="../results/testing_results/transformed_labels/WM/"+index+"/result.mhd", 
                              GM="../results/testing_results/transformed_labels/GM/"+index+"/result.mhd",   mask="../data/testing-set/testing-mask/1"+index+"_1C.nii.gz",
                              mode='prob_atlas', export='return')

# Segmentation via MNI Probabilistic Atlas
predicted_mask01 = labelPropg(CSF="../results/testing_results/transformed_labels_MNI/CSF/"+index+"/result.mhd", WM="../results/testing_results/transformed_labels_MNI/WM/"+index+"/result.mhd", 
                              GM="../results/testing_results/transformed_labels_MNI/GM/"+index+"/result.mhd",   mask="../data/testing-set/testing-mask/1"+index+"_1C.nii.gz",
                              mode="prob_atlas", export='return')


# Segmentation via Training Image-Based Probability Atlas
predicted_mask10 = labelPropg(CSF="../results/testing_results/transformed_labels/CSF/"+index+"/result.mhd", WM="../results/testing_results/transformed_labels/WM/"+index+"/result.mhd", 
                              GM="../results/testing_results/transformed_labels/GM/"+index+"/result.mhd",   mask="../data/testing-set/testing-mask/1"+index+"_1C.nii.gz",
                              mode="prob_inten_atlas", export="return")

# Segmentation via MNI Probabilistic Atlas
predicted_mask11 = labelPropg(CSF="../results/testing_results/transformed_labels_MNI/CSF/"+index+"/result.mhd", WM="../results/testing_results/transformed_labels_MNI/WM/"+index+"/result.mhd", 
                              GM="../results/testing_results/transformed_labels_MNI/GM/"+index+"/result.mhd",   mask="../data/testing-set/testing-mask/1"+index+"_1C.nii.gz",
                              mode="prob_inten_atlas", export="return")

# Segmentation via Expectation-Maximization with K-Means Initialization
mask1, score1   = segmentEM(volume_dir="../data/testing-set/testing-images/1"+index+".nii.gz", labels_dir="../data/testing-set/testing-labels/1"+index+"_3C.nii.gz",
                            mask_dir="../data/testing-set/testing-mask/1"+index+"_1C.nii.gz", init_mode="kmeans", atlas=None, mode="base", export="return")

# Segmentation via Expectation-Maximization with Probabilistic Atlas Initialization
mask2, score2   = segmentEM(volume_dir="../data/testing-set/testing-images/1"+index+".nii.gz", labels_dir="../data/testing-set/testing-labels/1"+index+"_3C.nii.gz",
                            mask_dir="../data/testing-set/testing-mask/1"+index+"_1C.nii.gz", init_mode="atlas", atlas='training', mode="base", export="return")

# Segmentation via Expectation-Maximization with Probabilistic Atlas Initialization and Late Fusion
mask3, score3   = segmentEM(volume_dir="../data/testing-set/testing-images/1"+index+".nii.gz", labels_dir="../data/testing-set/testing-labels/1"+index+"_3C.nii.gz",
                            mask_dir="../data/testing-set/testing-mask/1"+index+"_1C.nii.gz", init_mode="atlas", atlas='training', mode="atlas", export="return")

# Segmentation via Expectation-Maximization with MNI Atlas Initialization
mask4, score4   = segmentEM(volume_dir="../data/testing-set/testing-images/1"+index+".nii.gz", labels_dir="../data/testing-set/testing-labels/1"+index+"_3C.nii.gz",
                            mask_dir="../data/testing-set/testing-mask/1"+index+"_1C.nii.gz", init_mode="atlas", atlas='MNI', mode="base", export="return")

# Segmentation via Expectation-Maximization with MNI Atlas Initialization and Late Fusion
mask5, score5   = segmentEM(volume_dir="../data/testing-set/testing-images/1"+index+".nii.gz", labels_dir="../data/testing-set/testing-labels/1"+index+"_3C.nii.gz",
                            mask_dir="../data/testing-set/testing-mask/1"+index+"_1C.nii.gz", init_mode="atlas", atlas='MNI', mode="atlas", export="return")


#%% EXECUTE ONE-TIME TO STORE ORIGINALS

### PROBABILITY ATLASES
computeAtlasProb(export='save')

### TISSUE MODELS
computeTissueModels(export='save')

### LABEL PROPAGATION VIA PROBABILITY ATLASES
all_labelPropg(mode='prob_atlas')

### LABEL PROPAGATION VIA PROBABILITY ATLASES + TISSUE MODELS
all_labelPropg(mode='prob_inten_atlas')

