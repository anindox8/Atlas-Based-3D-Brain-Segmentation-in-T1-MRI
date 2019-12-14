# Atlas-Based Segmentation Algorithms in T1 MRI

**Problem Statement**: Fully supervised, multi-class 3D brain segmentation in T1 MRI.

**Data**: *Label 1*: Cerebrospinal Fluid (CSF); *Label 2:* White Matter (WM); *Label 3:* Gray Matter (GM). 


**Directories**  
  ● Convert DICOM to NIfTI Volumes: `preprocess/prime/DICOM_NIFTI.py`  
  ● Resample NIfTI Volume Resolutions: `preprocess/prime/resampleRes.py`  
  ● Infer StFA/DyFA Segmentation Sub-Model (DenseVNet): `python net_segment.py inference -c '../config.ini'`  
  ● Preprocess Full Dataset to Optimized I/O HDF5 Training Patch-Volumes: `preprocess/prime/preprocess_alpha.py`  
  ● Preprocess Full Dataset to Optimized I/O HDF5 Deployment Whole-Volumes: `preprocess/prime/preprocess_deploy.py`  
  ● Generate Data-Directory Feeder List: `feed/prime/feed_metadata.py`  
  ● Train StFA Classification Sub-Model: `train/prime/train_StFA.py`  
  ● Train DyFA Classification Sub-Model: `train/prime/train_DyFA.py`  
  ● Deploy Model (Validation): `deploy/prime/deployBinary.py`  
  ● Average Predictions for Same Patient: `deploy/prime/average_predictions.py`  
  ● Calculate AUC: `notebooks/binary_AUC.ipynb`
  

## Network Architecture  
  
  
![Network Architecture](reports/images/reg00.png)*Figure 1.  [left-to-right]: T1 MRI volume of the reference image (patient1000), the movingimage (patient1006) and the registered image (patient1006) using 3D affine and B-splineregistration with gradient descent optimizer and Mattes mutual information metric (optimizedfrom -0.355940 to -0.739491). Blue cross-hairs mark the same voxel across all 3 volumes at slice135, verifying successful registration.*  
  
    
    
## Multi-Resolution Deep Segmentation Features  
  
  
![Multi-Resolution Deep Segmentation Features](reports/images/reg02.png)*Figure 2.  [left-to-right]: Mean segmentation ground truth for all 15 registered training labels,the CSF probabilistic atlas, the WM probabilistic atlas and the GM probabilistic atlas. Bluecross-hairs mark the same voxel across all 4 volumes at slice 135 with 0.13335, 0.06665, 0.80000probabilities of belong to the CSF, WM and GM classes, respectively.*  
  
    
    
## Experimental Results  
  
  
![Binary AUC](reports/images/reg03.png)*Figure 3.  [left-to-right]: Independent tissue models (where the area under each curve is equal to1) and functional tissue models (where sum of the normalized frequency values/probabilities forall 3 classes at any given intensity value is equal to 1). Anomalous values appear in the functionaltissue models after the intensity value 2000, owing to division by very small probabilities values.*



## Experimental Results  
  
  
![Binary AUC](reports/images/reg03.png)*Figure 3.  [left-to-right]: Independent tissue models (where the area under each curve is equal to1) and functional tissue models (where sum of the normalized frequency values/probabilities forall 3 classes at any given intensity value is equal to 1). Anomalous values appear in the functionaltissue models after the intensity value 2000, owing to division by very small probabilities values.*
