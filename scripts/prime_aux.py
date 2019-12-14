### LIBRARIES
import numpy as np
import SimpleITK as sitk
import warnings
warnings.filterwarnings("ignore")

def volumeIntenProb(vol,mask):
    # Load Volume/Mask
    volume        = np.array(sitk.GetArrayFromImage(sitk.ReadImage(vol, sitk.sitkFloat32))).astype(np.int16)
    masked_volume = np.multiply(volume,np.array(sitk.GetArrayFromImage(sitk.ReadImage(mask))))
    
    # Load Tissue Models
    CSF           = np.load('../results/atlas/tissueModel_CSF.npy').astype(np.int16)
    WM            = np.load('../results/atlas/tissueModel_WM.npy').astype(np.int16)
    GM            = np.load('../results/atlas/tissueModel_GM.npy').astype(np.int16)
    
    # Store Tissue Model Histograms 
    iTP_CSF      = np.histogram(CSF, bins=5000, range=(0,5000))
    iTP_WM       = np.histogram(WM,  bins=5000, range=(0,5000))
    iTP_GM       = np.histogram(GM,  bins=5000, range=(0,5000))

    CSF_bins     = np.nan_to_num(iTP_CSF[0]/(iTP_CSF[0] + iTP_WM[0] + iTP_GM[0]))
    WM_bins      = np.nan_to_num(iTP_WM[0]/(iTP_CSF[0]  + iTP_WM[0] + iTP_GM[0]))
    GM_bins      = np.nan_to_num(iTP_GM[0]/(iTP_CSF[0]  + iTP_WM[0] + iTP_GM[0]))

    fTP_CSF      = (CSF_bins, iTP_CSF[1])
    fTP_WM       = (WM_bins,  iTP_WM[1])
    fTP_GM       = (GM_bins,  iTP_GM[1])
    
    # Create Intensity Probability Images
    intprob_CSF   = fTP_CSF[0][masked_volume]
    intprob_WM    = fTP_WM[0][masked_volume]
    intprob_GM    = fTP_GM[0][masked_volume]
    
    return intprob_CSF, intprob_WM, intprob_GM


def all_labelPropg(mode='prob_atlas'):
    test_indexes = ['003', '004', '005', '018', '019', '023', '024', '025', '038', '039', 
                    '101', '104', '107', '110', '113', '116', '119', '122', '125', '128']

    if (mode=='prob_atlas'):   
        for index in test_indexes:
            labelPropg(CSF='../results/testing_results/transformed_labels/CSF/'+index+'/result.mhd', WM='../results/testing_results/transformed_labels/WM/'+index+'/result.mhd', 
                       GM='../results/testing_results/transformed_labels/GM/'+index+'/result.mhd',   mask='../data/testing-set/testing-mask/1'+index+'_1C.nii.gz',
                       mode='prob_atlas', export='save')
    elif (mode=='prob_inten_atlas'):   
        for index in test_indexes:
            labelPropg(CSF='../results/testing_results/transformed_labels/CSF/'+index+'/result.mhd', WM='../results/testing_results/transformed_labels/WM/'+index+'/result.mhd', 
                       GM='../results/testing_results/transformed_labels/GM/'+index+'/result.mhd',   mask='../data/testing-set/testing-mask/1'+index+'_1C.nii.gz',
                       mode='prob_inten_atlas', export='save')
    return
        
        
def labelPropg(CSF,WM,GM,mask,export,mode='prob_atlas'):
    if (mode=='prob_atlas'):
        # Load Probability Atlases
        probatlas_CSF    = np.expand_dims(np.array(sitk.GetArrayFromImage(sitk.ReadImage(CSF))),axis=3)
        probatlas_WM     = np.expand_dims(np.array(sitk.GetArrayFromImage(sitk.ReadImage(WM))),axis=3)
        probatlas_GM     = np.expand_dims(np.array(sitk.GetArrayFromImage(sitk.ReadImage(GM))),axis=3)
        bin_mask         = np.array(sitk.GetArrayFromImage(sitk.ReadImage(mask)))
        
        # Assign Label Matching Max Class Probability
        max_index        = np.argmax(np.concatenate((probatlas_CSF,probatlas_WM,probatlas_GM),axis=3),axis=3)+1
        
        # Maskout External Label Predictions
        predicted_mask   = np.multiply(max_index,bin_mask).astype(np.uint8)
        
        if (export=='save'):
            output_prediction        = sitk.GetImageFromArray(predicted_mask)
            output_prediction.CopyInformation(sitk.ReadImage((mask.replace('/testing-mask','/testing-images')).replace('_1C','')))
            writer           = sitk.ImageFileWriter()
            output_dir       = (mask.replace('../data/testing-set/testing-mask','../results/testing_results/predictions/')).replace('_1C.nii.gz','.nii')
            writer.SetFileName(output_dir)
            writer.Execute(output_prediction)
        elif (export=='return'):
            return predicted_mask
    
    elif (mode=='prob_inten_atlas'):
        # Load Probability Atlases
        probatlas_CSF    = np.array(sitk.GetArrayFromImage(sitk.ReadImage(CSF)))
        probatlas_WM     = np.array(sitk.GetArrayFromImage(sitk.ReadImage(WM)))
        probatlas_GM     = np.array(sitk.GetArrayFromImage(sitk.ReadImage(GM)))
        bin_mask         = np.array(sitk.GetArrayFromImage(sitk.ReadImage(mask)))
        
        # Derive Intensity Probability Volume via Tissue Models
        intprob_CSF, intprob_WM, intprob_GM = volumeIntenProb(vol=(mask.replace('/testing-mask','/testing-images')).replace('_1C',''),mask=mask)
        
        # Combined Class Probabilities
        prob_CSF = np.expand_dims((probatlas_CSF*intprob_CSF),axis=3)
        prob_WM  = np.expand_dims((probatlas_WM*intprob_WM),axis=3)
        prob_GM  = np.expand_dims((probatlas_GM*intprob_GM),axis=3)
        
        # Assign Label Matching Max Class Probability
        max_index        = np.argmax(np.concatenate((prob_CSF, prob_WM, prob_GM),axis=3),axis=3)+1
        
        # Maskout External Label Predictions
        predicted_mask   = np.multiply(max_index,bin_mask).astype(np.uint8)
        
        if (export=='save'):
            output_prediction        = sitk.GetImageFromArray(predicted_mask)
            output_prediction.CopyInformation(sitk.ReadImage((mask.replace('/testing-mask','/testing-images')).replace('_1C','')))
            writer           = sitk.ImageFileWriter()
            output_dir       = (mask.replace('../data/testing-set/testing-mask','../results/testing_results/predictions/')).replace('_1C.nii.gz','.nii')
            writer.SetFileName(output_dir)
            writer.Execute(output_prediction)
        elif (export=='return'):
            return predicted_mask


def computeTissueModels(export):
    train_indexes = ['000', '001', '002', '006', '007', '008', '009', '010', 
                     '011', '012', '013', '014', '015', '017', '036']
    i = 0
    for index in train_indexes:  
        CSF, WM, GM = nonzeroBrainFeatures("../data/training-set/training-labels/1"+index+"_3C.nii.gz","../data/training-set/training-images/1"+index+".nii.gz")
        if (i==0):
            CSF_TM = CSF
            WM_TM  = WM
            GM_TM  = GM
        else:    
            CSF_TM = np.concatenate((CSF_TM, CSF),axis=0)
            WM_TM  = np.concatenate((WM_TM,  WM), axis=0)
            GM_TM  = np.concatenate((GM_TM,  GM), axis=0)
        i += 1
 
    if (export=='save'):
        np.save('../results/atlas/tissueModel_CSF.npy', CSF_TM)
        np.save('../results/atlas/tissueModel_WM.npy',  WM_TM)
        np.save('../results/atlas/tissueModel_GM.npy',  GM_TM)
    elif (export=='return'):    
        return CSF_TM,WM_TM,GM_TM


def nonzeroBrainFeatures(import_label,import_volume): 
    # Decomposing Volumes by Individual Masks
    maskedCSF, maskedWM, maskedGM =  maskBrain(import_label,import_volume)

    # Extracting Feature Vectors
    feature_vector_CSF            = maskedCSF.reshape(-1,1)
    feature_vector_WM             = maskedWM.reshape(-1,1)
    feature_vector_GM             = maskedGM.reshape(-1,1)
    feature_vector_CSF__indices   = [i for i,j in enumerate(feature_vector_CSF) if j.any()]
    feature_vector_CSF__nonzero   = feature_vector_CSF[feature_vector_CSF__indices]
    feature_vector_WM__indices    = [i for i,j in enumerate(feature_vector_WM) if j.any()]
    feature_vector_WM__nonzero    = feature_vector_WM[feature_vector_WM__indices]
    feature_vector_GM__indices    = [i for i,j in enumerate(feature_vector_GM) if j.any()]
    feature_vector_GM__nonzero    = feature_vector_GM[feature_vector_GM__indices]
    
    return feature_vector_CSF__nonzero,feature_vector_WM__nonzero,feature_vector_GM__nonzero



def maskBrain(import_label,import_volume): 
    # Load Volume/Mask
    label                 = np.array(sitk.GetArrayFromImage(sitk.ReadImage(import_label)))
    volume                = np.array(sitk.GetArrayFromImage(sitk.ReadImage(import_volume, sitk.sitkFloat32)))

    # Mask Volumes
    maskCSF               = label.copy()
    maskCSF[maskCSF!=1]   = 0
    maskedCSF             = np.multiply(volume,maskCSF.astype(bool))
    maskWM                = label.copy()
    maskWM[maskWM!=2]     = 0
    maskedWM              = np.multiply(volume,maskWM.astype(bool))
    maskGM                = label.copy()
    maskGM[maskGM!=3]     = 0
    maskedGM              = np.multiply(volume,maskGM.astype(bool))  
    return maskedCSF, maskedWM, maskedGM



def computeAtlasProb(export):
    train_indexes = ['000', '001', '002', '006', '007', '008', '009', '010', 
                 '011', '012', '013', '014', '015', '017', '036']
    i = 0

    for index in train_indexes:
        if (i==0):  # Reference Image
            label         = np.array(sitk.GetArrayFromImage(sitk.ReadImage("../data/training-set/training-labels/1"+index+"_3C.nii.gz")))
            CSF_labels    = np.zeros((label.shape[0],label.shape[1],label.shape[2],len(train_indexes)))
            WM_labels     = np.zeros((label.shape[0],label.shape[1],label.shape[2],len(train_indexes)))
            GM_labels     = np.zeros((label.shape[0],label.shape[1],label.shape[2],len(train_indexes)))           
        else:      # Registered Training Images
            label       = np.array(sitk.GetArrayFromImage(sitk.ReadImage("../results/training_results/transformed_labels/"+index+"/result.mhd")))

        CSF         = label.copy()
        CSF[CSF!=1] = 0
        WM          = label.copy()
        WM[WM!=2]   = 0
        GM          = label.copy()
        GM[GM!=3]   = 0
            
        CSF_labels[:,:,:,i] = CSF
        WM_labels[:,:,:,i]  = WM
        GM_labels[:,:,:,i]  = GM
        i += 1

    # Compute Probability Atlases for Each Label 
    prob_atlas_CSF = np.average(CSF_labels,axis=3)
    prob_atlas_WM  = np.average(WM_labels,axis=3)
    prob_atlas_GM  = np.average(GM_labels,axis=3)

    if (export=='save'):
        output_prob_atlas_CSF        = sitk.GetImageFromArray(prob_atlas_CSF/1)
        output_prob_atlas_WM         = sitk.GetImageFromArray(prob_atlas_WM/2)
        output_prob_atlas_GM         = sitk.GetImageFromArray(prob_atlas_GM/3)
        
        output_prob_atlas_CSF.CopyInformation(sitk.ReadImage("../data/training-set/training-labels/1000_3C.nii.gz"))
        output_prob_atlas_WM.CopyInformation(sitk.ReadImage("../data/training-set/training-labels/1000_3C.nii.gz"))
        output_prob_atlas_GM.CopyInformation(sitk.ReadImage("../data/training-set/training-labels/1000_3C.nii.gz"))
        
        writer           = sitk.ImageFileWriter()
        writer.SetFileName('../results/atlas/prob_atlas_CSF.nii')
        writer.Execute(output_prob_atlas_CSF)
        writer.SetFileName('../results/atlas/prob_atlas_WM.nii')
        writer.Execute(output_prob_atlas_WM)
        writer.SetFileName('../results/atlas/prob_atlas_GM.nii')
        writer.Execute(output_prob_atlas_GM)
        
    elif (export=='return'):
        return prob_atlas_CSF/1, prob_atlas_WM/2, prob_atlas_GM/3




















