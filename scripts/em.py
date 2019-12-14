### LIBRARIES
import numpy as np
import SimpleITK as sitk
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal
from em_aux import dice_metric, restructure_KMeans
from prime_aux import labelPropg, volumeIntenProb


def segmentEM(volume_dir,labels_dir,mask_dir,init_mode,mode,export,atlas=None):
    # Load Volumes
    volumeITK                = sitk.ReadImage(volume_dir, sitk.sitkFloat32)
    volume                   = np.array(sitk.GetArrayFromImage(volumeITK))
    labels                   = np.array(sitk.GetArrayFromImage(sitk.ReadImage(labels_dir)))
    mask                     = np.array(sitk.GetArrayFromImage(sitk.ReadImage(mask_dir)))
      
    # Masking Volume and Feature Vector
    masked_volume            = np.multiply(volume,mask.astype(bool))
    feature_vector           = masked_volume.reshape(-1,1)
    feature_vector_indices   = [i for i,j in enumerate(feature_vector) if j.any()]
    feature_vector_nonzero   = feature_vector[feature_vector_indices]
    
    
    if (init_mode=='kmeans'):
        ### K-MEANS CLUSTERING
        kmeans                     = KMeans(n_clusters=3, random_state=0).fit(feature_vector_nonzero)
        KMpredict, KMcentroids     = restructure_KMeans(kmeans,feature_vector_nonzero)
        predicted_mask_KM          = np.zeros((masked_volume.reshape(-1,1)).shape)
        predicted_mask_KM[feature_vector_indices] = KMpredict.reshape(-1,1)
        predicted_mask_KM          = np.reshape(predicted_mask_KM,masked_volume.shape)
        
        ### EXPECTATION AND MAXIMIZATION INITIALIZATION
        KM_CSF       = feature_vector_nonzero[KMpredict==1]
        KM_WM        = feature_vector_nonzero[KMpredict==2]
        KM_GM        = feature_vector_nonzero[KMpredict==3]
        
     
    elif (init_mode=='atlas'):
        ### LABEL PROPAGATION VIA PROBABILITY ATLASES + TISSUE MODELS
        index        = (volume_dir.replace("../data/testing-set/testing-images/1","")).replace(".nii.gz","")
        
        
        if (atlas=='training'):
            propg_labels = labelPropg(CSF='../results/testing_results/transformed_labels/CSF/'+index+'/result.mhd', WM='../results/testing_results/transformed_labels/WM/'+index+'/result.mhd', 
                                      GM='../results/testing_results/transformed_labels/GM/'+index+'/result.mhd',   mask='../data/testing-set/testing-mask/1'+index+'_1C.nii.gz',
                                      mode='prob_inten_atlas', export='return')
        elif (atlas=='MNI'):
            propg_labels = labelPropg(CSF='../results/testing_results/transformed_labels_MNI/CSF/'+index+'/result.mhd', WM='../results/testing_results/transformed_labels_MNI/WM/'+index+'/result.mhd', 
                                      GM='../results/testing_results/transformed_labels_MNI/GM/'+index+'/result.mhd',   mask='../data/testing-set/testing-mask/1'+index+'_1C.nii.gz',
                                      mode='prob_inten_atlas', export='return')    
        
        propg_labels               = propg_labels.reshape(-1,1)
        propg_labels_indices       = [i for i,j in enumerate(propg_labels) if j.any()]
        ATpredict                  = propg_labels[propg_labels_indices]
        
        ### EXPECTATION AND MAXIMIZATION INITIALIZATION
        KM_CSF       = feature_vector_nonzero[ATpredict==1]
        KM_WM        = feature_vector_nonzero[ATpredict==2]
        KM_GM        = feature_vector_nonzero[ATpredict==3]
                
        
    mean_CSF     = np.mean(KM_CSF, axis = 0)
    mean_WM      = np.mean(KM_WM,  axis = 0)
    mean_GM      = np.mean(KM_GM,  axis = 0)
    covar_CSF    = np.cov(KM_CSF,  rowvar = False)
    covar_WM     = np.cov(KM_WM,   rowvar = False)
    covar_GM     = np.cov(KM_GM,   rowvar = False)
    
    prob_CSF     = KM_CSF.shape[0] / feature_vector_nonzero.shape[0]
    prob_WM      = KM_WM.shape[0]  / feature_vector_nonzero.shape[0]
    prob_GM      = KM_GM.shape[0]  / feature_vector_nonzero.shape[0]

    prob_distr_0 = np.array((prob_CSF, prob_WM , prob_GM))
    
    # Iterative Update
    max_change   = 0.01
    max_steps    = 10
    
    for i in range(max_steps):
        # Expectation Step: Probability Density Function
        PDF_CSF      = multivariate_normal.pdf(feature_vector_nonzero, mean=mean_CSF, cov=covar_CSF)
        PDF_WM       = multivariate_normal.pdf(feature_vector_nonzero, mean=mean_WM,  cov=covar_WM)
        PDF_GM       = multivariate_normal.pdf(feature_vector_nonzero, mean=mean_GM,  cov=covar_GM)
        
        weights_CSF  = (prob_CSF * PDF_CSF)/((prob_CSF * PDF_CSF) + (prob_WM * PDF_WM) + (prob_GM * PDF_GM))
        weights_WM   = (prob_WM  * PDF_WM) /((prob_CSF * PDF_CSF) + (prob_WM * PDF_WM) + (prob_GM * PDF_GM))
        weights_GM   = (prob_GM  * PDF_GM) /((prob_CSF * PDF_CSF) + (prob_WM * PDF_WM) + (prob_GM * PDF_GM))
        weights      = np.concatenate((weights_CSF.reshape(-1,1),weights_WM.reshape(-1,1),weights_GM.reshape(-1,1)),axis=1)
        log_B        = sum((np.log(sum(weights))))
        
        # Maximization Step: New Probabilities
        _,counts     = np.unique(np.argmax(weights,axis=1)+1, return_counts=True)
        
        if (len(counts)==2):
            counts = [1e-7, counts[0], counts[1]]
            
        prob_CSF     = counts[0] / feature_vector_nonzero.shape[0]
        prob_WM      = counts[1] / feature_vector_nonzero.shape[0]
        prob_GM      = counts[2] / feature_vector_nonzero.shape[0]
        prob_distr_1 = np.array((prob_CSF, prob_WM , prob_GM))
        
        # Maximization Step: New Mean and Covariance
        mean_CSF     = (1/counts[0]) * (weights[:,0] @ feature_vector_nonzero)
        mean_WM      = (1/counts[1]) * (weights[:,1] @ feature_vector_nonzero)
        mean_GM      = (1/counts[2]) * (weights[:,2] @ feature_vector_nonzero)
        covar_CSF    = (1/counts[0]) * (weights[:,0] * np.transpose(feature_vector_nonzero - mean_CSF)) @ (feature_vector_nonzero - mean_CSF)
        covar_WM     = (1/counts[1]) * (weights[:,1] * np.transpose(feature_vector_nonzero - mean_WM))  @ (feature_vector_nonzero - mean_WM)
        covar_GM     = (1/counts[2]) * (weights[:,2] * np.transpose(feature_vector_nonzero - mean_GM))  @ (feature_vector_nonzero - mean_GM)
        
        # Expectation Step: Probability Density Function
        PDF_CSF      = multivariate_normal.pdf(feature_vector_nonzero, mean=mean_CSF, cov=covar_CSF)
        PDF_WM       = multivariate_normal.pdf(feature_vector_nonzero, mean=mean_WM,  cov=covar_WM)
        PDF_GM       = multivariate_normal.pdf(feature_vector_nonzero, mean=mean_GM,  cov=covar_GM)
        
        weights_CSF  = (prob_CSF * PDF_CSF)/((prob_CSF * PDF_CSF) + (prob_WM * PDF_WM) + (prob_GM * PDF_GM))
        weights_WM   = (prob_WM  * PDF_WM) /((prob_CSF * PDF_CSF) + (prob_WM * PDF_WM) + (prob_GM * PDF_GM))
        weights_GM   = (prob_GM  * PDF_GM) /((prob_CSF * PDF_CSF) + (prob_WM * PDF_WM) + (prob_GM * PDF_GM))
        weights      = np.concatenate((weights_CSF.reshape(-1,1),weights_WM.reshape(-1,1),weights_GM.reshape(-1,1)),axis=1)
        log_N        = sum((np.log(sum(weights)))) 
        
        # Update Trackers, Verify Conditions
        change_distr = np.linalg.norm(prob_distr_1 - prob_distr_0)
        change_log   = np.linalg.norm(log_N-log_B)
        
        print("Step: {counter}; Distribution Change: {change}".format(counter=i+1,change=change_distr))
        if (change_log <= max_change):
            prob_distr_0 = prob_distr_1
        else:
            break
    
    # Output Image Reconstruction    
    predicted_mask_EM                          = np.zeros((masked_volume.reshape(-1,1)).shape)
    
    if (mode=='atlas'):
        
        # Load Probability Atlases
        if (atlas=='training'):
            probatlas_CSF    = np.array(sitk.GetArrayFromImage(sitk.ReadImage('../results/testing_results/transformed_labels/CSF/'+index+'/result.mhd')))
            probatlas_WM     = np.array(sitk.GetArrayFromImage(sitk.ReadImage('../results/testing_results/transformed_labels/WM/'+index+'/result.mhd')))
            probatlas_GM     = np.array(sitk.GetArrayFromImage(sitk.ReadImage('../results/testing_results/transformed_labels/GM/'+index+'/result.mhd')))
            bin_mask         = np.array(sitk.GetArrayFromImage(sitk.ReadImage(mask_dir)))
        elif (atlas=='MNI'):
            probatlas_CSF    = np.array(sitk.GetArrayFromImage(sitk.ReadImage('../results/testing_results/transformed_labels_MNI/CSF/'+index+'/result.mhd')))
            probatlas_WM     = np.array(sitk.GetArrayFromImage(sitk.ReadImage('../results/testing_results/transformed_labels_MNI/WM/'+index+'/result.mhd')))
            probatlas_GM     = np.array(sitk.GetArrayFromImage(sitk.ReadImage('../results/testing_results/transformed_labels_MNI/GM/'+index+'/result.mhd')))
            bin_mask         = np.array(sitk.GetArrayFromImage(sitk.ReadImage(mask_dir)))
            
        # Derive Intensity Probability Volume via Tissue Models
        intprob_CSF, intprob_WM, intprob_GM = volumeIntenProb(vol=(mask_dir.replace('/testing-mask','/testing-images')).replace('_1C',''),mask=mask_dir)
         
        # Reconstruct EM Output Volumes
        predicted_mask_EM_CSF                         = np.zeros((masked_volume.reshape(-1,1)).shape)
        predicted_mask_EM_CSF[feature_vector_indices] = weights[:,0].reshape(-1,1)
        predicted_mask_EM_CSF                         = np.reshape(predicted_mask_EM_CSF, masked_volume.shape)
        predicted_mask_EM_WM                          = np.zeros((masked_volume.reshape(-1,1)).shape)
        predicted_mask_EM_WM[feature_vector_indices]  = weights[:,1].reshape(-1,1)
        predicted_mask_EM_WM                          = np.reshape(predicted_mask_EM_WM, masked_volume.shape)                
        predicted_mask_EM_GM                          = np.zeros((masked_volume.reshape(-1,1)).shape)
        predicted_mask_EM_GM[feature_vector_indices]  = weights[:,2].reshape(-1,1)
        predicted_mask_EM_GM                          = np.reshape(predicted_mask_EM_GM, masked_volume.shape)
        
        # Combined Class Probabilities
        prob_CSF = np.expand_dims((probatlas_CSF*intprob_CSF*predicted_mask_EM_CSF),axis=3)
        prob_WM  = np.expand_dims((probatlas_WM*intprob_WM*predicted_mask_EM_WM),axis=3)
        prob_GM  = np.expand_dims((probatlas_GM*intprob_GM*predicted_mask_EM_GM),axis=3)
        
        # Assign Label Matching Max Class Probability
        max_index          = np.argmax(np.concatenate((prob_CSF, prob_WM, prob_GM),axis=3),axis=3)+1
        
        # Maskout External Label Predictions
        predicted_mask_EM  = np.multiply(max_index,bin_mask).astype(np.uint8)
        
    elif (mode=='base'):
        predicted_mask_EM                          = np.zeros((masked_volume.reshape(-1,1)).shape)
        predicted_mask_EM[feature_vector_indices]  = (np.argmax(weights,axis=1)+1).reshape(-1,1)
        predicted_mask_EM                          = np.reshape(predicted_mask_EM,masked_volume.shape)


    if (export=='return'):
        return predicted_mask_EM, dice_metric(predicted_mask_EM,labels,3)
   

    elif (export=='save'):
        output_prediction        = sitk.GetImageFromArray(predicted_mask_EM)
        output_prediction.CopyInformation(sitk.ReadImage((mask_dir.replace('/testing-mask','/testing-images')).replace('_1C','')))
        writer                   = sitk.ImageFileWriter()
        
        if (init_mode=='kmeans'):
            output_dir       = (mask_dir.replace('../data/testing-set/testing-mask','../results/testing_results/predictions/predictions02_em-kmeans')).replace('_1C.nii.gz','.nii')
        elif (init_mode=='atlas'):
            if (mode=='base'):
                if (atlas=='training'):
                    output_dir       = (mask_dir.replace('../data/testing-set/testing-mask','../results/testing_results/predictions/predictions03_em-atlas-init')).replace('_1C.nii.gz','.nii')
                elif (atlas=='MNI'):
                    output_dir       = (mask_dir.replace('../data/testing-set/testing-mask','../results/testing_results/predictions/predictions04_em-atlasMNI-init')).replace('_1C.nii.gz','.nii')
            elif (mode=='atlas'):
                if (atlas=='training'):
                    output_dir       = (mask_dir.replace('../data/testing-set/testing-mask','../results/testing_results/predictions/predictions05_em-atlas-full')).replace('_1C.nii.gz','.nii')
                elif (atlas=='MNI'):
                    output_dir       = (mask_dir.replace('../data/testing-set/testing-mask','../results/testing_results/predictions/predictions06_em-atlasMNI-full')).replace('_1C.nii.gz','.nii')
               
        writer.SetFileName(output_dir)
        writer.Execute(output_prediction)













