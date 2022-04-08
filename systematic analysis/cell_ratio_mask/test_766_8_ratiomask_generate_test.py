
import numpy as np
import tifffile
import time, os, math, json, copy
from radialDistribution_function_main_v2 import * 


# datasetnum = '766_2'
datasetnum = '766_8'
# mainpath = f'F:/modelling/Cluster-bkp/merged_prediction_raw_masks'
mainpath = f'/public/home/liad/raw_data_predict_all_mask/merged_mask'  ## for manual masks
mrcpath = f'/public/home/liad/mrc_rawdata'
outputpath = f'/public/home/liad/ratio_rdf_mask_new/output'
times_ = 11 ## shrunken times 
times_2 = 3 ## ratio mask devided on which scale, 1 represent on origonal size  
nn = 0.5  ## to classified voxel in parts 

datasetinfo_ = dict()
datasetinfo_['datasetnum'] = datasetnum
datasetinfo_['mainpath'] = mainpath
datasetinfo_['mrcpath'] = mrcpath
datasetinfo_['outputpath'] = outputpath
datasetinfo_['times_'] = times_
datasetinfo_['times_2'] = times_2
datasetinfo_['nn'] = nn 

ratio_mask_main(datasetinfo_)
