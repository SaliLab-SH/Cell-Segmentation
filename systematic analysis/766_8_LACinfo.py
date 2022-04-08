
import numpy as np
import tifffile
import time, os, math, json, copy
from mask_volume_LAC_info_v1 import * 


# datasetnum = '766_2'
datasetnum = '766_8'
# mainpath = f'F:/modelling/Cluster-bkp/merged_prediction_raw_masks'
# mainpath = f'F:/PBC_data/datasets/organelle_mask_5min-new/merged_manual_masks'  ## for manual masks
mainpath = f'/public/home/liad/raw_data_predict_all_mask/merged_mask'
isgpath = f'/public/home/liad/raw_data_predict_all_mask/matched_mito_ins_mask'  ## isg file are in mito files
mitopath = f'/public/home/liad/raw_data_predict_all_mask/matched_mito_ins_mask'
mrcpath = f'/public/home/liad/mrc_rawdata'
ratiomaskpath = f'/public/home/liad/ratio_rdf_mask_new_final/output'
outputpath = f'/public/home/liad/ratio_rdf_mask_new_final/output_LAC_info'

# read 
# with open('F:/PBC_data/LAC/LAC_value.json','r') as mf:
with open(f'LAC_value.json','r') as mf:
    LAC_value = json.load(mf)


if not os.path.exists(outputpath):  
    os.mkdir(outputpath)


datasetinfo_ = dict()
datasetinfo_['datasetnum'] = datasetnum
datasetinfo_['mainpath'] = mainpath
datasetinfo_['mrcpath'] = mrcpath
datasetinfo_['isgpath'] = isgpath
datasetinfo_['mitopath'] = mitopath
datasetinfo_['ratiomaskpath'] = ratiomaskpath
datasetinfo_['outputpath'] = outputpath
datasetinfo_['LAC'] = LAC_value



print('now on 766_8 lac info')
mask_info_main(datasetinfo_)



