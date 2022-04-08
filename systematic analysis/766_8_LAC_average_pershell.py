
import numpy as np
import tifffile
import time, os, math, json, copy
from isg_LAC_distribution_per_shell_v1 import * 


# datasetnum = '766_2'
datasetnum = '766_8'

mainpath = f'/public/home/liad/raw_data_predict_all_mask/merged_mask'
isgpath = f'/public/home/liad/raw_data_predict_all_mask/matched_mito_ins_mask'  ## isg file are in mito files
mitopath = f'/public/home/liad/raw_data_predict_all_mask/matched_mito_ins_mask'
mrcpath = f'/public/home/liad/mrc_rawdata'
ratiomaskpath = f'/public/home/liad/ratio_rdf_mask_new_final/output'
outputpath = f'/public/home/liad/ratio_rdf_mask_new_final/output_LAC_info_shell'


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


mask_info_main(datasetinfo_)
