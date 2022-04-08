# semantic segmentation



## File discrptions:
dataloaders: scripts for preprocessing input data, data augmentation.  
experiments: scripts for evaluating semantic results: cell/nucleus/mitochondria.  
figure: the workflow figure for this paper.  
networks: framework structure of unet.  
script: commands to run model on test data.  
utils: function for calculating loss, ramps.  
training model: Unet model frameworks. From [UPSnet](https://github.com/charlotte12l/UPSNet)




# Workflow
## Data Preparation for Soft X-ray tomograms (SXT) Dataset
**1. Download SXT dataset from PBC**

Please go to [PBC Dataset](https://pbcconsortium.isrd.isi.edu/) and download [SXT images](https://drive.google.com/drive/folders/1ReftAgqdNtjqm5ZKe2rkoTYWYfz-s5vb?usp=sharing) and [manual labels](https://drive.google.com/drive/folders/1DCPRYQXFNRHMd5FFDtTr81hclttLgfC_?usp=sharing), and put them under data folder.
```
# symlink the pascal dataset
cd data
ln -s /path_to_sxt_images/ ./image_xyz
ln -s /path_to_manual_labels/ ./mask_xyz
```

**2. Download pretrained model**
Download the [FS_mito](https://drive.google.com/file/d/1RCwETRuUYfAURD8XXJ1RQVGOTIQavQuE/view?usp=sharing) 
and [FS_mem_nu](https://drive.google.com/file/d/1TRk_AWms32EFG-sqhfd4q4Td3ptmlxEp/view?usp=sharing)
weights and put them under ./results folder.
```
# download the pretrained model
cd results
mkdir FS_mito
mkdir FS_mem_nu
```


## Inference in Command Line for test data
```
# Test mitochondria model 
sh script/test_mito.sh

# Test membrane and nuclear model
sh script/test_mem_nu.sh
```
There are some hyperparameters that you can adjust in the script.

```
python test/eval_mito.py --gpu 0 --exp FS_mito --num-workers 8 --batch-size 1 --num-classes 4 --test_idx 'iso'
```
PS: You can change the ```EXP``` in script to your pretrained model name. The above is just an example.

semantic segmentation frameworks are partially based on [PANet](https://github.com/kaixin96/PANet)

## Semantic Results
We list the performance w/o 3D fusion.
|   Before 3D fusion | Membrane |  Nucleus  | Mito |    
|:-----------------------:|:--------------------------------------------------:|:--------------------------------------------------:|:--------------------------------------------------:|
| 766_8 | 90.74| 93.21| 68.58| 
| 784_5 | 87.43| 89.95| 63.17|
| 842_17 | 85.34| 83.50| 65.03||
| mean | **87.84** | **88.89**| **65.59** |

|    After 3D fusion | Membrane |  Nucleus  | Mito |
|:-----------------:|:-----------------:|:-----------------:|:-----------------:|
| 766_8 |93.54           |  93.92  |70.34 | 
| 784_5 | 89.41            |  91.82  | 67.29|
| 842_17 | 91.85      |  89.49  |67.40|
| mean |**91.60** |  **91.74**  |**68.34**|


## Reference: 
semantic segmentation frameworks are partially based on [PANet](https://github.com/kaixin96/PANet)
