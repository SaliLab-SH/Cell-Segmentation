'''
1. generate index masks
2. find_pairs 
3. cgal to separate shells
4. generate rdfs
%. shrinkun final ratio mask for quick test


'''

import numpy as np
import tifffile
import time, os, math, json, copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile
import mrcfile
import scipy.ndimage as ndimage
from radialDistribution_functions import * 



def test_decorator(f):
    def inner(*args, **kwargs):
        tiff_nucleus, tiff_wholecell = f(*args )
        shape_ = tiff_nucleus.shape
        roi = [slice(int(shape_[0]/4),int(shape_[0]/4*3)), slice(int(shape_[1]/4),int(shape_[1]/4*3)), slice(int(shape_[2]/4),int(shape_[2]/4*3))]
        tiff_nucleus = tiff_nucleus[roi]
        tiff_wholecell = tiff_wholecell[roi]
        # shape2 = tiff_nucleus.shape
        # tiff_nucleus_new = np.zeros([shape2[0] + 2, shape2[1] + 2,shape2[2] + 2])
        # tiff_nucleus_new[1:-1,1:-1,1:-1] = tiff_nucleus
        # tiff_wholecell_new = np.zeros_like(tiff_nucleus_new)
        # tiff_wholecell_new[1:-1,1:-1,1:-1] = tiff_wholecell

        return tiff_nucleus, tiff_wholecell
        # return tiff_nucleus_new, tiff_wholecell_new
    return inner



# @test_decorator
def read_image(datasetnum, mainpath, check_ = True):

    for maindir, subdir, file_name_list in os.walk(mainpath, topdown=False):
        filelist = np.array(file_name_list)

    for name in filelist:
        if f'{datasetnum}_' in name :
            tif_name = f'{mainpath}/{name}'
            print('current tiff ',tif_name)


    temp_tiff = tifffile.imread(tif_name)
    # mrc = mrcfile.open(mrc_name, permissive=True).data


    tiff_wholecell = np.zeros_like(temp_tiff)
    tiff_nucleus = np.zeros_like(temp_tiff)
    tiff_isg = np.zeros_like(temp_tiff)
    tiff_mito = np.zeros_like(temp_tiff)
    
    tiff_wholecell[np.where(temp_tiff != 0)] = 1
    tiff_nucleus[np.where(temp_tiff == 2)] = 1
    tiff_isg[np.where(temp_tiff ==4)] = 1
    tiff_mito[np.where(temp_tiff == 3)] = 1
    
    # pm_nc_tiff[70:380,:,70:380] = temp_tiff
    # wholecell_tiff = np.zeros_like(mrc)
    # nc_tiff = np.zeros_like(mrc)
    # wholecell_tiff[np.where(pm_nc_tiff != 0)] = 1
    # nc_tiff[np.where(pm_nc_tiff ==2 )] = 1


    if check_ :

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4, figsize=(20,5))
        # fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(20,5))       
        # ax1.imshow(mrc[232, :, :] )
        ax1.imshow(tiff_isg[232, :, :])
        ax2.imshow(tiff_mito[232, :, :])
        ax3.imshow(tiff_nucleus[232, :, :])
        ax4.imshow(tiff_wholecell[232, :, :])

        plt.title(f'{datasetnum}_all_organelles_checkplot')
        # plt.savefig(f'{datasetnum}_all_organelles_checkplot.png')
        plt.show()
        plt.close()
    
    # return datasetnum
    return  tiff_nucleus, tiff_wholecell




def read_mrcimage(datasetnum, mainpath, check_ = False):
    # if datasetnum == '783_11': datasetnum = '783_6'
    # elif datasetnum == '931_9': datasetnum = '931_10'
    mrcpath = mainpath
    for maindir, subdir, file_name_list in os.walk(mrcpath, topdown=False):
        filelist = np.array(file_name_list)
        
    # if datasetnum == '1068_17':
    #     for name in filelist:
    #         if datasetnum in name:
    #             mrc_name = f'{mrcpath}/{name}'
    #             filename = mrc_name
    #             print(mrc_name)
    #     mrc = tifffile.imread(mrc_name) 
    
    # else:
    for name in filelist:
        if f'{datasetnum}_' in name and  '.mrc' in name:
            mrc_name = f'{mrcpath}/{name}'
            filename = mrc_name
            print(mrc_name)
        else:
            pass
        

    mrc = mrcfile.open(mrc_name, permissive=True).data
    
    

    if check_ :
        plt.imshow(mrc[232])
        plt.show()
        plt.close()
    
    return mrc





def ratio_mask_main(datasetinfo_):
    datasetnum = datasetinfo_['datasetnum']
    mainpath = datasetinfo_['mainpath']
    mrcpath = datasetinfo_['mrcpath']
    outputpath = datasetinfo_['outputpath']
    times_ = datasetinfo_['times_']
    times_2 = datasetinfo_['times_2']
    nn = datasetinfo_['nn']



    # # datasetnum = '766_2'
    # datasetnum = '1082_22'
    # # mainpath = f'F:/modelling/Cluster-bkp/merged_prediction_raw_masks'
    # mainpath = f'F:/PBC_data/datasets/organelle_mask_5min-new/merged_manual_masks'  ## for manual masks
    # mrcpath = f'F:/PBC_data/datasets/raw_image'
    # outputpath = f'F:/PBC_data/RadialDistributionFunction_X-ray_tomograms/output'

    print(datasetnum)

    ## peremeters 
    # times_ = 11 ## shrunken times 
    # times_2 = 1 ## ratio mask devided on which scale, 1 represent on origonal size  
    # nn = 0.5  ## to classified voxel

    ## get img
    # add decorator now 
    temp_nc_tiff, temp_pm_tiff = read_image(datasetnum, mainpath, check_ = True)   
    mrcfile = read_mrcimage(datasetnum, mrcpath, check_ = True)
    normed_mrc = mrcfile * 27.161   #LAC_factor[f'{datasetnum}']


    # for i in range((temp_nc_tiff.shape[0]//7)):

    #     plt.imshow(temp_nc_tiff[int(temp_nc_tiff.shape[0]/2)])
    #     plt.imshow(temp_nc_tiff[0 + 7 * i ])
    #     plt.show()
    #     plt.close()
    # print(np.sum(temp_nc_tiff))

    


    ## shrunken masks
    # nc_tiff_new, pm_tiff_new, = shrink_mask_with_extend(temp_nc_tiff, times_), shrink_mask_with_extend(temp_pm_tiff, times_)
    ne_shrunken = shrink_mask_with_include(temp_nc_tiff, times_)
    pm_shrunken = shrink_mask_with_include(temp_pm_tiff, times_)

    shape2 = ne_shrunken.shape
    ne_shrunken_new = np.zeros([shape2[0] + 2, shape2[1] + 2,shape2[2] + 2])
    ne_shrunken_new[1:-1,1:-1,1:-1] = ne_shrunken
    pm_shrunken_new = np.zeros([shape2[0] + 2, shape2[1] + 2,shape2[2] + 2])
    pm_shrunken_new[1:-1,1:-1,1:-1] = pm_shrunken
    ne_shrunken = ne_shrunken_new
    pm_shrunken = pm_shrunken_new

    plt.imshow(ne_shrunken[int(ne_shrunken.shape[0]/2)])
    plt.show()
    plt.close()
    plt.imshow(pm_shrunken[int(pm_shrunken.shape[0]/2)])
    plt.show()
    plt.close()





    ne_edt = ndimage.distance_transform_edt(ne_shrunken)
    ne_edge = np.zeros_like(ne_shrunken)
    ne_edge[np.where(ne_edt == 1)] = 1
    # del ne_edt

    pm_edt = ndimage.distance_transform_edt(pm_shrunken)
    pm_edge = np.zeros_like(pm_shrunken)
    pm_edge[np.where(pm_edt == 1)] = 1
    # del pm_edt

    plt.imshow(ne_edt[int(ne_edt.shape[0]/2)])
    plt.show()
    plt.close()
    plt.imshow(pm_edt[int(pm_edt.shape[0]/2)])
    plt.show()
    plt.close()


    # ne_edge, pm_edge, = shrink_mask_with_include(ne_edge, times_), shrink_mask_with_include(pm_edge, times_)




    plt.imshow(ne_edge[int(ne_edge.shape[0]/2)])
    plt.show()
    plt.close()
    plt.imshow(pm_edge[int(pm_edge.shape[0]/2)])
    plt.show()
    plt.close()


    ## find pairs
    ne_edge_coord = np.array(np.where(ne_edge == 1)).T
    pm_edge_coord = np.array(np.where(pm_edge == 1)).T
    ne_coords = np.where(ne_shrunken == 1)
    center = [np.mean(ne_coords[0]), np.mean(ne_coords[1]), np.mean(ne_coords[2])]

    ###  classified region, to reduce calculations

    classified_dict = classified_region(pm_edge_coord, center, nn_ = nn)
    # print(classified_dict)

    # pair_out=open(f'{outputpath}/{datasetnum}_pair_ne-pm_1.xvg', 'w') # output
    # pairs = []
    print(len(pm_edge_coord))
    print(int(len(pm_edge_coord) / 8.0 ))
    ne_vect_dict = {}
    for i in range(0, int(len(ne_edge_coord))): ## need check
        # temp =  [ min_angle(ne, pm[i], center), pm[i] ]
        # pairs.append(temp)
        if i % 200 == 0 : print(i)
        # print(min_angle(ne_edge_coord, pm_edge_coord[i], center))
        # print('pm_edge_coord',pm_edge_coord[i])

        direction_ = classified_point(ne_edge_coord[i], center, nn_ = nn)
        # print('direction_',direction_)
        pm_edge_coord_partial = classified_dict[f'{direction_}']
        # print(len(pm_edge_coord_partial))
        # print(pm_edge_coord_partial)

        pmpoint = min_angle(pm_edge_coord_partial, pm_edge_coord, ne_edge_coord[i], center) 
        vector_ = ne_edge_coord[i] - pmpoint
        ne_vect_dict[f'{ne_edge_coord[i]}'] = vector_.tolist()

        # print(str(' '.join(map(str, min_angle(ne_edge_coord, pm_edge_coord[i], center) + pm_edge_coord[i]))))
        # pair_out.write(' '.join(map(str, min_angle(ne_edge_coord, pm_edge_coord[i], center) + pm_edge_coord[i])))
        # pair_out.write("\n")  

    with open(f'{outputpath}/{datasetnum}_needge_vectors_{times_}_shrunken.json', 'w') as f:
        json.dump(ne_vect_dict, f)


    ## classified_linears


    ## generate ratiomask
    cytosol_mask_shrunken = pm_shrunken - ne_shrunken + ne_edge
    cytosol_coord = np.where(cytosol_mask_shrunken == 1)
    plt.imshow(cytosol_mask_shrunken[int(cytosol_mask_shrunken.shape[0]/2)])
    plt.show()
    plt.close()
    # print('cyto num',np.sum(cytosol_mask_shrunken



    classified_tiff = np.zeros_like(ne_edge)
    ne_edge_coord2 = np.where(ne_edge == 1)
    print(len(ne_edge_coord2[0]))

    for i in range(len(ne_edge_coord2[0])):
        coord = [ne_edge_coord2[0][i], ne_edge_coord2[1][i], ne_edge_coord2[2][i]]
        # print(coord)
        # print('i', i)

        classified_tiff[coord[0]][coord[1]][coord[2]] = i + 1
        # print(list(set(list(classified_tiff.reshape(1, -1))[0])))

    classified_dict_ne = classified_region(ne_edge_coord, center, nn_ = nn)
    print(len(cytosol_coord[0]))

    for i in range(len(cytosol_coord[0])):
        if i % 2000 == 0: print(i)
        current_coord = [cytosol_coord[0][i], cytosol_coord[1][i], cytosol_coord[2][i]]
        direction_2 = classified_point(current_coord, center, nn_ = nn)

        ne_edge_coord_partial = classified_dict_ne[f'{direction_2}']
        # print('ne_edge_coord_partial', ne_edge_coord_partial)

        nepoint = min_angle(ne_edge_coord_partial, ne_edge_coord , current_coord, center) 
        # print('nepoint', nepoint)
        classified_tiff[current_coord[0]][current_coord[1]][current_coord[2]] = classified_tiff[nepoint[0]][nepoint[1]][nepoint[2]]
    
    classified_tiff = classified_tiff[1:-1, 1:-1, 1:-1]
    # print(classified_tiff.shape)
    tifffile.imsave(f'{outputpath}/{datasetnum}_classified_cytosol_shrunken_mask.tiff', classified_tiff)
    # print('sum', np.sum(classified_tiff))
    # print(list(set(list(classified_tiff.reshape(1, -1))[0])))
    # print(classified_tiff[int(classified_tiff.shape[0]/2)])
    plt.imshow(classified_tiff[int(classified_tiff.shape[0]/2)])
    plt.show()
    plt.close()
    

    ## generate_ratio_masks
    '''
    edt to center, min dist, max dist,every voxel dist
    ratio = (voxel dist - min dist / max dist)
    '''

    ne_edge = np.zeros_like(temp_nc_tiff) 
    ne_edt_1size = ndimage.distance_transform_edt(ne_edge)
    ne_edge[np.where(ne_edt_1size == 1)] = 1
    cytosol_mask = temp_pm_tiff - temp_nc_tiff + ne_edge
    del ne_edt_1size, ne_edge

    plt.imshow(cytosol_mask[int(cytosol_mask.shape[0]/2)])
    plt.show()
    plt.close()

    print('classified_tiff shape', classified_tiff.shape)
    print('check cytosol_mask', list(set(list(cytosol_mask.reshape(1,-1))[0])))

    classified_mask_extendsize = back_originaltiff(classified_tiff, times_ = times_ )
    classified_mask_originalsize = np.zeros_like(cytosol_mask)
    classified_mask_originalsize = classified_mask_extendsize[:cytosol_mask.shape[0], :cytosol_mask.shape[1],:cytosol_mask.shape[2]]
    del classified_mask_extendsize

    plt.imshow(classified_mask_originalsize[int(cytosol_mask.shape[0]/2)])
    plt.show()
    plt.close()

    print('classified_mask_originalsize', classified_mask_originalsize.shape)
    print('cytosol mask', cytosol_mask.shape )
    classified_cytosol_mask = classified_mask_originalsize * cytosol_mask

    plt.imshow(classified_cytosol_mask[int(cytosol_mask.shape[0]/2)])
    plt.show()
    plt.close()
    del classified_mask_originalsize
    
    
    ratio_mask = generate_ratio_mask(temp_nc_tiff, cytosol_mask, classified_cytosol_mask, times_ = times_2)

    plt.imshow(ratio_mask[int(ratio_mask.shape[0]/2)])
    plt.imsave(f'{outputpath}/{datasetnum}_check_output_plot.tiff', ratio_mask[int(ratio_mask.shape[0]/2)])
    plt.show()
    plt.close()

    tifffile.imsave(f'{outputpath}/{datasetnum}_cytosol_ratio_mask.tiff', ratio_mask)
    del ratio_mask






    print(f'done with {datasetnum}')


def multi_main_test():

    # datasetnum = '766_2'
    datasetnum = '1082_22'
    # mainpath = f'F:/modelling/Cluster-bkp/merged_prediction_raw_masks'
    mainpath = f'F:/PBC_data/datasets/organelle_mask_5min-new/merged_manual_masks'  ## for manual masks
    mrcpath = f'F:/PBC_data/datasets/raw_image'
    outputpath = f'F:/PBC_data/RadialDistributionFunction_X-ray_tomograms/output'
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



if __name__ == "__main__":
    multi_main_test()





