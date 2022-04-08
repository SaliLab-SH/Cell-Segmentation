'''
1. generate index masks
2. find_pairs 
3. cgal to separate shells
4. generate rdfs

'''


import sys, os, time 
import numpy as np
import tifffile
import time, os, math, json, copy,csv, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile
import mrcfile
import scipy.ndimage as ndimage
from scipy.ndimage import gaussian_filter, gaussian_laplace
from skimage.feature import peak_local_max
from scipy import spatial
from skimage.morphology import watershed
import skimage
import operator
from operator import sub




def shrink_mask_with_extend(mask, times_):
    n = times_
    if n == 1: 
        return mask
    else:
        mid_n = int((n - 1)//2)
        shape_x, shape_y, shape_z = mask.shape[0], mask.shape[1], mask.shape[2]
        if shape_x % n != 0: shape_x += n
        if shape_y % n != 0: shape_y += n
        if shape_z % n != 0: shape_z += n

        new_mask = np.zeros([shape_x//n, shape_y//n, shape_z//n])
        for x in range(new_mask.shape[0]):
            for y in range(new_mask.shape[1]):
                for z in range(new_mask.shape[2]):
                    new_mask[x][y][z] = mask[n*x + mid_n][n*y + mid_n][n*z + mid_n]
                    
        # plt.imshow(new_mask[77])
        # plt.show()
        # plt.close()

        shape = new_mask.shape
        temptiff = np.zeros([shape[0] + 2,shape[1] + 2, shape[2] + 2])
        temptiff[1:-1,1:-1,1:-1] = new_mask

    return temptiff




def shrink_mask_with_include(mask, times_):
    n = times_

    if n == 1: 
        return mask
    else:
        mid_n = int((n - 1)//2)
        shape_x, shape_y, shape_z = mask.shape[0], mask.shape[1], mask.shape[2]
        if shape_x % n != 0: shape_x += n
        if shape_y % n != 0: shape_y += n
        if shape_z % n != 0: shape_z += n

        new_mask = np.zeros([shape_x//n, shape_y//n, shape_z//n])
        for x in range(new_mask.shape[0]):
            for y in range(new_mask.shape[1]):
                for z in range(new_mask.shape[2]):
                    roi = [slice(n * x , n * (x + 1)), 
                            slice(n * y , n * (y + 1)), 
                            slice(n * z , n * (z + 1))]
                    cube_ = mask[roi[0],roi[1],roi[2]]
                    if np.sum(cube_)!= 0:
                        # print('not empty')
                        new_mask[x][y][z] = 1

        temptiff = new_mask            
        # plt.imshow(new_mask[77])
        # plt.show()
        # plt.close()

        # shape = new_mask.shape
        # temptiff = np.zeros([shape[0] + 2,shape[1] + 2, shape[2] + 2])
        # temptiff[1:-1,1:-1,1:-1] = new_mask

    return temptiff



def shrink_mask(mask, times_):
    n = times_

    if n == 1: 
        return mask
    else:
        mid_n = int((n - 1)//2)
        shape_x, shape_y, shape_z = mask.shape[0], mask.shape[1], mask.shape[2]
        if shape_x % n != 0: shape_x += n
        if shape_y % n != 0: shape_y += n
        if shape_z % n != 0: shape_z += n

        new_mask = np.zeros([shape_x//n, shape_y//n, shape_z//n])
        for x in range(new_mask.shape[0]):
            for y in range(new_mask.shape[1]):
                for z in range(new_mask.shape[2]):
                    roi = [slice(n * x , n * (x + 1)), 
                            slice(n * y , n * (y + 1)), 
                            slice(n * z , n * (z + 1))]
                    cube_ = mask[roi[0],roi[1],roi[2]]
                    new_mask[x][y][z] = cube_[int(cube_.shape[0]/2)][int(cube_.shape[1]/2)][int(cube_.shape[2]/2)]

        temptiff = new_mask            
        # plt.imshow(new_mask[77])
        # plt.show()
        # plt.close()

        # shape = new_mask.shape
        # temptiff = np.zeros([shape[0] + 2,shape[1] + 2, shape[2] + 2])
        # temptiff[1:-1,1:-1,1:-1] = new_mask

    return temptiff




    n = times_
    if n == 1: 
        return mask
    else:
        mid_n = int((n - 1)//2)
        shape_x, shape_y, shape_z = mask.shape[0], mask.shape[1], mask.shape[2]
        if shape_x % n != 0: shape_x += n
        if shape_y % n != 0: shape_y += n
        if shape_z % n != 0: shape_z += n

        new_mask = np.zeros([shape_x//n, shape_y//n, shape_z//n])
        for x in range(new_mask.shape[0]):
            for y in range(new_mask.shape[1]):
                for z in range(new_mask.shape[2]):
                    new_mask[x][y][z] = mask[n*x + mid_n][n*y + mid_n][n*z + mid_n]
                    
        # plt.imshow(new_mask[77])
        # plt.show()
        # plt.close()
        temptiff = new_mask    

    return temptiff





def back_originaltiff(tiff, times_ = 3):
    # print('hi')
    time1 = time.time()
    shape = tiff.shape
    newtiff = np.zeros([times_*shape[0], times_*shape[1], times_*shape[2]])
    for i in range(newtiff.shape[0]):
        for j in range(newtiff.shape[1]):
            for k in range(newtiff.shape[2]):
                newtiff[i][j][k] = tiff[i//times_][j//times_][k//times_]
    time2 = time.time()
    print('time',time2 - time1)
    return newtiff




def classified_point(point1, center, nn_ = 0.2):
    point1 = np.array(point1)
    center = np.array(center)
    vect_2 = sub(point1, center)
    length_2 = np.sqrt( np.square(vect_2[0]) + np.square(vect_2[1]) + np.square(vect_2[2]) )
    v2_std = vect_2 / length_2
    temp_name = [int(v2_std[0]//nn_), int(v2_std[1]//nn_), int(v2_std[2]//nn_)]
    return temp_name

def classified_region(ne, center, nn_ = 0.2):
    
    min_ = int(-1 // nn_)
    max_ = int(1 // nn_)    
    vect_dict = {}
    # print(min_, max_)
    
    for x in range(min_,max_+1):
        for y in range(min_,max_+1):
            for z in range(min_,max_+1):
                temp_name = [x,y,z]
                vect_dict[f'{temp_name}'] = []   




    for i in range(0,len(ne)):
        v1 = list(map(sub, ne[i], center))
        length_ = np.sqrt( np.square(v1[0]) + np.square(v1[1]) + np.square(v1[2]) )
        v1_std = v1 / length_
        nn = nn_
        temp_name = [int(v1_std[0]//nn), int(v1_std[1]//nn), int(v1_std[2]//nn)]
        # print(temp_name)
        vect_dict[f'{temp_name}'].append(ne[i].tolist())

    return vect_dict

def min_angle(pm, pm_all, nepoint, center):

    '''
    find the minimal angle between two vectors center-to-p1 and center-to-p2
    '''

    tmp = []
    for i in range(0,len(pm)):
        v1 = list(map(sub, pm[i], center))
        v2 = list(map(sub, nepoint, center))
        # print(v1)
        # print(v2)
        cross = round(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)),4) # to obtain the angle by math.acos
        # print('cross', cross)
        tmp.append(cross)
        if cross == 1:
            return pm[i]
    # print(tmp)
    if len(tmp) == 0:
        for i in range(0,len(pm_all)):
            v1 = list(map(sub, pm_all[i], center))
            v2 = list(map(sub, nepoint, center))
            # print(v1)
            # print(v2)
            cross = round(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)),4) # to obtain the angle by math.acos
            # print('cross', cross)
            tmp.append(cross)
            if cross == 1:
                return pm_all[i]
        return pm_all[tmp.index(max(tmp))]

    return pm[tmp.index(max(tmp))]



def generate_ratio_mask(ne_mask, cyto_mask, classified_mask, times_ = 3 ):
    ne_orginal = copy.deepcopy(ne_mask)
    # if times_ == 1:
        # ne_mask =ne_orginal

    # else: 
 
    ne_mask = shrink_mask_with_include(ne_orginal, times_)
    classified_mask = shrink_mask(classified_mask, times_)
    
    necoord = np.where(ne_mask == 1)
    centcoord = [int(np.mean(necoord[0])), int(np.mean(necoord[1])), int(np.mean(necoord[2]))]
    temp_cent_mask = np.ones_like(ne_mask)
    temp_cent_mask[centcoord[0]][centcoord[1]][centcoord[2]] = 0
    centcoord_edt = ndimage.distance_transform_edt(temp_cent_mask)
    del temp_cent_mask

    classified_mask_binary = np.zeros_like(classified_mask)
    classified_mask_binary[np.where(classified_mask != 0 )] = 1


    # plt.imshow(centcoord_edt[centcoord[0]])
    # plt.show()
    # plt.close()

    centcoord_edt = centcoord_edt * classified_mask_binary
    del classified_mask_binary 

    plt.imshow(centcoord_edt[centcoord[0]])
    plt.show()
    plt.close()
    

    indexlst = list(set(list(classified_mask.reshape(1, -1))[0]))
    indexlst.remove(0)

    ## change classified mask on 1 d 
    # classified_mask_origional  = copy.deepcopy(classified_mask)
    # classified_mask = classified_mask.reshape(1, -1)
    # centcoord_edt_line = centcoord_edt.reshape(1,-1)
    min_mask = np.zeros_like(classified_mask)
    max_mask = np.zeros_like(classified_mask)

    print('num', len(indexlst))
    print(np.max(indexlst))


    for index in indexlst:
        if index % 200 == 0 :
            print(index)

        ## cut area and get num 
        indexcoords = np.where(classified_mask == index)
        roi2 = [slice(np.min(indexcoords[0]), np.max(indexcoords[0]) + 1),
                slice(np.min(indexcoords[1]), np.max(indexcoords[1]) + 1),
                slice(np.min(indexcoords[2]), np.max(indexcoords[2]) + 1)]

        cube_ = classified_mask[roi2[0],roi2[1],roi2[2]]
        centcoord_edt_cube = centcoord_edt[roi2[0],roi2[1],roi2[2]]
        
        cube_binary = np.zeros_like(cube_)
        cube_binary[np.where(cube_ == index)] = 1
        # print(np.sum(cube_binary))
        indexmask_edt = cube_binary * centcoord_edt_cube

        # current_mask = np.zeros_like(classified_mask)
        # current_mask[np.where(classified_mask == index)] = 1
        # indexmask_edt = current_mask * centcoord_edt

        distrange = list(set(list(indexmask_edt.reshape(1, -1))[0]))
        if 0 in distrange:
            distrange.remove(0)
        # print(distrange)
        min_ = np.min(distrange) 
        max_ = np.max(distrange)
        # print('min', min_)
        # print('max', max_)
       

        min_mask[np.where(classified_mask == index)] = min_
        max_mask[np.where(classified_mask == index)] = max_

    # min_mask = min_mask.reshape(centcoord_edt.shape[0], centcoord_edt.shape[1], centcoord_edt.shape[2])
    # max_mask = max_mask.reshape(centcoord_edt.shape[0], centcoord_edt.shape[1], centcoord_edt.shape[2])


    ratiomask = (centcoord_edt - min_mask) / (max_mask - min_mask + 10e-9)
    ratiomask[np.where(ratiomask> 1)] = 1 
    del min_mask, max_mask, centcoord_edt
    print('ratiomask max', np.max(ratiomask))
    print('ratiomask min', np.min(ratiomask))

    plt.imshow(ratiomask[int(ratiomask.shape[0]/2)])
    plt.show()
    plt.close()

    ratiomask_mask_extendsize = back_originaltiff(ratiomask, times_ = times_ )
    ratiomask_mask_originalsize = np.zeros_like(ne_orginal)
    ratiomask_mask_originalsize = ratiomask_mask_extendsize[:ne_orginal.shape[0], :ne_orginal.shape[1],:ne_orginal.shape[2]]
    ratiomask_mask_originalsize = ratiomask_mask_originalsize * cyto_mask
    del ratiomask_mask_extendsize

    return ratiomask_mask_originalsize


