import skimage
import numpy as np
import os, copy, math, time, json
import matplotlib.pyplot as plt
import tifffile
import mrcfile
import skimage
import pandas as pd
import scipy 
from scipy import stats
from multiprocessing import Pool
from multiprocessing import Process, freeze_support
import skimage.morphology
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import rc
import matplotlib
import matplotlib.ticker as mtick
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib import rcParams
from pylab import *
import pylab as pylab
import scipy.ndimage

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
    return  tiff_nucleus, tiff_wholecell, tiff_isg


def read_isg_image(datasetnum, mainpath, check_ = True):

    for maindir, subdir, file_name_list in os.walk(mainpath, topdown=False):
        filelist = np.array(file_name_list)

    for name in filelist:
        if f'{datasetnum}_' in name and 'isg' in name :
            tif_name = f'{mainpath}/{name}'
            print('current tiff ',tif_name)


    temp_tiff = tifffile.imread(tif_name)

    if check_ :
        plt.imshow(temp_tiff[int(temp_tiff.shape[0]/2)])
        plt.show()
        plt.close()
    
    # return datasetnum
    return  temp_tiff

def read_ratiomask(datasetnum, mainpath, check_ = False):

    for maindir, subdir, file_name_list in os.walk(mainpath, topdown=False):
        filelist = np.array(file_name_list)
    # print(filelist)
    for name in filelist:
        if f'{datasetnum}_' in name and 'ratio_mask.tiff' in name:
            tif_name = f'{mainpath}/{name}'
            print('current tiff ',tif_name)


    temp_tiff = tifffile.imread(tif_name)

    if check_ :
        plt.imshow(temp_tiff[int(temp_tiff.shape[0]/2)])
        plt.show()
        plt.close()
    
    # return datasetnum
    return  temp_tiff



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

  
def mask_rdf(datanum, cellmask, ncmask, isgmask, ratiomask, bins_ = 8):
    
    cytomask = cellmask - ncmask
    cyto_volume = len(np.where(cytomask == 1)[0])
    print(cyto_volume)
    isg_indexlst = list(set(list(isgmask.reshape(1, -1))[0]))
    isg_indexlst.remove(0)
    print(isg_indexlst)
    average_num = (len(isg_indexlst) - 1) / cyto_volume
    print(average_num)
    wholecell_edt = scipy.ndimage.morphology.distance_transform_edt(cellmask)
    
    
    

    # print(os.getcwd())
    # path_ = f'./isg_ratio_position'
    # if not os.path.exists(path_):
    #     os.makedirs(path_)

    isg_ratiolst = []
    for i in range(len(isg_indexlst)):
    # for i in range(n):
        isgindex = isg_indexlst[i]
        # print('isgindex',isgindex)
        tempratio = ratiomask[int(np.mean(np.where(isgmask==isgindex)[0]))][int(np.mean(np.where(isgmask==isgindex)[1]))] [int(np.mean(np.where(isgmask==isgindex)[2]))]
        # print('tempratio',tempratio)
        if tempratio == 0  and cellmask[int(np.mean(np.where(isgmask==isgindex)[0]))][int(np.mean(np.where(isgmask==isgindex)[1]))] [int(np.mean(np.where(isgmask==isgindex)[2]))] <2: tempratio = 1
        isg_ratiolst.append(tempratio)
    
    # isg_ratiolst = [ratiomask[int(np.mean(np.where(isgmask==index)[0]))][int(np.mean(np.where(isgmask==index)[1]))] [int(np.mean(np.where(isgmask==index)[2]))] for index in isg_indexlst[:20]]
    print(isg_ratiolst)
    # with open(f'./isg_ratio_position/{datanum}_isg_ratio_position.json', 'w') as result_file:
    #     json.dump(isg_ratiolst, result_file)
    # index = isg_indexlst
    #isg_ratiolst2 = [[index, ratiomask[int(np.mean(np.where(isgmask==index)[0]))][int(np.mean(np.where(isgmask==index)[1]))] [int(np.mean(np.where(isgmask==index)[2]))]] for index in isg_indexlst]
    
    
    print(len(isg_ratiolst))
    isg_layernum, edges1 = np.histogram(isg_ratiolst, bins= bins_, range=(0,1))
    edges2 = sorted(edges1, reverse= True)
    print(edges1)
    ## generate mask for visualize
    ## wholecell 1 nc 2 ratio 3 granule 4

    
    # path_2 = f'./partial_cell_tiff'
    # if not os.path.exists(path_2):
    #     os.makedirs(path_2)   
    
    
    # ## for 0  index
    # if 0 in isg_ratiolst:
    #     partial_isg_tiff = np.zeros_like(cytomask)

    #     print('isg_indexlst', len(isg_indexlst))
    #     part_indexlst = [isg_indexlst[i] for i in range(len(isg_indexlst[:])) if isg_ratiolst[i] == 0  ]
    #     for index in part_indexlst:
    #         partial_isg_tiff[(np.where(isgmask == index))] = 1

    #     print('partial_isg_tiff', list(set(list(partial_isg_tiff.reshape(1,-1))[0])))   
    #     partial_cell_mask0 = np.zeros_like(cytomask)
    #     partial_cell_mask0[np.where(cellmask== 1)] = 1        
    #     partial_cell_mask0[np.where(ncmask == 1)] = 2
    # #     partial_cell_mask0[np.where(partial_ratio_mask == 1)] = 3
    #     partial_cell_mask0[np.where(partial_isg_tiff == 1)] = 4    
    #     print('check partial mask',  list(set(list(partial_cell_mask0.reshape(1,-1))[0])))
    #     tifffile.imsave(f'./partial_cell_tiff/{datanum}_partial_cell_mask_0.tiff', partial_cell_mask0)
    #     print(f'isg mask 0 saved.')    

    

    # for i in range(len(edges2) - 1):
        
    #     ratio_range = [edges2[i+1], edges2[i]]
    #     print(ratio_range)

    #     partial_ratio_mask = np.zeros_like(cytomask)
    #     print(partial_ratio_mask.shape)
    #     print(ratiomask.shape)
        
    #     partial_ratio_mask[np.where(ratiomask <= float(ratio_range[1]))] = 1
    #     partial_ratio_mask[np.where(ratiomask <= float(ratio_range[0]))] = 0      
        
        
    #     # part_indexlst = []
    #     # for index in isg_indexlst[:]:
    #     #     print(index)
    #     #     print(type(index))
    #     #     print('isg_ratiolst[(np.where(isg_indexlst == index))]', isg_ratiolst[(np.where(isg_indexlst == index))[0][0]]) 
              
    #     #     if  (isg_ratiolst[(np.where(isg_indexlst == index))] > ratio_range[0] and isg_ratiolst[(np.where(isg_indexlst == index)[0][0])] <= ratio_range[1]) :
            
            
    #     part_indexlst = [index for index in isg_indexlst[:] if (isg_ratiolst[(np.where(isg_indexlst == index))[0][0]] > ratio_range[0] and isg_ratiolst[(np.where(isg_indexlst == index)[0][0])] <= ratio_range[1]) ]
    #     print('part indexlst', part_indexlst)
    #     partial_isg_tiff = np.zeros_like(cytomask)
    #     for index in part_indexlst:
    #         partial_isg_tiff[np.where(isgmask == index)] = 1
    #         # tempmask = copy.deepcopy(partial_isg_tiff)
    #         # partial_isg_tiff = tempmask
    #     print('partial isg tiff', list(set(list(partial_isg_tiff.reshape(1, -1))[0])))


    #     partial_cell_mask = np.zeros_like(cytomask)
    #     partial_cell_mask[np.where(cellmask== 1)] = 1        
    #     partial_cell_mask[np.where(ncmask == 1)] = 2
    #     partial_cell_mask[np.where(partial_ratio_mask == 1)] = 3
    #     partial_cell_mask[np.where(partial_isg_tiff == 1)] = 4
        
    #     print('partial_cell_mask ', list(set(list(partial_cell_mask.reshape(1,-1))[0])))
    #     tifffile.imsave(f'./partial_cell_tiff/{datanum}_partial_cell_mask_{ratio_range}.tiff', partial_cell_mask)
    #     print(f'isg mask {ratio_range} saved.')

        

        

    
    def not_background(x):
        return x >0 and x <= 1
    ratiolist = list(filter(not_background, list(ratiomask.reshape(1,-1))[0]))
    print(len(ratiolist))
    layernum, edges2 = np.histogram(ratiolist, bins= bins_, range=(0,1))
    
    # with open(f'./isg_ratio_position/{datanum}_cytosol_ratio_position.json', 'w') as result_file:
    #     json.dump(ratiolist, result_file)
    
    
    g_average = np.zeros(len(isg_layernum))
    radii = np.zeros(len(isg_layernum))

    for i in range(len(g_average)):
        g_average[i] = (isg_layernum[i] / layernum[i]) / average_num
        radii[i] = (edges1[i] + edges1[i+1]) / 2
    
    return (radii, g_average)
    # print(layernum[0])



def rdfplot_main( k1_, seq, output_, save_ = True, check_ = True ,fused_ = False) :

    # Read data file
    # k1 = np.loadtxt("random_ISG-NE_RDF{}.xvg".format(seq))
    k1 = k1_
    # Specifiy environmental parameter
    rc('font',**{'family':'serif','serif':['Arial']})
    
    # Create axes 
    fig = plt.figure(figsize=(8.5,6)) #cm
    fig.subplots_adjust(bottom=0.15)
    fig.subplots_adjust(left=0.18)
    
    # Main figure
    ax1 = plt.subplot2grid((1,1), (0, 0))

    #ax1.set_title("Plot title...")    
    ax1.set_xlabel('$ratio$',fontname="Arial",fontweight="normal",fontsize="20")
    ax1.set_ylabel('g($r$)',fontname="Arial",fontweight="normal",fontsize="20")
    ax1.tick_params(direction='in', pad=6)
    ax1.xaxis.set_label_coords(0.5, -0.1)
    ax1.yaxis.set_label_coords(-0.1, 0.2)
    ax1.set_xlim([0,1])
    ax1.set_ylim([0,3])
    ax1.set_title(f'{seq} plots')
    xmajorLocator   = MultipleLocator(0.1)
    xmajorFormatter = FormatStrFormatter('%.1f')
    xminorLocator   = MultipleLocator(0.05)
    ax1.xaxis.set_major_locator(xmajorLocator)
    ax1.xaxis.set_major_formatter(xmajorFormatter)
    ax1.xaxis.set_minor_locator(xminorLocator)
    ymajorLocator   = MultipleLocator(0.5)
    ymajorFormatter = FormatStrFormatter('%.2f')
    yminorLocator   = MultipleLocator(0.5)
    ax1.yaxis.set_major_locator(ymajorLocator)
    ax1.yaxis.set_major_formatter(ymajorFormatter)
    ax1.yaxis.set_minor_locator(yminorLocator)
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)
        tick.label.set_fontname("Arial")
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(20)
        tick.label.set_fontname("Arial")
    for axis in ['bottom','left']:ax1.spines[axis].set_linewidth(2)
    for axis in ['top','right']:ax1.spines[axis].set_linewidth(0)
    for line in ax1.xaxis.get_ticklines():
        line.set_markersize(5)
        line.set_markeredgewidth(2)
    for line in ax1.yaxis.get_ticklines():
        line.set_markersize(5)
        line.set_markeredgewidth(2)
    rcParams['xtick.direction'] = 'out'
    rcParams['ytick.direction'] = 'out'
    for line in ax1.yaxis.get_minorticklines():
        line.set_markersize(2.5)
        line.set_markeredgewidth(2)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')
    #ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
    #ax1.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
    #xtick_locs=[0.00,0.03,0.05,0.10,0.20]
    #xtick_lbls=['0.00','0.03',' 0.05','0.10','0.20']
    #plt.xticks(xtick_locs,xtick_lbls)
    #ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
    #ax1.set_yticks([0.015,0.025,0.035,0.045,0.055])
    #ax1.set_axis_bgcolor('none')
    #ax1.grid(True)
    ax1.plot((0,5),(1,1),'k',linestyle="-",linewidth=0.5)
    #ax1.plot((0,1000),(5.4,5.4),'grey',linestyle=":",linewidth=1.5)
    #ax1.plot((0,1000),(5.6,5.6),'grey',linestyle=":",linewidth=1.5)
    #ax1.plot((0,1000),(5.8,5.8),'grey',linestyle=":",linewidth=1.5)
    
    #12:278
    
    # Plot 
    # ax1.plot(k1[:,0]/100000, k1[:,1],linestyle='-',c='k',linewidth=1,alpha=1)
    # ax1.plot(k1[0], k1[1],linestyle='-',c='k',linewidth=1,alpha=1)

    dist_half = ((k1[0][-1]- k1[0][0])/ (len(k1[0])-1)/2)
    # print(dist_half)
    for index in range(len(k1[1])):
        'horizon line'
        x = np.linspace( (k1[0][index] - dist_half), (k1[0][index] + dist_half),100)
        y = k1[1][index] + 0 * x
        ax1.plot(x, y, c = 'black')

    for index2 in range(len(k1[1])-1):
        y = np.linspace( k1[1][index2], k1[1][index2+1] ,100)
        x = k1[0][index2]+ dist_half + 0 * y
        ax1.plot(x, y, c = 'k')        
        


    #ax1.plot((0,np.mean(k2[13:278,1]))(0.6,np.mean(k2[13:278,1])),linestyle='-',c='r',linewidth=2)
    #ax1.errorbar(k2[:,0],k2[:,1], yerr=k2[:,2],marker='o',linestyle='none',markersize=6, fmt='-',capsize=4, elinewidth=2,linewidth=2,c='green')
    #ax1.plot(k2[:,0],k2[:,1],marker='o',linestyle='none',c='green',markersize=8)
    
    # build a rectangle in axes coords
    left, width = .25, .5
    bottom, height = .25, .5
    right = left + width
    top = bottom + height
    ax1.text(0.04*(left+right), 0.96*(bottom+top), 'NE-ISG',horizontalalignment='left',verticalalignment='center',fontsize=20,fontname="Arial",fontweight="normal",color='k',transform=ax1.transAxes)
    #ax1.text(0.02*(left+right), 0.89*(bottom+top), 'anodic layer',horizontalalignment='left',verticalalignment='center',fontsize=20,fontname="Arial",fontweight="normal",color='chocolate',transform=ax1.transAxes)
    #ax1.text(0.02*(left+right), 0.83*(bottom+top), 'bilayer',horizontalalignment='left',verticalalignment='center',fontsize=20,fontname="Arial",fontweight="normal",color='k',transform=ax1.transAxes)
    
    if check_:
        plt.show()
        plt.close()
    
    if save_ :

        fig.savefig("{}/{}_voxel_normalized_isg_RDF_v2.png".format(output_,seq),dpi=1200)
    print('done')









def rdf_mask_main(datasetinfo_):
    datasetnum = datasetinfo_['datasetnum']
    mainpath = datasetinfo_['mainpath']
    isgpath = datasetinfo_['isgpath']
    mrcpath = datasetinfo_['mrcpath']
    ratiomaskpath = datasetinfo_['ratiomaskpath']
    outputpath = datasetinfo_['outputpath']

 

    print(datasetnum)

    ratiomask = read_ratiomask(datasetnum, ratiomaskpath, check_ = True)
    temp_nc_tiff, temp_pm_tiff, tt_isg_tiff= read_image(datasetnum, mainpath, check_ = True)   
    temp_isg_tiff = read_isg_image(datasetnum, isgpath, check_ = True)
    isg_tiff = temp_isg_tiff * temp_pm_tiff

    
    cyto_mask = temp_pm_tiff - temp_nc_tiff
    cyto_volume = np.sum(cyto_mask)
    cyto_ratiomask = cyto_mask * ratiomask

    cyto_position_tiff2 = list(cyto_ratiomask.reshape(1,-1))[0]
    cyto_ratiolst = [ratio for ratio in cyto_position_tiff2 if ratio != 0]
    cytolayernum, edges2 = np.histogram(cyto_ratiolst, bins= 8, range=(0,1))


    isg_tiff = tt_isg_tiff
    isg_position_tiff = isg_tiff * ratiomask
    mrcfile = read_mrcimage(datasetnum, mrcpath, check_ = False)


    print('isg tiff size', isg_tiff.shape)
    print('mrcfile shape', mrcfile.shape)
    print('ratiomask shape', ratiomask.shape)



    isg_position_tiff2 = list(isg_position_tiff.reshape(1,-1))[0]
    print('len isg',len(isg_position_tiff2))
    isg_ratiolst = [ratio for ratio in isg_position_tiff2 if ratio != 0]
    print('len isg ratio',len(isg_ratiolst))
    isg_layernum, edges1 = np.histogram(isg_ratiolst, bins= 8, range=(0,1))


    position_num = len(isg_ratiolst)
    isg_volume = position_num


    average_num = len(isg_ratiolst) / cyto_volume
    
    g_average = np.zeros(len(isg_layernum))
    radii =  np.zeros(len(isg_layernum))

    print(len(g_average))
    print(isg_layernum)
    print(cytolayernum)
    
    for i in range(len(g_average)):
        g_average[i] = (isg_layernum[i] / cytolayernum[i]) / average_num
        radii[i] = (edges1[i] + edges1[i+1]) / 2
    
    isg_g_average = g_average


    print(isg_g_average)


    RDF_data = [radii ,isg_g_average]

    # RDF_data = mask_rdf(datasetnum, temp_pm_tiff, temp_nc_tiff, isg_tiff, ratiomask, bins_ = 8)

    print(RDF_data)
    print(RDF_data[0])
    print(isg_volume)

    rdf_info_ = {}
    rdf_info_['dataset'] = datasetnum
    rdf_info_['histgram'] = [RDF_data[0].tolist(), RDF_data[1].tolist()]
    rdf_info_['isg volume'] = isg_volume
    with open(f'{outputpath}/{datasetnum}_isg_histgram_info.json', 'w') as f:
        json.dump(rdf_info_, f)

    rdfplot_main(RDF_data, datasetnum, output_ = outputpath, save_ = True, check_ = True ,fused_ = False)
    



def multi_main_test():

    # datasetnum = '766_2'
    datasetnum = '766_8'
    # mainpath = f'F:/modelling/Cluster-bkp/merged_prediction_raw_masks'
    # mainpath = f'F:/PBC_data/datasets/organelle_mask_5min-new/merged_manual_masks'  ## for manual masks
    mainpath = f'F:/modelling/Cluster-bkp/merged_prediction_raw_masks'

    # isgpath = f''
    isgpath = f'F:/modelling/Cluster-bkp/merged_prediction_mask_match_isg'

    # mrcpath = f'F:/PBC_data/datasets/raw_image'
    mrcpath = f'F:/PBC_data/datasets/raw_image'
    
    ratiomaskpath = f'F:/modelling/Cluster-bkp'

    # outputpath = f'F:/PBC_data/RadialDistributionFunction_X-ray_tomograms/output'
    outputpath = f'F:/PBC_data/RadialDistributionFunction_X-ray_tomograms/output'


    datasetinfo_ = dict()
    datasetinfo_['datasetnum'] = datasetnum
    datasetinfo_['mainpath'] = mainpath
    datasetinfo_['mrcpath'] = mrcpath
    datasetinfo_['isgpath'] = isgpath
    datasetinfo_['ratiomaskpath'] = ratiomaskpath
    datasetinfo_['outputpath'] = outputpath



    rdf_mask_main(datasetinfo_)



if __name__ == "__main__":
    multi_main_test()









