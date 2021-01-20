import openslide
import numpy as np
import os
import glob
import pandas as pd
import scipy.ndimage

import matplotlib.pyplot as plt
import matplotlib.path as mp

import h5py
import argparse

from matplotlib.colors import ListedColormap

from sklearn import preprocessing as skpreproc
import xml.etree.ElementTree as et
from joblib import Parallel, delayed

from utils import utils


def main(slidePath, annotationPath, outFolder, counter, N, full_tile_anno=True, tsz=256,
         separate_files=True, debug=False):

    # WSI
    slide = openslide.OpenSlide(slidePath)
    svsbase = os.path.splitext(os.path.basename(slidePath))[0]
#     print("processing slide: ", svsbase)

    # WSI properties
    width, height = slide.dimensions

#     classnames = ['Micro calcification', 'Muscle', 'MC', 'Mastopatic', 'CIS', 'Necrosis', 'NormalEpithelial', 'None',
#                   'IDC', 'Stroma', 'Lymfocyten', 'Reactive changes', 'Adipose', 'RedBlood', 'Benigne', 'ILC']
    
    # classnames = ['Mastopatic', 'CIS', 'Necrosis', 'NormalEpithelial', 'IDC', 'Stroma', 'Lymfocyten',
    #               'Adipose', 'RedBlood', 'ILC']
    classnames = ['CIS', 'IDC', 'ILC', 'Stroma', 'Adipose', 'Other']
    otherNames = ['Mastopatic', 'Necrosis', 'NormalEpithelial', 'RedBlood']

    # Load xml annotations and sort
    xmlbase = os.path.splitext(os.path.basename(annotationPath))[0]

    annotation = utils.Annotation(annotationPath, names=classnames, otherNames=otherNames)
    annotation = utils.sort_anno(annotation)
    
    # xy-coordinates of top left corner of each tile (Exclude boundary tiles: start from 2*tsz)
    xint = np.arange(2*tsz, width - 3*tsz, tsz)
    yint = np.arange(2*tsz, height - 3*tsz, tsz)

    # Expand to matrices
    mx, my = np.meshgrid(xint, yint)

    # xy-coordinates of all tile in vector format
    xytiles = np.vstack((mx.flatten(), my.flatten())).transpose()

    # find coordinates of the tiles that fall within the annotations + their annotation indices and labels (can be more than one)
    xytiles_anno, tileannoidx, tileannoidxlabel = utils.anno_coords(xytiles, annotation.coords, annotation.labelid,
                                                                    full_tile_anno, tsz)
    # print("xytiles_anno: ", xytiles_anno.shape)
    # print("tileannoidx: ", tileannoidx[0:10])
    # print("tileannoidxlabel: ", tileannoidxlabel[0:10])
    # tileannoidx is index of the label in labelid

    
    # Start sampling tiles from WSI
    patches =  []
    labels = []
    for i, xytile in enumerate(xytiles_anno):

        # Compute the annotation masks for the given tiles
        # label0 = utils.get_label(xytile, annotation.coords, tileannoidx[i], annotation.labelid, tsz, method='single')
        if debug:
            label1_, label16_, label0, label16 = utils.get_label(
                xytile, annotation.coords, tileannoidx[i], annotation.labelid, tsz, method='multi', debug=debug)
        else:
            label0, label16 = utils.get_label(
                xytile, annotation.coords, tileannoidx[i], annotation.labelid, tsz, method='multi', debug=debug)

        patch0 = np.array(slide.read_region((xytile[0], xytile[1]), 0, (tsz, tsz)))  # dtype is: dtype=uint8
        patch0 = patch0[:, :, :3]

        if debug:
            patch4 = np.array(slide.read_region((
                xytile[0]- int(0.5 * tsz), xytile[1]- int(0.5 * tsz)), 0, (2 * tsz, 2 * tsz)))  # dtype is: dtype=uint8
            patch4 = patch4[:, :, :3]

            patch16 = np.array(slide.read_region((
                xytile[0]- int(1.5 * tsz), xytile[1]- int(1.5 * tsz)), 0, (4 * tsz, 4 * tsz)))  # dtype is: dtype=uint8
            patch16 = patch16[:, :, :3]


            fig, axs = plt.subplots(2, 3, figsize=(18, 12))
            axs[0][0].imshow(patch0)
            axs[0][1].imshow(patch4)
            axs[0][2].imshow(patch16)
            axs[1][0].imshow(label1_)
            axs[1][1].imshow(label4_)
            axs[1][2].imshow(label16_)
            fig.suptitle("original sizes")
            plt.show()

        # # resample 20 tile size
        # patch4 = scipy.ndimage.interpolation.zoom(patch4.astype('float') , (0.50, 0.50, 1.0),
        #                                           order=3, mode='nearest')
        # patch16 = scipy.ndimage.interpolation.zoom(patch16.astype('float'), (0.25, 0.25, 1.0),
        #                                            order=3, mode='nearest')
        # patch4 = np.clip(patch4, 0, 255)
        # patch16 = np.clip(patch16, 0, 255)

        # patch4 = slide.read_region((
        #     xytile[0] - int(0.5 * tsz), xytile[1] - int(0.5 * tsz)), 0, (2 * tsz, 2 * tsz))  # dtype is: dtype=uint8
        # patch4 = patch4.resize((tsz, tsz))
        # patch4 = np.array(patch4)[:, :, :3]

        patch16 = slide.read_region((
            xytile[0] - int(1.5 * tsz), xytile[1] - int(1.5 * tsz)), 0, (4 * tsz, 4 * tsz))  # dtype is: dtype=uint8
        patch16 = patch16.resize((tsz, tsz))
        patch16 = np.array(patch16)[:, :, :3]

        # print(np.unique(patch0, return_counts=True))
        # print(np.unique(patch4, return_counts=True))
        # print(np.unique(patch16, return_counts=True))

        if debug:
            print("label0: ", np.unique(label0, return_counts=True))
            print("label4: ", np.unique(label4, return_counts=True))
            print("label16: ", np.unique(label16, return_counts=True))
            fig, axs = plt.subplots(2, 3, figsize=(18, 12))
            axs[0][0].imshow(patch0)
            axs[0][1].imshow(patch4)
            axs[0][2].imshow(patch16)
            axs[1][0].imshow(label0)
            axs[1][1].imshow(label4)
            axs[1][2].imshow(label16)
            fig.suptitle("resampled to tile size")
            plt.show()

        # Check if sufficient tissue is annotated and report the dominating label and pixelcount
        flag, labelclass, labelcount = utils.label_check(label0, thr=0.20)
        # print('flag: ', flag)

        if flag:
            labels_temp = [label0, label16]
            patches_temp = [patch0, patch16]
            if separate_files:
                pdsetnames = ['patches_20x', 'patches_5x']
                outputPath = os.path.join(outFolder, svsbase + "_" + str(i) + ".h5")
                with h5py.File(outputPath, 'w') as f:
                    for i, setname in enumerate(pdsetnames):
                        grp = f.create_group(setname)
                        grp.create_dataset('segmentation', data=labels_temp[i], dtype='uint8')
                        grp.create_dataset('patch', data=patches_temp[i])
                    f.close()
            else:
                labels.append(label0)
                patches.append(patch)

    if len(patches) > 0:
        # Write hdf5 file
        pdsetnames = 'patches_20x'
        outputPath = os.path.join(outFolder, svsbase + ".h5")
        with h5py.File(outputPath, 'w') as f:
            grp = f.create_group(pdsetnames)
            grp.create_dataset('segmentations', data=labels, dtype='uint8')
            grp.create_dataset('patches', data=patches)
            f.close()

    print("Stored tissue patches for {}, {:d} of {:d} processed".format(svsbase, counter, N))

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser("Extract data and label patches from Whole-Slide-Image (WSI)")
       
    # Mandatory arguments
    parser.add_argument('--svsfolder', default=None, type=str, required=True)
    parser.add_argument('--xmlfolder', default=None, type=str, required=True)
    parser.add_argument('--outfolder', default=None, type=str, required=True)
    parser.add_argument('--csv', default=None, type=str, required=True)
    
    # Optional arguments
    parser.add_argument("--tsz", default=256, type=int, help='Size of the patches (size x size x 3)')
    parser.add_argument('--num_processes', default=4, type=int, help='how many processes to extract in parallel')
    
    args = parser.parse_args()
    num_cores = args.num_processes
    os.makedirs(args.outfolder, exist_ok=True)
    
    # Load csv file
    df = pd.read_csv(args.csv)

    images = []
    for sid in df['slide_id']:
        if not os.path.isfile(os.path.join(args.outfolder, '{}.h5'.format(sid))):
            slidePath = os.path.join(args.svsfolder, sid + ".svs")
            annotationPath = os.path.join(args.xmlfolder, sid + ".xml")
#             outputPath = os.path.join(args.outfolder, sid + ".h5")
            outFolder = args.outfolder
            images.append([slidePath, annotationPath, outFolder])  # outputPath

    tsz = args.tsz
    num_cores = args.num_processes

    # Run jobs in parallel
    Parallel(n_jobs=num_cores)(
        delayed(main)(path[0], path[1], path[2], counter, len(images)) for counter, path in enumerate(images))

