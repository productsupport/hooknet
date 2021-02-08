import numpy as np
import glob
import os
import pandas as pd
import openslide
import matplotlib.path as mp
import xml.etree.ElementTree as et
import scipy.ndimage

from typing import List, Any

from skimage.color import rgb2gray
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.measure import regionprops, label

from sklearn import preprocessing
from sklearn.model_selection import train_test_split


def tissue_segment(img):
    imgg = rgb2gray(img)
    # Transform to optical density
    imgg[imgg == 0] = 10 ** -10
    imgg = -np.log10(imgg)
    # Create mask
    idx = np.array((imgg > 0.1) & (np.var(img, axis=2) > 0.001))

    # Exclude tiles at boundary
    bsz = int(np.min(img.shape) * 0.03)
    if bsz > 0:
        idxbound = np.zeros(idx.shape, dtype=bool)
        idxbound[bsz:-bsz, bsz:-bsz] = True
        idxbound = np.invert(idxbound)
        idx[idxbound] = False

    # Area size of each region
    areas = [r.area for r in regionprops(label(idx))]
    # Remove small holes and small loose areas
    idx = remove_small_holes(remove_small_objects(idx, min_size=np.max(areas) * 0.1, connectivity=1),
                             area_threshold=10, connectivity=1)

    if np.sum(idx.flatten()>30):
        # Remove very elongated objects (probably boundary artifacts)
        labels = label(idx)
        lenratio = np.array(
            [(r.major_axis_length - r.minor_axis_length) / r.major_axis_length for r in regionprops(label(idx))])
        idxlabel = np.where(lenratio > 0.95)[0]
        for i in idxlabel:
            idx[labels == i + 1] = False
    return idx


# XML reader for annotations written by the ASAP WSI-viewer (https://computationalpathologygroup.github.io/ASAP/)
class Annotation:
    def __init__(self, xmlfile, names=None, InvTumourClass=None, OtherClass=None):
        self.xmlfile = xmlfile
        tree = et.parse(self.xmlfile)
        root = tree.getroot()
        annotations = root.iter('Annotation')
        xy = []
        label = []
        for elem in annotations:
            # Loop over annotations
            label_temp = elem.get('PartOfGroup')
            if label_temp in OtherClass:
                label_temp = 'Other'
                label.append(label_temp)
            elif label_temp in InvTumourClass:
                label_temp = 'InvTumour'
                label.append(label_temp)
            else:
                label.append(label_temp)
            coordinates = []
            for k in elem[0]:
                # Loop over xy-coordinates of polygon points
                x = k.get('X')
                y = k.get('Y')
                coordinates.append([x, y])
                # print(i,x,y)
            xy.append(np.array(coordinates, dtype='float'))

        # print('labels in the annotated file: ', list(np.unique(label)))
        le = preprocessing.LabelEncoder()
        if names is not None:
            le.fit(names)
            # print('classes: ', list(le.classes_))
            idx = [j for j, word in enumerate(label) if word in names]
            # print("idx: ", idx)
            label = [label[i] for i in idx]
            # print('label: ', label)
            classnames = list(le.classes_)
            xy = [xy[i] for i in idx]
        else:
            le.fit(label)
        self.coords = xy
        self.numlabel = len(np.unique(label))
        self.labelid = le.transform(label)
        #         self.labelnames = le.inverse_transform(np.sort(np.unique(self.labelid)))
        self.labelnames = le.inverse_transform(self.labelid)
        self.classnames = classnames
#         print('self.labelid: ', self.labelid)
#         print('self.labelnames: ', self.labelnames)


def anno_coords(xy, anno, label, full_tile_anno=True, tsz=256):
    """

    :param xy:      Coordinates of all tiles in WSI
    :param anno:    Coordinates of the annotation polygons
    :param tsz:     Tile size
    :return:        xyanno:     Coordinates of the tiles that fall within the annotations
                    annoi:      Annotation index per tile (can be more than one)
    """
    # Expand coordinates to include the coordinates of all corners of tile
    xycorners = np.vstack((xy,
                           np.hstack((xy[:, 0].reshape((-1, 1)) + tsz, xy[:, 1].reshape((-1, 1)))),
                           np.hstack((xy[:, 0].reshape((-1, 1)), xy[:, 1].reshape((-1, 1)) + tsz)),
                           xy + tsz))

    # print("xycorners: ", xycorners.shape)

#     Check if tile at original tsz falls within annotations
    annoidx = []
    for i, a in enumerate(anno):
        mpath = mp.Path(a)
        idx = mpath.contains_points(xycorners)
        if full_tile_anno:
            # idx_reshape = idx.reshape(4, -1)
            # print("idx_reshape: ", idx_reshape.shape)
            # print("idx_reshape: ", idx_reshape[:, 0])
            # print("idx_reshape: ", idx_reshape[:, 100])
            # print("idx_reshape: ", idx_reshape)
            # print('sum idx: ', np.sum(idx.reshape((4, -1)).all(axis=0)))
            annoidx.append(idx.reshape((4, -1)).all(axis=0))

        else:
            annoidx.append(idx.reshape((4, -1)).any(axis=0))

    annoidx = np.array(annoidx)

    xyanno = xy[annoidx.any(axis=0), :]
    annoidx = annoidx[:, annoidx.any(axis=0)]

    # nm = 5
    # xc = [];
    # yc = []
    # for i in xyanno:
    #     mx, my = np.meshgrid(np.linspace(i[0] - 1.5 * tsz, i[0] + 2.5 * tsz, nm),
    #                          np.linspace(i[1] - 1.5 * tsz, i[1] + 2.5 * tsz, nm))
    #     xc.append(mx.flatten())
    #     yc.append(my.flatten())
    #
    # xycorners = np.hstack((np.array(xc).reshape((-1, 1)),
    #                        np.array(yc).reshape((-1, 1))))
    #
    # annoidx = []
    # for i, a in enumerate(anno):
    #     mpath = mp.Path(a)
    #     idx = mpath.contains_points(xycorners)
    #     annoidx.append(idx.reshape((-1, nm ** 2)).any(axis=1))
    #
    # annoidx = np.array(annoidx)
    #
    # if full_tile_anno:
    #     xyanno = xyanno[annoidx.all(axis=0), :]
    #     annoidx = annoidx[:, annoidx.all(axis=0)]
    # else:
    #     xyanno = xyanno[annoidx.any(axis=0), :]
    #     annoidx = annoidx[:, annoidx.any(axis=0)]

    annoi = []
    annoilabel = []
    for i in annoidx.transpose():
        annoilabel.append(label[np.nonzero(i)[0]])
        annoi.append(np.nonzero(i)[0])
    return xyanno, annoi, annoilabel


def sort_anno(anno):
    """Sorts annotations such that annotations within annotations end up last

    :param anno:   Annotation object
    :return:       Sorted annotation object
    """
    # Number of annotations
    nanno = len(anno.coords)
    annoidx = np.arange(nanno)
    annoinside = np.zeros((nanno, nanno), dtype=bool)
    for i in range(nanno):
        for j in range(nanno):
            # Create polygon of annotation
            a = mp.Path(anno.coords[annoidx[i]])
            if i != j:
                # Check if polygon contains points from other annotations
                annoinside[i, j] = a.contains_points(anno.coords[annoidx[j]]).any()
    # Re-order based on the number of annotations within annotations
    isinside = np.sum(annoinside, axis=0)
    # print("isinside: ", isinside.shape)
    anno.coords = np.array(anno.coords, dtype=object)[np.argsort(isinside)]
    anno.labelid = np.array(anno.labelid)[np.argsort(isinside)]

    return anno


def get_label(xy, anno, annoidx, labelid, tsz, method='multi', debug=False):
    """Extract tissue masks from annotations at a given tile

    :param xy:          Coordinates of top-left corner of tile
    :param anno:        Coordinates of the annotation polygons
    :param annoidx:     Indices of the annotations within the specified tile
    :param labelid:     Label identifiers of the tissue types
    :param tsz:         Tile size
    :return:            label0, label1, label2. Tissue masks at 20x, 10x and 5x magnification, respectively.
    """
    if method == 'single':
        label0 = np.zeros((tsz, tsz))
        for i in annoidx:
            mx, my = np.meshgrid(np.linspace(xy[0], xy[0] + tsz, tsz),
                                 np.linspace(xy[1], xy[1] + tsz, tsz))
            path = mp.Path(anno[i])
            label = path.contains_points(np.array([mx.flatten(), my.flatten()]).transpose()).reshape((tsz, tsz))
            label0[label] = labelid[i] + 1
        return label0
    elif method == 'multi':
        label0 = np.zeros((tsz, tsz))
        # label1 = np.zeros((tsz, tsz))
        label2 = np.zeros((tsz, tsz))

        if debug:
            label1_  = np.zeros((tsz, tsz))
            # label4_  = np.zeros((2 * tsz, 2 * tsz))
            label16_ = np.zeros((4 * tsz, 4 * tsz))

        for i in annoidx:
            mx, my = np.meshgrid(np.linspace(xy[0], xy[0] + tsz, tsz),
                                 np.linspace(xy[1], xy[1] + tsz, tsz))
            path = mp.Path(anno[i])
            label = path.contains_points(np.array([mx.flatten(), my.flatten()]).transpose()).reshape((tsz, tsz))

            label0[label] = labelid[i] + 1
            if debug:
                label = path.contains_points(np.array([mx.flatten(), my.flatten()]).transpose()).reshape((tsz, tsz))
                label1_[label] = labelid[i] + 1

            # mx, my = np.meshgrid(np.linspace(xy[0] - int(0.5 * tsz), xy[0] + int(1.5 * tsz), 2 * tsz),
            #                      np.linspace(xy[1] - int(0.5 * tsz), xy[1] + int(1.5 * tsz), 2 * tsz))
            # label = path.contains_points(np.array([mx.flatten(), my.flatten()]).transpose()).reshape(2 * tsz, 2 * tsz)
            # if debug:
            #     label4_[label] = labelid[i] + 1
            #
            # label = scipy.ndimage.interpolation.zoom(label, 0.5,
            #                                          order=0, mode='nearest')
            # label1[label] = labelid[i] + 1


            mx, my = np.meshgrid(np.linspace(xy[0] - int(1.5 * tsz), xy[0] + int(2.5 * tsz), 4 * tsz),
                                 np.linspace(xy[1] - int(1.5 * tsz), xy[1] + int(2.5 * tsz), 4 * tsz))
            label = path.contains_points(np.array([mx.flatten(), my.flatten()]).transpose()).reshape((4 * tsz, 4 * tsz))
            if debug:
                label16_[label] = labelid[i] + 1

            label = scipy.ndimage.interpolation.zoom(label, 0.25,
                                                     order=0, mode='nearest')
            label2[label] = labelid[i] + 1


        if debug:
            return label1_, label16_, label0, label2
        return label0, label2
    elif method == 'augment':
        label2 = np.zeros((4 * tsz, 4 * tsz))

        for i in annoidx:
            path = mp.Path(anno[i])
            mx, my = np.meshgrid(np.linspace(xy[0] - int(1.5 * tsz), xy[0] + int(2.5 * tsz), 4 * tsz),
                                 np.linspace(xy[1] - int(1.5 * tsz), xy[1] + int(2.5 * tsz), 4 * tsz))
            label = path.contains_points(np.array([mx.flatten(), my.flatten()]).transpose()).reshape((4 * tsz, 4 * tsz))
            label2[label] = labelid[i] + 1
        return label2


def label_check(label, thr=0.2):
    """Check if sufficient labelled tissue is available (>%thr) and return largest annotation

    :param label:       labeled image
    :param thr:         Pixel threshold as a fraction of total pixelcount in label
    :return: flag:      True is labelled tissue is >%thr
             labint:    Largest annotation
             count:     Pixelcount for largest annotation
    """
    labint, labcount = np.unique(label, return_counts=True)
    labcount = labcount[labint != 0]
    if labcount.size == 0:
        flag = False
        return flag, [], []
    else:
        labint = labint[labint != 0]
        flag = max(labcount) > thr * np.prod(label.shape)
        return flag, labint[np.argmax(labcount)], labcount[np.argmax(labcount)]


def limit_tiles(xy, idx, idxlabel, maxtiles):
    idxlabel = np.array([i[0] for i in idxlabel])
    classuni, count = np.unique(idxlabel, return_counts=True)
    if np.any(count > maxtiles):
        keep = np.ones(len(idx), dtype=bool)
        for i in classuni[count > maxtiles]:
            classind = np.where(idxlabel == i)[0]
            removeid = np.random.permutation(classind)[0:-maxtiles]
            keep[removeid] = False
        xy = xy[keep, :]
        idx = [i for index, i in enumerate(idx) if keep[index]]
    return xy, idx


# colors for segmented classes
colorB = [255, 128, 232, 70, 156, 153, 153,  30,   0,  35, 152]
colorG = [255,  64,  35, 70, 102, 153, 153, 170, 220, 142, 251]
colorR = [255, 128, 244, 70, 102, 190, 153, 250, 220, 107, 152]
CLASS_COLOR = list()
for i in range(0, len(colorB)):
    CLASS_COLOR.append([colorR[i], colorG[i], colorB[i]])
COLORS = np.array(CLASS_COLOR, dtype="float32")


def give_color_to_seg_img(seg, n_classes):
    if len(seg.shape)==3:
        seg = seg[:,:,0]
    seg_img = np.zeros( (seg.shape[0],seg.shape[1], 3) ).astype('float')
    #colors = sns.color_palette("hls", n_classes) #DB
    colors = COLORS
    for c in range(n_classes):
        segc = (seg == c)
        seg_img[:,:, 0] += (segc*( colors[c][0]/255.0 ))
        seg_img[:,:, 1] += (segc*( colors[c][1]/255.0 ))
        seg_img[:,:, 2] += (segc*( colors[c][2]/255.0 ))

    return(seg_img)


def colour_augment_hed(x):
    ah = 0.95 + np.random.random() * 0.1
    bh = -0.05 + np.random.random() * 0.1
    ae = 0.95 + np.random.random() * 0.1
    be = -0.05 + np.random.random() * 0.1
    ad = 0.95 + np.random.random() * 0.1
    bd = -0.05 + np.random.random() * 0.1
    hed = rgb2hed(x)

    hed[:, :, 0] = ah * hed[:, :, 0] + bh
    hed[:, :, 1] = ae * hed[:, :, 1] + be
    hed[:, :, 2] = ad * hed[:, :, 2] + bd

    x = hed2rgb(hed)
    x = np.clip(x, 0, 1.0).astype(np.float32)
    # x = x.astype(np.float32)
    return x


def store_patches(patches, coords, mask, file, jpeg=False):
    d = os.path.dirname(file)
    sid = os.path.splitext(os.path.basename(file))[0]
    ftmp = os.path.join(d, 'tmp_{}.jpg'.format(sid))
    patchesjpg = []
    for patchesi in patches:
        pjpgi = []
        for p, pa in enumerate(patchesi):
            if jpeg:
                pjpgi.append(patch2jpeg(pa, ftmp))
            else:
                pjpgi.append(pa)
        pjpgi = np.concatenate(pjpgi)
        patchesjpg.append(pjpgi)
    if jpeg:
        os.remove(ftmp)

    # Write hdf5 file
    pdsetnames = ['segmentations_20x']
    with h5py.File(file, 'w') as f:
        for i, pjpg in enumerate(patchesjpg):
            f.create_dataset(pdsetnames[i], data=pjpg)
        f.create_dataset('mask', data=mask)
        f.create_dataset('coordinates', data=coords)


def patch2jpeg(patch, name):
    patch.save(name, 'JPEG', quality=80)
    with open(name, 'rb') as img_f:
        image_file = img_f.read()
        return np.asarray(image_file).reshape(1, )