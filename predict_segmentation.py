import numpy as np
import pandas as pd
import h5py
import io
from PIL import Image

from shutil import copyfile
import argparse
import os
from skimage.color import rgb2hed, hed2rgb
import skimage.measure

from joblib import Parallel, delayed

from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model

import torch

# Custom imports
from utils import logging


def load_patches(file, mags, jpeg=True):
    names = ['patches_20x', 'patches_10x', 'patches_05x']
    with h5py.File(file, 'r') as f:
        plist = []
        for i in mags:
            plist.append(list(f[names[i]]))
        coords = list(f['coordinates'])
        mask = list(f['mask'])

    patches = [[] for i in range(len(mags))]
    for pi in range(len(mags)):
        for p in plist[pi]:
            if jpeg:
                patches[pi].append(jpeg2patch(p))
            else:
                patches[pi].append(p)
    return patches, coords, mask


def jpeg2patch(patch):
    return Image.open(io.BytesIO(np.array(patch)))


def get_segmentation(patches, model, batch_size=32):
    # Number of batches to iterate over
    steps = int(np.ceil(len(patches[0]) / batch_size))
    print('steps: ', steps)
    segmentations = []

    for index in range(steps):
        X_c = []
        X_t = []

        batchlist = []
        for i in range(len(patches)):
            batchlist.append(patches[i][index * batch_size:(index + 1) * batch_size])

        # Transform image in a batch to tensors
        for i in range(len(batchlist[0])):
            patch_t = batchlist[0][i]
            patch_c = batchlist[1][i]

            patch_t = np.array(patch_t)
            patch_c = np.array(patch_c)
            patch_t = patch_t.astype('float') / 255
            patch_c = patch_c.astype('float') / 255

            X_c.append(patch_c)
            X_t.append(patch_t)

        X_c = np.array(X_c)
        X_t = np.array(X_t)

        # Extract the features
        segmentations_batch = model.predict([X_c, X_t])
        segmentations_batch = segmentations_batch.astype(np.uint8)
        segmentations.append(segmentations_batch)
    return np.concatenate(segmentations, axis=0)


def main(file, n, N, model, args):
    # Slide identifier

    # Slide on ssd
    filessd = os.path.join(args.ssd_dir, os.path.basename(file))

    # Check if slide is processed already
    if np.any(not os.path.isfile(outfileseg)) & os.path.isfile(file):
        copyfile(file, filessd)
        # Get tissue patches with their corresponding pixel coordinates from slide
        patches, coords, mask = load_patches(filessd, args.mags)

        # Get features from each tissue patch
        features = get_features(patches, model, batch_size=args.batch_size)
        print("features shape: ", features.shape)
        # Features

        outfilessd = os.path.join(args.ssd_dir, slideid + '.h5')
        utils.store_patches(patches, coords, mask, outfilessd, not args.jpeg)
        copyfile(outfilessd, outfile)

        features = torch.from_numpy(features)
        torch.save(features, outfileseg)

        os.remove(filessd)

        print("Stored tissue patches for {}, {:d} of {:d} processed".format(slideid, n, N))


if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    parser = argparse.ArgumentParser("Whole-Slide-Image (WSI) encoding using neural image compression")

    # Mandatory arguments
    parser.add_argument('--datadir', default=None, type=str, required=True)
    parser.add_argument('--csv', default=None, type=str, required=True)
    parser.add_argument('--savedir', default=None, type=str, required=True)
    parser.add_argument('--modelpath', default=None, type=str, required=True)

    # Optional arguments
    parser.add_argument('--ssd-dir', default=None, type=str, required=False, help='Copy image first from bulk to ssd')
    parser.add_argument('--filetype', default='h5', type=str, required=False, help='Data type patches')
    parser.add_argument('--num-processes', default=4, type=int, help='How many processes to run in parallel')
    parser.add_argument('--mags', default=[0], nargs='+', type=int, help='Which magnifications', required=False)
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for extracting features")
    parser.add_argument("--params", default=None, type=str, choices=[None, 'byol', 'siamese'], help='Load a different (from ImageNet) parameterset')

    args = parser.parse_args()
    num_cores = args.num_processes
    os.makedirs(args.savedir, exist_ok=True)
    os.makedirs(args.ssd_dir, exist_ok=True)

    # Log arguments
    logging.log_command(os.path.basename(__file__), args,
                        os.path.join(args.savedir, 'commands.log'))

    # Load csv file
    df = pd.read_csv(args.csv)
    images = []
    for sid in df['slide_id']:
        for m in args.mags:
            if (not os.path.isfile(os.path.join(args.savedir, '{}_{:d}.pt'.format(sid, m)))):
                images.append(os.path.join(args.datadir, sid + '.' + args.filetype))
                break

    # Load encoder model
    model_full = load_model(args.modelpath)
    model = Model(inputs=model_full.input,
                  outputs=model_full.get_layer("conv2d_31").output)
    del model_full

    # Run jobs in parallel
    # Parallel(n_jobs=num_cores)(delayed(main)(file, i, len(images), model, args) for i, file in enumerate(images))
    [main(file, i, len(images), model, args) for i, file in enumerate(images)]