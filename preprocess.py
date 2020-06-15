import time
from tqdm import tqdm
import numpy as np
import pydicom as dicom
import os
import scipy.ndimage
import matplotlib.pyplot as plt
from utils import apply_window
from pathlib import Path

from skimage import measure, morphology
from collections import defaultdict
from sys import argv
from random import shuffle


# This pixel size/coarseness of the scan differs from scan to scan (e.g. the distance between slices may differ), which can hurt performance of 
# CNN approaches. We can deal with this by isomorphic resampling.
# 
# Below is code to load a scan, which consists of multiple slices, which we simply save in a Python list. Every folder in the dataset is one 
# scan (so one patient). One metadata field is missing, the pixel size in the Z direction, which is the slice thickness. 
# Fortunately we can infer this, and we add this to the metadata.

# Load a volume from the given folder path
def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path) if os.path.splitext(s)[0].isdigit()]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))

    if not slices[0].SliceThickness:
        try:
            slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
        except:
            slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
            
        for s in slices:
            s.SliceThickness = slice_thickness  
    return slices


# The unit of measurement in CT scans is the **Hounsfield Unit (HU)**, which is a measure of radiodensity. 
# CT scanners are carefully calibrated to accurately measure this.  From Wikipedia:
# By default however, the returned values are not in this unit. Let's fix this.
# Some scanners have cylindrical scanning bounds, but the output image is square. 
# The pixels that fall outside of these bounds get the fixed value -2000. The first step is setting these values to 0, which currently corresponds to air. 
# Next, let's go back to HU units, by multiplying with the rescale slope and adding the intercept (which are conveniently stored in the metadata of the scans!).

def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
            
        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)

# # Resampling
# A scan may have a pixel spacing of `[2.5, 0.5, 0.5]`, which means that the distance between slices is `2.5` millimeters. 
# For a different scan this may be `[1.5, 0.725, 0.725]`, 
# this can be problematic for automatic analysis (e.g. using ConvNets).
# A common method of dealing with this is resampling the full dataset to a certain isotropic resolution. 
# If we choose to resample everything to 1.5mm*1.5mm*1.5mm pixels we can use 3D convnets without worrying about learning zoom/slice thickness invariance. 
# Whilst this may seem like a very simple step, it has quite some edge cases due to rounding. Also, it takes quite a while.

def resample(scan_hu, scan_file, scan, new_spacing, verbose=False):
    # Determine current pixel spacing
    spacing = np.array([scan_file[0].SliceThickness] + list(scan_file[0].PixelSpacing), dtype=np.float32)
    if verbose:
        print('Spacing:', spacing)
    resize_factor = spacing / new_spacing
    new_real_shape = scan_hu.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / scan_hu.shape
    new_spacing = spacing / real_resize_factor
    
    scan_hu = scipy.ndimage.interpolation.zoom(scan_hu, real_resize_factor, mode='nearest')
    return scan_hu, new_spacing 

def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)
    counts = counts[vals != bg]
    vals = vals[vals != bg]
    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

# # Lung segmentation
# In order to reduce the problem space, we segment the lungs (and usually some tissue around it).
# It consists of a series of applications of region growing and morphological operations. In this case, 
# we will use only connected component analysis.
# 
# The steps:  
# * Threshold the image (-320 HU is a good threshold, but it doesn't matter much for this approach)
# * Do connected components, determine label of air around person, fill this with 1s in the binary image
# * Optionally: For every axial slice in the scan, determine the largest solid connected component 
# (the body+air around the person), and set others to 0. This fills the structures in the lungs in the mask.
# * Keep only the largest air pocket (the human body has other pockets of air here and there).
def segment_lung_mask(image, fill_lung_structures=True):
    
    # not actually binary, but 1 and 2. 
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -320, dtype=np.int8)+1
    labels = measure.label(binary_image)
    
    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air 
    #   around the person in half
    background_label = labels[0,0,0]
    
    #Fill the air around the person
    binary_image[background_label == labels] = 2
    
    # Method of filling the lung structures (that is superior to something like 
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)
            if l_max is not None: # This slice contains some lung
                binary_image[i][labeling != l_max] = 1
    
    binary_image -= 1 # Make the image actual binary
    binary_image = 1-binary_image # Invert it, lungs are now 1
    
    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None: # There are air pockets
        binary_image[labels != l_max] = 0
    return binary_image

def bbox2_3D(img):
    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))
    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]
    return rmin, rmax, cmin, cmax, zmin, zmax

def preprocess(scan, errors_map, num_slices=224, crop_size=224, voxel_size=1.5, windowing=False, sample_img=True, verbose=True):
    orig_scan = load_scan(scan)
    num_orig_slices = len(orig_scan)
    if num_orig_slices < 50:
        errors_map['insufficient_slices'] += 1
        raise ValueError(scan[-4:] + ': Insufficient muber of slices (<50).')
    orig_scan_np = np.stack([s.pixel_array for s in orig_scan]).astype(np.int16)

    scan_hu = get_pixels_hu(orig_scan)

    # Let's resample our patient's pixels to an isomorphic resolution
    resampled_scan, _ = resample(scan_hu, orig_scan, orig_scan_np, [voxel_size, voxel_size, voxel_size], verbose=verbose)
    if verbose:
        print("Shape before resampling:", scan_hu.shape)
        print("Shape after resampling:", resampled_scan.shape)

    if resampled_scan.shape[0] < 180:
        errors_map['small_z'] += 1
        raise ValueError(scan[-4:] + ': Insufficient number of resampled slices (<200).')

    lung_mask = segment_lung_mask(resampled_scan, True)

    z_min, z_max, x_min, x_max, y_min, y_max = bbox2_3D(lung_mask)
    box_size = (z_max - z_min, x_max - x_min, y_max - y_min)
    if verbose:
        print('Lung bounding box (min, max):', (z_min, z_max), (x_min, x_max), (y_min, y_max))
        print('Bounding box size:', box_size)

    for dim in box_size:
        if dim < 100:
            errors_map['seg_error'] += 1
            raise ValueError(scan[-4:] + ': Segmentation error.')   

    lung_center = np.array([z_min + z_max, x_min + x_max, y_min + y_max]) // 2
    context = np.array([num_slices, crop_size, crop_size])

    img_starts = np.array([max(0, lung_center[i] - context[i] // 2) for i in range(3)])
    img_ends = np.array([min(resampled_scan.shape[i], lung_center[i] + context[i] // 2) for i in range(3)])
    img_size = img_ends - img_starts
        
    starts = context // 2 - img_size // 2
    ends = starts + img_size

    lungs_padded = np.zeros((num_slices, crop_size, crop_size))
    lungs_padded[starts[0]: ends[0], starts[1]: ends[1], starts[2]: ends[2]] = \
            resampled_scan[img_starts[0]: img_ends[0], img_starts[1]: img_ends[1], img_starts[2]: img_ends[2]]

    if verbose:
        print("Final shape", lungs_padded.shape)
        
    if sample_img:
        # Generate an RGB slice for display
        lungs_rgb = np.stack((lungs_padded, lungs_padded, lungs_padded), axis=3)
        lungs_sample_slice = lungs_rgb[lungs_rgb.shape[0] // 2]
    else:
        lungs_sample_slice = None

    return lungs_padded, lungs_sample_slice
    
def walk_dicom_dirs(base_in, base_out=None, print_dirs=True):
    print()
    for root, _, files in os.walk(base_in):
        path = root.split(os.sep)
        if print_dirs:
            print((len(path) - 1) * '---', os.path.basename(root))
        if len(files) >= 50 and os.path.splitext(files[0])[0].isdigit():
            if base_out:
                yield root, base_out + os.path.relpath(root, base_in)
            else:
                yield root

def walk_np_files(base_in, print_dirs=True):
    pathlist = Path(base_in).glob('**/*.np*')
    for path in pathlist:
        np_path = str(path)
        print(np_path)
        yield np_path

def preprocess_all(input_dir, overwrite=False, num_slices=224, crop_size=224, voxel_size=1.5):
    start = time.time()
    scans = os.listdir(input_dir)
    scans.sort()
    errors_map = defaultdict(int)
    base_out = input_dir.rstrip('/') + '_preprocessed/'
    valid_scans = 0

    scans_num = len(list(walk_dicom_dirs(input_dir, base_out, False)))
    for scan_dir_path, out_path in tqdm(walk_dicom_dirs(input_dir, base_out), total=scans_num):
        try:
            out_dir = os.path.dirname(out_path)
            if overwrite or not os.path.exists(out_dir) or not os.listdir(out_dir):
                preprocessed_scan, scan_rgb_sample = \
                    preprocess(scan_dir_path, errors_map, num_slices, crop_size, voxel_size)

                plt.imshow(scan_rgb_sample)
                plt.savefig(out_path + '.png', bbox_inches='tight')
                os.makedirs(out_dir, exist_ok=True)
                np.savez_compressed(out_path + '.npz', data=preprocessed_scan)

            valid_scans += 1
            print('\n++++++++++++++++++++++++\nDiagnostics:')
            print(errors_map.items())

        except FileExistsError as e:
            valid_scans += 1
            print('Exists:', out_path)

        except ValueError as e:
            print('\nERROR!!!!\n', e)

    print('Total scans: {}'.format(scans_num))
    print('Valid scans: {}'.format(valid_scans))
    print('Scans with insufficient slices: {}'.format(errors_map['insufficient_slices']))
    print('Scans with bad segmentation: {}'.format(errors_map['bad_seg']))
    print('Scans with small resampled z dimension: {}'.format(errors_map['small_z']))
    print((time.time() - start) / scans_num, 'sec/image')

def create_train_test_list(positives, negatives, lists_dir, print_dirs=False, split_ratio=0.7, base_dir=''):
    positive_paths = []
    negative_paths = []
    
    for preprocessed_dir, path_list, label in (positives, positive_paths, '1'), (negatives, negative_paths, '0'):
        for root, _, files in os.walk(os.path.join(base_dir, preprocessed_dir)):
            path = root.split(os.sep)
            if print_dirs:
                print((len(path) - 1) * '---', os.path.basename(root))
            for f in files:
                if f.endswith('.npz'):
                    path_list.append((root + '/' + f, label))
        print('\n INFO:', '.npz files with label', label, len(path_list))

    train_list = []
    test_list = []
    shuffle(positive_paths)
    split_pos = round(split_ratio * len(positive_paths))
    shuffle(negative_paths)
    split_neg = round(split_ratio * len(negative_paths))
    train_list = positive_paths[:split_pos] + negative_paths[:split_neg]
    test_list = positive_paths[split_pos:] + negative_paths[split_neg:]
    shuffle(train_list)
    shuffle(test_list)

    lists_dir = os.path.join(base_dir, lists_dir)
    os.makedirs(lists_dir, exist_ok=True)
    with open(lists_dir + '/test.list', 'w') as test_f:
        for path, label in test_list:
            test_f.write(path + ' ' + label + '\n')

    with open(lists_dir + '/train.list', 'w') as train_f:
        for path, label in train_list:
            train_f.write(path + ' ' + label + '\n')


if __name__ == "__main__":
    preprocess_all('/home/daniel_nlp/Lung-Cancer-Risk-Prediction/data/datasets/NLST2', \
        overwrite=False, num_slices=145, voxel_size=1.5)
    # preprocess_all(argv[1])
    # create_train_test_list(positives='confirmed_scanyr_1_filtered-522_volumes', 
    #                         negatives='no_cancer_numscreens_2-971_volumes', 
    #                         lists_dir='lists', 
    #                         base_dir='/home/daniel_nlp/Lung-Cancer-Risk-Prediction/data/datasets/NLST_preprocessed')
