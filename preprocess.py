import time
from tqdm import tqdm
import numpy as np # linear algebra
import pydicom as dicom
import os
import scipy.ndimage
import matplotlib.pyplot as plt

from skimage import measure, morphology
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from collections import defaultdict
from sys import argv
from random import shuffle

# # Loading the files
# Dicom is the de-facto file standard in medical imaging. This is my first time working with it, but it seems to be fairly straight-forward.  
# These files contain a lot of metadata (such as the pixel size, so how long one pixel is in every dimension in the real world). 
# 
# This pixel size/coarseness of the scan differs from scan to scan (e.g. the distance between slices may differ), which can hurt performance of 
# CNN approaches. We can deal with this by isomorphic resampling, which we will do later.
# 
# Below is code to load a scan, which consists of multiple slices, which we simply save in a Python list. Every folder in the dataset is one 
# scan (so one patient). One metadata field is missing, the pixel size in the Z direction, which is the slice thickness. 
# Fortunately we can infer this, and we add this to the metadata.

# Load the scans in given folder path
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


# The unit of measurement in CT scans is the **Hounsfield Unit (HU)**, which is a measure of radiodensity. CT scanners are carefully calibrated to accurately measure this.  From Wikipedia:
# 
# ![HU examples][1]
# 
# By default however, the returned values are not in this unit. Let's fix this.
# 
# Some scanners have cylindrical scanning bounds, but the output image is square. The pixels that fall outside of these bounds get the fixed value -2000. The first step is setting these values to 0, which currently corresponds to air. Next, let's go back to HU units, by multiplying with the rescale slope and adding the intercept (which are conveniently stored in the metadata of the scans!).
# 
#   [1]: http://i.imgur.com/4rlyReh.png

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


# Looking at the table from Wikipedia and this histogram, we can clearly see which pixels are air and which are tissue. 
# We will use this for lung segmentation in a bit :)


# # Resampling
# A scan may have a pixel spacing of `[2.5, 0.5, 0.5]`, which means that the distance between slices is `2.5` millimeters. 
# For a different scan this may be `[1.5, 0.725, 0.725]`, 
# this can be problematic for automatic analysis (e.g. using ConvNets)! 
# 
# A common method of dealing with this is resampling the full dataset to a certain isotropic resolution. 
# If we choose to resample everything to 1mm*1mm*1mm pixels we can use 3D convnets without worrying about learning zoom/slice thickness invariance. 
# 
# Whilst this may seem like a very simple step, it has quite some edge cases due to rounding. Also, it takes quite a while.
# 
# Below code worked well for us (and deals with the edge cases):


def resample(scan_hu, scan_file, scan, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing = np.array([scan_file[0].SliceThickness] + list(scan_file[0].PixelSpacing), dtype=np.float32)
    print('Spacing:', spacing)
    resize_factor = spacing / new_spacing
    new_real_shape = scan_hu.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / scan_hu.shape
    new_spacing = spacing / real_resize_factor
    
    scan_hu = scipy.ndimage.interpolation.zoom(scan_hu, real_resize_factor, mode='nearest')
    # scan = scipy.ndimage.interpolation.zoom(scan, real_resize_factor, mode='nearest')
    return scan_hu, new_spacing #, scan


# Please note that when you apply this, to save the new spacing! Due to rounding this may be slightly
#  off from the desired spacing (above script picks the best possible spacing with rounding).
# 


# # 3D plotting the scan
# For visualization it is useful to be able to show a 3D image of the scan. Unfortunately, the
#  packages available in this Kaggle docker image is very limited in this sense, so we will use 
# marching cubes to create an approximate mesh for our 3D object, and plot this with matplotlib. Quite slow and ugly, but the best we can do.

def plot_3d(image, filename, threshold=-300):
    
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)
    
    verts, faces = measure.marching_cubes_classic(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.savefig('out_preprocess/' + filename + '.png')
    # plt.show()


# Spooky!
# 
# # Lung segmentation
# In order to reduce the problem space, we can segment the lungs (and usually some tissue around it).
# It consists of a series of applications of region growing and morphological operations. In this case, 
# we will use only connected component analysis.
# 
# The steps:  
# 
# * Threshold the image (-320 HU is a good threshold, but it doesn't matter much for this approach)
# * Do connected components, determine label of air around person, fill this with 1s in the binary image
# * Optionally: For every axial slice in the scan, determine the largest solid connected component 
# (the body+air around the person), and set others to 0. This fills the structures in the lungs in the mask.
# * Keep only the largest air pocket (the human body has other pockets of air here and there).

def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

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
            
            if l_max is not None: #This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    
    binary_image -= 1 #Make the image actual binary
    binary_image = 1-binary_image # Invert it, lungs are now 1
    
    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None: # There are air pockets
        binary_image[labels != l_max] = 0
 
    return binary_image


# ![pacemaker example][1]
# 
# # Normalization
# Our values currently range from -1024 to around 2000. Anything above 400 is not interesting to us, as these are simply bones with different radiodensity.  A commonly used set of thresholds in the LUNA16 competition to normalize between are -1000 and 400. Here's some code you can use:
# 
# 
#   [1]: http://i.imgur.com/po0eX1L.png

MIN_BOUND = -1000.0
MAX_BOUND = 400.0
    
def apply_window(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image

def bbox2_3D(img):
    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return rmin, rmax, cmin, cmax, zmin, zmax

# # Zero centering
# 
# As a final preprocessing step, it is advisory to zero center your data so that your mean value is 0. To do this you simply subtract the mean pixel value from all pixels. 
# 
# To determine this mean you simply average all images in the whole dataset.  If that sounds like a lot of work, we found this to be around 0.25 in the LUNA16 competition. 
# 
# **Warning: Do not zero center with the mean per image (like is done in some kernels on here). The CT scanners are calibrated to return accurate HU measurements. There is no such thing as an image with lower contrast or brightness like in normal pictures.**

PIXEL_MEAN = 0.25

def zero_center(image):
    image = image - PIXEL_MEAN
    return image


# # What's next? 
# 
# With these steps your images are ready for consumption by your CNN or other ML method :). You can do all these steps offline (one time and save the result), and I would advise you to do so and let it run overnight as it may take a long time. 
# 
# **Tip:** To save storage space, don't do normalization and zero centering beforehand, but do this online (during training, just after loading). If you don't do this yet, your image are int16's, which are smaller than float32s and easier to compress as well.
# 
# **If this tutorial helped you at all, please upvote it and leave a comment :)**

def preprocess(scan, errors_map, context):
    print('scan:', scan)
    
    orig_scan = load_scan(scan)
    num_slices = len(orig_scan)
    if num_slices < 50:
        errors_map['insufficient_slices'] += 1
        raise ValueError(scan[-4:] + ': number of slices is less than 50')
    orig_scan_np = np.stack([s.pixel_array for s in orig_scan]).astype(np.int16)

    scan_hu = get_pixels_hu(orig_scan)

    # Let's resample our patient's pixels to an isomorphic resolution of 1.5 by 1.5 by 1.5 mm.
    print("Shape before resampling\t", scan_hu.shape)
    resampled_scan, spacing = resample(scan_hu, orig_scan, orig_scan_np, [1.5,1.5,1.5])
    print("Shape after resampling\t", resampled_scan.shape)

    if resampled_scan.shape[0] < 200:
        errors_map['small_z'] += 1
        raise ValueError(scan[-4:] + ': resampled Z dimension is less than 200')

    lung_mask = segment_lung_mask(resampled_scan, True)

    # plt.imshow(orig_scan_np[orig_scan_np.shape[0]//2], cmap=plt.cm.gray)
    # plt.savefig('out_preprocess/in_' + scan[-4:] + '.png', bbox_inches='tight')

    z_min, z_max, x_min, x_max, y_min, y_max = bbox2_3D(lung_mask)
    print('Lung bounding box (min,max):', (z_min, z_max), (x_min, x_max), (y_min, y_max))

    z_start = (z_max + z_min) // 2 - context // 2
    z_end = z_start + context
    y_start = (y_max + y_min) // 2 - context // 2
    y_end = y_start + context
    x_start = (x_max + x_min) // 2 - context // 2
    x_end = x_start + context

    print('starts,end:', (z_start, z_end), (x_start, x_end), (y_start, y_end))

    if x_start < 0 or y_start < 0 or y_start < 0:
        errors_map['bad_seg'] += 1
        raise ValueError(scan[-4:] + ': bad segmentation')

    windowed_scan = apply_window(resampled_scan)

    # lung_bounds = windowed_scan[(z_max - z_min) // 2, x_min: x_max, y_min: y_max]
    # plt.imshow(lung_bounds, cmap=plt.cm.gray)
    # plt.savefig('out_preprocess/lung_bounds_' + scan[-4:] + '.png', bbox_inches='tight')

    mid_slice = windowed_scan.shape[0] // 2
    # plt.imshow(windowed_scan[mid_slice], cmap=plt.cm.gray)
    # plt.savefig('out_preprocess/norm_resampled_' + scan[-4:] + '.png', bbox_inches='tight')

    lung_context = windowed_scan[max(0, z_start) : z_end, max(0, x_start) : x_end, max(0, y_start) : y_end]

    # plt.imshow(lung_context[z_start + (context//2)], cmap=plt.cm.gray)
    # plt.savefig('out_preprocess/lung_context_' + scan[-4:] + '.png', bbox_inches='tight')

    lung_rgb = np.stack((lung_context, lung_context, lung_context), axis=3)      
    lung_rgb_sample = lung_rgb[lung_rgb.shape[0]//2]

    lung_rgb_norm = (lung_rgb * 2.0) - 1.0
    print("Final shape\t", lung_rgb_norm.shape, '\n\n')

    return lung_rgb_norm, lung_rgb_sample

def walk_dicom_dirs(base_in, base_out, print_dirs=True):
    for root, dirs, files in os.walk(base_in):
        path = root.split(os.sep)
        if print_dirs:
            print((len(path) - 1) * '---', os.path.basename(root))
        if len(files) >= 50 and os.path.splitext(files[0])[0].isdigit():
            yield root, base_out + os.path.relpath(root, base_in)

def preprocess_all(input_dir, overwrite=False):
    start = time.time()
    scans = os.listdir(input_dir)
    scans.sort()
    errors_map = defaultdict(int)
    base_out = input_dir.rstrip('/') + '_preprocessed/'
    context = 200

    scans_num = len(list(walk_dicom_dirs(input_dir, base_out, False)))
    for scan_dir_path, out_path in tqdm(walk_dicom_dirs(input_dir, base_out), total=scans_num):
        try:
            os.makedirs(os.path.dirname(out_path), exist_ok=overwrite)
            preprocessed_scan, scan_rgb_sample = preprocess(scan_dir_path, errors_map, context)

            plt.imshow(scan_rgb_sample)
            plt.savefig(out_path + '.png', bbox_inches='tight') 

            np.save(out_path, preprocessed_scan)

            print('\n++++++++++++++++++++++++\nDiagnostics:')
            print(errors_map)

        except FileExistsError as e:
            print('Exists:', out_path)

        except ValueError as e:
            print('\nERROR!!!!\n', e)

    print('Total scans: {}'.format(scans_num))
    print('Scans with insufficient slices: {}'.format(errors_map['insufficient_slices']))
    print('Scans with bad segmentation: {}'.format(errors_map['bad_seg']))
    print('Scans with small resampled z dimension: {}'.format(errors_map['small_z']))
    print((time.time() - start) / scans_num, 'sec/image')

def create_train_test_list(positives_dir, negatives_dir, lists_dir, print_dirs=False):
    positive_paths = []
    negative_paths = []
    for preprocessed_dir, path_list, label in (positives_dir, positive_paths, '1'), (negatives_dir, negative_paths, '0'):
        for root, dirs, files in os.walk(preprocessed_dir):
            path = root.split(os.sep)
            if print_dirs:
                print((len(path) - 1) * '---', os.path.basename(root))
            for f in files:
                if f.endswith('.npy'):
                    path_list.append((root + '/' + f, label))

    train_list = []
    test_list = []
    shuffle(positive_paths)
    split_pos = round(0.75 * len(positive_paths))
    shuffle(negative_paths)
    split_neg = round(0.75 * len(negative_paths))
    train_list = positive_paths[:split_pos] + negative_paths[:split_neg]
    test_list = positive_paths[split_pos:] + negative_paths[split_neg:]
    shuffle(train_list)
    shuffle(test_list)

    with open(lists_dir + '/test.list', 'w') as test_f:
        for path, label in test_list:
            test_f.write(path + ' ' + label + '\n')

    with open(lists_dir + '/train.list', 'w') as train_f:
        for path, label in train_list:
            train_f.write(path + ' ' + label + '\n')


if __name__ == "__main__":
    preprocess_all('/home/daniel_nlp/Lung-Cancer-Risk-Prediction/data/datasets/NLST')
    # preprocess_all(argv[1])
    # create_train_test_list('/workdisk/Lung-Cancer-Risk-Prediction/datasets/NLST_preprocessed/conflc=confirmed',
    #     '/workdisk/Lung-Cancer-Risk-Prediction/datasets/NLST_preprocessed/conflc=confirmed_no_cancer',
    #     '/workdisk/Lung-Cancer-Risk-Prediction/i3d/data')
