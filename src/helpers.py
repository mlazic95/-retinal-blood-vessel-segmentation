import pandas as pd
import numpy as np
import seaborn as sns
import random
from pylab import imread, imshow, imsave, plt
from skimage.filters import threshold_li, threshold_minimum
from skimage.color import rgb2gray
from skimage import exposure 
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, confusion_matrix
from enum import Enum
import matplotlib
import os
class Dataset(Enum):
    """Three different datasets."""
    DRIVE = 0
    STARE = 1
    CHASE = 2

DRIVE_TRAINING_IMAGES_PATH = '../../data/DRIVE/training/images/'
DRIVE_TRAINING_SEG_1_PATH = '../../data/DRIVE/training/1st_manual/'
DRIVE_TRAINING_MASK_PATH = '../../data/DRIVE/training/mask/'
DRIVE_TEST_IMAGES_PATH = '../../data/DRIVE/test/images/'
DRIVE_TEST_SEG_1_PATH = '../../data/DRIVE/test/1st_manual/'
DRIVE_TEST_SEG_2_PATH = '../../data/DRIVE/test/2nd_manual/'
DRIVE_TEST_MASK_PATH = '../../data/DRIVE/test/mask/'

CHASE_IMAGES_PATH = '../../data/CHASEDB1/images/'
CHASE_SEG_1_PATH = '../../data/CHASEDB1/segmentation/1st/'
CHASE_SEG_2_PATH = '../../data/CHASEDB1/segmentation/2nd/'
CHASE_MASK_PATH = '../../data/CHASEDB1/masks/'

STARE_IMAGES_PATH = '../../data/STARE/seg-img/'
STARE_SEG_AH_PATH = '../../data/STARE/labels-ah/'
STARE_SEG_VK_PATH = '../../data/STARE/label-vk/'

'''
drive_images = load_images_from_folder(DRIVE_TRAINING_IMAGES_PATH)
drive_segmentation = load_images_from_folder(DRIVE_TRAINING_SEG_1_PATH)
drive_mask = load_images_from_folder(DRIVE_TRAINING_MASK_PATH)

stare_images = load_images_from_folder(STARE_IMAGES_PATH)
stare_segmentation = load_images_from_folder(STARE_SEG_AH_PATH)

chase_image = load_images_from_folder(CHASE_IMAGES_PATH)[0]
chase_segmentation = load_images_from_folder(CHASE_SEG_1_PATH)[0]
chase_mask = load_images_from_folder(CHASE_MASK_PATH)[0]
'''


def load_images_from_folder(folder_path):
    file_names = os.listdir(folder_path)
    file_names.sort()
    images = []
    for filename in file_names:
        try:
            image = imread(folder_path + filename)
            images.append(image)
        except:
            continue

    return images

def save_image(image_path, image):
    imsave(image_path, image)

def create_chase_mask(image, image_path=None):
    im_gray = rgb2gray(image)
    thresh_val = threshold_li(im_gray)
    mask = np.where(im_gray > thresh_val, 1, 0)

    # Make sure the larger portion of the mask is considered background
    if np.sum(mask==0) < np.sum(mask==1):
        mask = np.where(mask, 0, 1)
    if image_path:
        save_image(image_path, mask)

def create_stare_mask(image, image_path=None):
    im_gray = rgb2gray(image)
    thresh_val = threshold_minimum(im_gray)
    mask = np.where(im_gray > thresh_val, 1, 0)
    # Make sure the larger portion of the mask is considered background
    if np.sum(mask==0) < np.sum(mask==1):
        mask = np.where(mask, 0, 1)
    if image_path:
        save_image(image_path, mask)
    return mask


def create_patches(images, segmentations, masks, n, patches_per_image):
    nrOfImages = 0
    for (i, image) in enumerate(images):
        patch_images, patch_labels = get_image_pathes(image, segmentations[i], n, patches_per_image, Dataset.DRIVE, masks[i])
        for (j, patch) in enumerate(patch_images):
            name = str(i) + '_' + str(j)
            save_image('../../data/DRIVE/training/patches/' + name + '.png', patch)
            save_image('../../data/DRIVE/training/patchLabels/' + name + '.png' , patch_labels[j])
            nrOfImages += 1


def global_contrast_normalization(image, s, lmda, epsilon):

    mean = np.mean(image)
    image = image - mean

    contrast = np.sqrt(lmda + np.mean(image**2))

    image = s * image / max(contrast, epsilon)
    #image = 255*(image - np.min(image))/np.ptp(image).astype(int)
    image = (image - np.min(image))/np.ptp(image)

    return image

def image_to_non_overlapping_patches(image, mask, m, dataset):
    mask_condition = get_mask_condition_function(dataset)
    patches, indexes = [], []
    for x in range(image.shape[0]//m):
        for y in range(image.shape[1]//m):
            x_start = x * m
            x_end = x_start + m
            y_start = y * m
            y_end = y_start + m
            mask_patch = mask[x_start:x_end, y_start:y_end]
            if mask_condition(mask_patch):
                continue
            else:
                image_patch = image[x_start:x_end, y_start:y_end]
                patches.append(image_patch)
                indexes.append((x,y))
    return patches, indexes


def patches_to_image(patches, indexes, m, image_width, image_height):
    image = np.zeros((image_width, image_height))
    np.set_printoptions(threshold=10)
    for (i, (x,y)) in enumerate(indexes):
        x_start = x * m
        x_end = x_start + m
        y_start = y * m
        y_end = y_start + m
        image[x_start:x_end, y_start:y_end] = patches[i]
    return image


def adaptive_equalization(image):
    image = rgb2gray(image)
    img_adapteq = exposure.equalize_adapthist(image, clip_limit=0.03)
    return img_adapteq



def contrast_stretching(image):
    image = rgb2gray(image)
    p2, p98 = np.percentile(image, (2, 98))
    img_rescale = exposure.rescale_intensity(image, in_range=(p2, p98))
    return img_rescale




def get_patches_with_mask(image, labels, m, n_patches, mask, condition_function):
    patches = []
    label_patches = []
    while len(patches) < n_patches:
        random_x, random_y = (random.randint(int(m/2),image.shape[0] - int(m/2) - 1), random.randint(int(m/2), image.shape[1] - int(m/2) - 1))
        start_x = random_x - int(m/2)
        end_x = random_x + int(m/2)
        start_y = random_y - int(m/2)
        end_y = random_y + int(m/2)
        mask_patch = mask[start_x : end_x,start_y:end_y]
        if condition_function(mask_patch):
            continue
        else:
            image_patch = image[start_x : end_x,start_y:end_y]
            label_patch = labels[start_x : end_x,start_y:end_y]
            patches.append(image_patch)
            label_patches.append(label_patch)
    return patches, label_patches

def get_mask_condition_function(dataset):
    if dataset == Dataset.DRIVE:
        def mask_condition(mask_patch):
            return any(0 in row for row in mask_patch)
    else:
        def mask_condition(mask_patch):
            return any(any(col[0] > 0.3 for col in row) for row in mask_patch)
    return mask_condition


def histogram_equalization(image):
    image = rgb2gray(image)
    img_eq = exposure.equalize_hist(image)
    return img_eq


def get_image_pathes(image, labels, m, n_patches, dataset, mask=None):
    """"""
    if dataset == Dataset.DRIVE:
        if mask is None:
            raise Exception('Mask must be provided for DRIVE dataset')
        return get_patches_with_mask(image, labels, m, n_patches, mask, get_mask_condition_function(dataset))
    elif dataset == Dataset.CHASE:
        if mask is None:
            raise Exception('Mask must be provided for CHASE dataset')
        def mask_condition(mask_patch):
            return any(any(col[0] > 0.3 for col in row) for row in mask_patch)
        return get_patches_with_mask(image, labels, m, n_patches, mask, get_mask_condition_function(dataset))
    return

def roc_auc(image, label, mask):
    masked_image, masked_label = create_image_mask(image, label, mask)
    score = roc_auc_score(masked_label, masked_image)
    return score

def create_image_mask(image, label, mask):
    masked_image = []
    masked_label = []
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            if mask[x,y] == 255:
                masked_image.append(image[x,y])
                masked_label.append(label[x,y])
    masked_label = [1 if x > 0 else 0 for x in masked_label]
    return masked_image, masked_label

def accuracy(image, label, mask):
    masked_image, masked_label = create_image_mask(image, label, mask)
    acc = accuracy_score(masked_label, masked_image)
    return acc

def sensitivity(image, label, mask):
    masked_image, masked_label = create_image_mask(image, label, mask)
    sens = recall_score(masked_label, masked_image)
    return sens

def specificity(image, label, mask):
    masked_image, masked_label = create_image_mask(image, label, mask)
    tn, fp, fn, tp = confusion_matrix(masked_label, masked_image).ravel()
    return tn / (tn + fp)
    
def segment_whole_image(image, mask, m, dataset, model):
    image = rgb2gray(image)
    image_prob_map = []
    for x in range(image.shape[0]):
        image_prob_map.append([])
        for y in range(image.shape[1]):
            image_prob_map[x].append([])
    mask_condition = get_mask_condition_function(dataset)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            if mask[x][y] == 0 or len(image_prob_map[x][y]) > 0:
                continue
            near_mask = False
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if mask[x + i][y + j] == 0:
                        near_mask = True
                        break
            if near_mask:
                continue
            patch_result = create_patch_for_pixel(image, (x,y), mask, m, mask_condition)
            if not patch_result:
                continue
            patch_image, patch_direction = patch_result
            patch_image = patch_image.reshape(1, m, m, 1)
            y_pred = model.predict(patch_image).reshape(32,32)
            #y_pred_thr = y_pred.copy()
            #y_pred_thr[y_pred > 0.5] = 1
            #y_pred_thr[y_pred <= 0.5] = 0
            if patch_direction == (1,1):
                for row in range(m):
                    for col in range(m):
                        image_prob_map[x + row][y + col].append(y_pred[row][col])
            elif patch_direction == (-1,1):
                for row in range(m - 1, -1, -1):
                    for col in range(m):
                        image_prob_map[x + (row - m + 1)][y + col].append(y_pred[row][col])
            elif patch_direction == (1,-1):
                for row in range(m):
                    for col in range(m - 1, -1, -1):
                        image_prob_map[x + row][y + (col - m + 1)].append(y_pred[row][col])
            elif patch_direction == (-1,-1):
                for row in range(m - 1, -1, -1):
                    for col in range(m - 1, -1, -1):
                        image_prob_map[x + (row - m + 1)][y + (col - m + 1)].append(y_pred[row][col])
            '''
            fig, ax = plt.subplots(1,2, figsize=(20,20))
            ax[0].imshow(patch_image.reshape(32,32))
            ax[1].imshow(y_pred_thr)
            plt.show() '''

    final_image = np.zeros((image.shape[0], image.shape[1]))
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            final_image[x][y] = 0 if len(image_prob_map[x][y]) == 0 else sum(image_prob_map[x][y]) / len(image_prob_map[x][y])
    final_image[final_image > 0.5] = 1
    final_image[final_image <= 0.5] = 0
    
    return final_image


def create_patch_for_pixel(image, pixel, mask, m, mask_condition):
    # Check down-right
    x_start = pixel[0]
    x_end = x_start + m
    y_start = pixel[1]
    y_end = y_start + m
    if not (x_start < 0 or y_start < 0 or x_end >= image.shape[0] or y_end >= image.shape[1]):
        mask_patch = mask[x_start:x_end, y_start:y_end]
        if not mask_condition(mask_patch):
            return image[x_start:x_end, y_start:y_end], (1,1)

    # Check down-left
    x_end = pixel[0] + 1
    x_start = x_end - m
    y_start = pixel[1]
    y_end = y_start + m
    if not (x_start < 0 or y_start < 0 or x_end >= image.shape[0] or y_end >= image.shape[1]):
        mask_patch = mask[x_start:x_end, y_start:y_end]
        if not mask_condition(mask_patch):
            return image[x_start:x_end, y_start:y_end], (-1,1)

    # Check up-right
    x_start = pixel[0]
    x_end = x_start + m
    y_end = pixel[1] + 1
    y_start = y_end - m
    if not (x_start < 0 or y_start < 0 or x_end >= image.shape[0] or y_end >= image.shape[1]):
        mask_patch = mask[x_start:x_end, y_start:y_end]
        if not mask_condition(mask_patch):
            return image[x_start:x_end, y_start:y_end], (1,-1)

    # Check down-left
    x_end = pixel[0] + 1
    x_start = x_end - m
    y_end = pixel[1] + 1
    y_start = y_end - m
    if not (x_start < 0 or y_start < 0 or x_end >= image.shape[0] or y_end >= image.shape[1]):
        mask_patch = mask[x_start:x_end, y_start:y_end]
        if not mask_condition(mask_patch):
            return image[x_start:x_end, y_start:y_end], (-1,-1)
        
        
def segment_image(image, mask, model, m, dataset):
    image = rgb2gray(image)
    patches, indexes = image_to_non_overlapping_patches(image, mask, m, Dataset.DRIVE)
    patches = np.array(patches)
    patches = patches.reshape(patches.shape[0], m, m, 1)
    y_pred = model.predict(patches).reshape(patches.shape[0], 32,32)
    y_pred_thr = y_pred.copy()
    y_pred_thr[y_pred > 0.5] = 1
    y_pred_thr[y_pred <= 0.5] = 0
    image = patches_to_image(y_pred_thr, indexes, m, image.shape[0], image.shape[1] )
    return image    




    


    
    
    
    
    
    
