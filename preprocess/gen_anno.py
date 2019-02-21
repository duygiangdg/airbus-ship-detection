import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from skimage.morphology import label
from skimage.measure import regionprops

def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction

df = pd.read_csv('../input/train_ship_segmentations_v2.csv')

all_images = df.ImageId
all_images = np.unique(all_images)
print('There are ' + str(len(all_images)) + ' image files')

images_with_ship = df.ImageId[df.EncodedPixels.isnull()==False]
images_with_ship = np.unique(images_with_ship.values)
print('There are ' + str(len(images_with_ship)) + ' image files with masks')

images_no_ship = df.ImageId[df.EncodedPixels.isnull()]
images_no_ship = np.unique(images_no_ship.values)
print('There are ' + str(len(images_no_ship)) + ' image files without masks')

random.shuffle(all_images)

anno = []
for image in tqdm(all_images):
    if np.random.randint(0, 6) > 0:
        dataset = 'train'
    else:
        dataset = 'test'
    masks = df[df.ImageId == image]
    for index, row in masks.iterrows():
        rle = row['EncodedPixels']
        if isinstance(rle, float):
            anno.append([image, None, None, None, None, dataset])
        else:
            mask = rle_decode(rle)
            lbl = label(mask)
            props = regionprops(lbl)
            for prop in props:
                anno.append([image, prop.bbox[1], prop.bbox[0], prop.bbox[3], prop.bbox[2], dataset])

anno_df = pd.DataFrame(anno)
anno_df.columns = ['image', 'x1', 'y1', 'x2', 'y2', 'dataset']
anno_df.to_csv('annotation.csv', index=False)