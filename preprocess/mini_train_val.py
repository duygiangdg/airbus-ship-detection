import numpy as np
import pandas as pd
import random

masks = pd.read_csv('../input/train_ship_segmentations_v2.csv')

all_images = masks.ImageId
all_images = np.unique(all_images)
print('There are ' + str(len(all_images)) + ' image files')

images_with_ship = masks.ImageId[masks.EncodedPixels.isnull()==False]
images_with_ship = np.unique(images_with_ship.values)
print('There are ' + str(len(images_with_ship)) + ' image files with masks')

images_no_ship = masks.ImageId[masks.EncodedPixels.isnull()]
images_no_ship = np.unique(images_no_ship.values)
print('There are ' + str(len(images_no_ship)) + ' image files without masks')

mini_data_with_ship = random.sample(list(images_with_ship), 3000)
mini_data_no_ship = random.sample(list(images_no_ship), 1000)

mini_train = images_with_ship[0:2000]
mini_val = mini_data_with_ship[2000:3000] + mini_data_no_ship
random.shuffle(mini_val)

print('There are ' + str(len(mini_train)) + ' image files in mini train data set')
print('There are ' + str(len(mini_val)) + ' image files in mini val data set')

mini_train_masks = masks[masks.ImageId.isin(mini_train)]
mini_val_masks = masks[masks.ImageId.isin(mini_val)]

print('Train set: \n' + mini_train_masks.head().to_string(index=False))
print('Val set: \n' + mini_val_masks.head().to_string(index=False))

mini_train_masks.to_csv('train.csv', index=False)
mini_val_masks.to_csv('val.csv', index=False)
