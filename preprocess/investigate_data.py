import os
import cv2
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

ship_dir = '../input'
train_image_dir = os.path.join(ship_dir, 'train_v2')

df = pd.read_csv('./annotation.csv')
images_with_ship = df.image[df.x1.isnull()==False]
images_with_ship = np.unique(images_with_ship.values)

print('There are ' + str(len(images_with_ship)) + ' image files with ship')

directory = 'annotated_images'
if not os.path.exists(directory):
    os.makedirs(directory)

for i in range(10):
    imageId = images_with_ship[i]

    img = cv2.imread(train_image_dir + '/' + imageId)
    rows = df.query('image=="' + imageId + '"')

    print ('Image', imageId)

    for index, row in rows.iterrows():
        x1, y1, x2, y2 = int(row.x1), int(row.y1), int(row.x2), int(row.y2)
        print('Found bbox', [x1, y1, x2, y2])
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    result_path = directory + '/{}.png'.format(os.path.basename(imageId).split('.')[0])
    cv2.imwrite(result_path, img)

print('result saved into ./annotated_images')