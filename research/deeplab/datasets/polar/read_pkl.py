import pickle
import matplotlib.image as mpimg
from imageio import imwrite
import numpy as np
np.set_printoptions(threshold=10000)

def save_images(data, name, path_name):
    for i, img in enumerate(data):
        mpimg.imsave(path_name + '/' + name + '_' + str(i) + '.png', img)

def save_segments(data, name):
    for i, img in enumerate(data):
        imwrite('SegmentationClass/' + name + '_' + str(i) + '.png', img)


with open('/om/user/amineh/insideness_data/ID52_DATA_C6_D0_dataset.pkl', 'rb') as file:
    data = pickle.load(file)

test_img = data['test_img']
test_gt = data['test_gt']
train_img = data['train_img']
train_gt = data['train_gt']

for i, img in enumerate(test_gt):
    img[test_img[i] == 1] = 255

for i, img in enumerate(train_gt):
    img[train_img[i] == 1] = 255

print("train size: ", len(train_img))
print("test size: ", len(test_img))

#save_images(test_img, 'test', 'JPEGImages')
#save_images(train_img, 'train', 'JPEGImages')

save_segments(test_gt, 'test')
save_segments(train_gt, 'train')

with open('ImageSets/train.txt', 'a') as train_file:
    for i in range(len(train_img)):
        train_file.write('train_' + str(i) + '.png\n')

with open('ImageSets/test.txt', 'a') as test_file:
    for i in range(len(train_img)):
        test_file.write('test_' + str(i) + '.png\n')

with open('ImageSets/val.txt', 'a') as val_file:
    for i in range(len(test_img)/100):
        val_file.write('test_' + str(i) + '.png\n')
