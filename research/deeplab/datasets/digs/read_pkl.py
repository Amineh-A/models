import pickle
import matplotlib.image as mpimg
from scipy import misc

def save_images(data, name, path_name):
    for i, img in enumerate(data):
        mpimg.imsave(path_name + '/' + name + '_' + str(i) + '.png', img)

def save_segments(data, name):
    for i, img in enumerate(data):
        misc.imsave('SegmentationClass/' + name + '_' + str(i) + '.png', img)


with open('ID53_DATA_C7_D0_dataset.pkl', 'rb') as file:
    data = pickle.load(file)

test_img = data['test_img']
test_gt = data['test_gt']
train_img = data['train_img']
train_gt = data['train_gt']

#save_images(test_img, 'test', 'JPEGImages')
#save_images(train_img, 'train', 'JPEGImages')

# save_segments(test_gt, 'test')
# save_segments(train_gt, 'train')

# save_images(test_gt, 'test', 'SegmentationClass')
# save_images(train_gt, 'train', 'SegmentationClass')


# with open('ImageSets/train.txt', 'a') as train_file:
#     for i in range(len(train_img)):
#         train_file.write('train_' + str(i) + '.png\n')
#
# with open('ImageSets/test.txt', 'a') as test_file:
#     for i in range(len(train_img)):
#         test_file.write('test_' + str(i) + '.png\n')

