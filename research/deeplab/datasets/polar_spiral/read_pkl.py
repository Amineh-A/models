import pickle
import matplotlib

def save_images(data, name):
    for i in range(1):
        img = data[i]
        matplotlib.image.imsave('JPEGImages/' + name + '_' + str(i) + '.png', img)

with open('ID53_DATA_C7_D0_dataset.pkl', 'rb') as file:
    data = pickle.load(file)

test_img = data['test_img']
test_gt = data['test_gt']
train_img = data['train_img']
train_gt = data['train_gt']

# save_images(test_img, 'test_img')
# save_images(test_gt, 'test_gt')
# save_images(train_img, 'train_img')
# save_images(train_gt, 'train_gt')


# with open('ImageSets/train.txt', 'a') as train_file:
#     for i in range(len(train_img)):
#         train_file.write('train_img_' + str(i) + '.png\n')
#         train_file.write('train_gt_' + str(i) + '.png\n')
#
# with open('ImageSets/test.txt', 'a') as test_file:
#     for i in range(len(train_img)):
#         test_file.write('test_img_' + str(i) + '.png\n')
#         test_file.write('test_gt_' + str(i) + '.png\n')


