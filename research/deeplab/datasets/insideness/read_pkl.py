import pickle
import numpy as np
import matplotlib.pyplot as plt
import random
np.set_printoptions(threshold=np.nan)

def save_images(data, name):
    for i in range(len(data)):
        img = data[i]
        plt.imshow(img)
        plt.savefig('JPEGImages/' + name + '_' + str(i) + '.png', bbox_inches='tight', pad_inches=0)


with open('ID53_DATA_C7_D0_dataset.pkl', 'rb') as file:
    data = pickle.load(file)

test_img = data['test_img']
test_gt = data['test_gt']
train_img = data['train_img']
train_gt = data['train_gt']

axes = plt.axes()
axes.get_xaxis().set_visible(False)
axes.get_yaxis().set_visible(False)

save_images(test_img, 'test_img')
save_images(test_gt, 'test_gt')
save_images(train_img, 'train_img')
save_images(train_gt, 'train_gt')
