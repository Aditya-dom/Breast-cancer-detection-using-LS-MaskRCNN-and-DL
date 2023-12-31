

import numpy as np
from matplotlib import pyplot as plt


directory = '/home/kazzastic/Documents/Breast-Cancer-Lump-Segmentation-MaskRCNN/data/patches/calcification/no_tumor'
img_array = np.load(directory+'/Calc-Test_P_00038_LEFT_CC_ROTATE_64.npy')
print(img_array.shape)


plt.imshow(img_array, cmap='gray')
plt.show()