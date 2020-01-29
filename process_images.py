import utils
import pathlib
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
matplotlib.rcParams['interactive'] == True

img_set = pathlib.Path("images_2000/")
# Convert all paths into a string
all_image_paths = [str(img_path) for img_path in list(img_set.glob("*.jpg"))]
#print(all_image_paths)

tensors=[]
for i in all_image_paths:
  tensors.append(utils.load_and_preprocess_images(i))

#fig = plt.figure(1, figsize=(10, 10))
#plt.imshow(tensors[3])
#plt.savefig('figure_3_paysage.png')
# create tensor vectors with skipthought vectors as input
print('Save images vector : loading ....')
np.save('imgvectors_2000.npy', tensors)
print('Save images vector : DONE !')