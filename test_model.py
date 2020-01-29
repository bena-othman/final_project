from tensorlayer import models
import numpy as np
import skipthoughts
import utils
import tensorflow as tf
import tensorlayer as tl

M = models.Model.load('checkpoint/G.h5')
M.eval()

z = np.random.normal(loc=0.0, scale=1.0, size=[1, z_dim]).astype(np.float32)
test_descr = 'forest near glacier mountain during day'

model = skipthoughts.load_model()
print('Creation of skipthought vectors : loading ....')
caption_vectors = skipthoughts.encode(model, test_descr)
#tensor_captions = tf.convert_to_tensor(caption_vectors)
reduced_text_embedding = utils.lrelu(utils.linear(caption_vectors, 256))

img = M([z, test_descr])

img = img.numpy().squeeze().astype(np.uint8)
tl.visualize.save_image(img, 'model_test/img_test.jpg')