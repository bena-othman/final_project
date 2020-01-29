from tensorlayer.layers import *
from tensorlayer.activation import *
import tensorflow as tf
import tensorlayer as tl

batch_size = 5
t_dim = 254
#z_dim = 512         # Noise dimension
image_size = 64     # 64 x 64
c_dim = 3           # for rgb

def generator_txt2img_resnet(input_z_txt, is_train=True, reuse=False, batch_size=batch_size):
    """ z + (txt) --> 64x64 """
    # https://github.com/hanzhanggit/StackGAN/blob/master/stageI/model.py
    s = image_size # output image size [64]
    s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)
    gf_dim = 128

    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)

    set_name_reuse(reuse)
    net_in = Input(input_z_txt)

    # if t_txt is not None:
    # net_txt = Input(t_txt)
    # net_txt = Dense(n_units=t_dim, act=lambda x: lrelu(x, 0.2), W_init=w_init, name='g_reduce_text/dense')(net_txt)
    # net_in = Concat([net_in, net_txt], concat_dim=1, name='g_concat_z_txt')

    net_h0 = Dense(gf_dim * 8 * s16 * s16, act=tf.identity, W_init=w_init, b_init=None, name='g_h0/dense')(net_in)
    net_h0 = BatchNorm(is_train=is_train, gamma_init=gamma_init, name='g_h0/batch_norm')(net_h0)
    net_h0 = Reshape([-1, s16, s16, gf_dim * 8], name='g_h0/reshape')(net_h0)

    net = Conv2d(gf_dim * 2, (1, 1), (1, 1), padding='VALID', act=None, W_init=w_init, b_init=None,
                 name='g_h1_res/conv2d')(net_h0)
    net = BatchNorm(act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='g_h1_res/batch_norm')(net)
    net = Conv2d(gf_dim * 2, (3, 3), (1, 1), padding='SAME', act=None, W_init=w_init, b_init=None,
                 name='g_h1_res/conv2d2')(net)
    net = BatchNorm(act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='g_h1_res/batch_norm2')(net)
    net = Conv2d(gf_dim * 8, (3, 3), (1, 1), padding='SAME', act=None, W_init=w_init, b_init=None,
                 name='g_h1_res/conv2d3')(net)
    net = BatchNorm(is_train=is_train, gamma_init=gamma_init, act=lrelu, name='g_h1_res/batch_norm3')(net)
    net_h1 = Elementwise(combine_fn=tf.add, name='g_h1_res/add')([net_h0, net])
    # net_h1 = Layer(act=lrelu)(net_h1)
    # net_h1.outputs = tf.nn.relu(net_h1.outputs)

    net_h2 = DeConv2d(gf_dim * 4, (4, 4), strides=(2, 2), padding='SAME', act=None,W_init=w_init, name='g_h2/decon2d')(net_h1)
    net_h2 = BatchNorm(is_train=is_train, gamma_init=gamma_init, name='g_h2/batch_norm')(net_h2)

    net = Conv2d(gf_dim, (1, 1), (1, 1), padding='VALID', act=None, W_init=w_init, b_init=None, name='g_h3_res/conv2d')(
        net_h2)
    net = BatchNorm(act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='g_h3_res/batch_norm')(net)
    net = Conv2d(gf_dim, (3, 3), (1, 1), padding='SAME', act=None, W_init=w_init, b_init=None, name='g_h3_res/conv2d2')(
        net)
    net = BatchNorm(act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='g_h3_res/batch_norm2')(net)
    net = Conv2d(gf_dim * 4, (3, 3), (1, 1), padding='SAME', act=None, W_init=w_init, b_init=None,
                 name='g_h3_res/conv2d3')(net)
    net = BatchNorm(is_train=is_train, gamma_init=gamma_init, name='g_h3_res/batch_norm3')(net)
    net_h3 = Elementwise(act=lrelu, combine_fn=tf.add, name='g_h3/add')([net_h2, net])
    # net_h3.outputs = tf.nn.relu(net_h3.outputs)

    net_h4 = DeConv2d(gf_dim * 2, (4, 4), strides=(2, 2), padding='SAME', act=None,
                      W_init=w_init, name='g_h4/decon2d')(net_h3)
    net_h4 = BatchNorm(act=lrelu, is_train=is_train, gamma_init=gamma_init, name='g_h4/batch_norm')(net_h4)

    net_h5 = DeConv2d(gf_dim, (4, 4), strides=(2, 2), padding='SAME', act=None, W_init=w_init,
                      name='g_h5/decon2d')(net_h4)

    net_h5 = BatchNorm(act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='g_h5/batch_norm')(net_h5)

    net_ho = DeConv2d(c_dim, (4, 4), strides=(2, 2), padding='SAME', act=htanh, W_init=w_init,
                      name='g_ho/decon2d')(net_h5)

    return tl.models.Model(inputs=net_in, outputs=net_ho)

def discriminator_txt2img_resnet(input_images, t_txt, is_train=True, reuse=False):
    """ 64x64 + (txt) --> real/fake """
    # https://github.com/hanzhanggit/StackGAN/blob/master/stageI/model.py
    # Discriminator with ResNet : line 197 https://github.com/reedscot/icml2016/blob/master/main_cls.lua
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init=tf.random_normal_initializer(1., 0.02)
    df_dim = 64  # 64 for flower, 196 for MSCOCO
    s = 64 # output image size [64]
    s2, s4, s8, s16 = int(s / 2), int(s / 4), int(s / 8), int(s / 16)

    tl.layers.set_name_reuse(reuse)
    net_in = Input(input_images)
    net_h0 = Conv2d(df_dim, (4, 4), (2, 2), act=lambda x: tl.act.lrelu(x, 0.2), padding='SAME', W_init=w_init,
                    name='d_h0/conv2d')(net_in)

    net_h1 = Conv2d(df_dim * 2, (4, 4), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=None,
                    name='d_h1/conv2d')(net_h0)
    net_h1 = BatchNorm(act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init,
                       name='d_h1/batchnorm')(net_h1)
    net_h2 = Conv2d(df_dim * 4, (4, 4), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=None,
                    name='d_h2/conv2d')(net_h1)
    net_h2 = BatchNorm(act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init,
                       name='d_h2/batchnorm')(net_h2)
    net_h3 = Conv2d(df_dim * 8, (4, 4), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=None,
                    name='d_h3/conv2d')(net_h2)
    net_h3 = BatchNorm(is_train=is_train, gamma_init=gamma_init, name='d_h3/batchnorm')(net_h3)

    net = Conv2d(df_dim * 2, (1, 1), (1, 1), act=None, padding='VALID', W_init=w_init, b_init=None,
                 name='d_h4_res/conv2d')(net_h3)
    net = BatchNorm(act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init,
                    name='d_h4_res/batchnorm')(net)
    net = Conv2d(df_dim * 2, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=None,
                 name='d_h4_res/conv2d2')(net)
    net = BatchNorm(act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init,
                    name='d_h4_res/batchnorm2')(net)
    net = Conv2d(df_dim * 8, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=None,
                 name='d_h4_res/conv2d3')(net)
    net = BatchNorm(is_train=is_train, gamma_init=gamma_init, name='d_h4_res/batchnorm3')(net)
    net_h4 = Elementwise(act=lrelu, combine_fn=tf.add, name='d_h4/add')([net_h3, net])
    # net_h4.outputs = tl.act.lrelu(net_h4.outputs, 0.2)

    if t_txt is not None:
        net_in2 = Input(t_txt)
        #net_txt = Dense(n_units=t_dim, act=lambda x: tl.act.lrelu(x, 0.2), W_init=w_init, name='d_reduce_txt/dense')(net_txt)
        net_txt = ExpandDims(1, name='d_txt/expanddim1')(net_in2)
        net_txt = ExpandDims(1, name='d_txt/expanddim2')(net_txt)
        net_txt = Tile([1, 4, 4, 1], name='d_txt/tile')(net_txt)
        net_h4_concat = Concat(concat_dim=3, name='d_h3_concat')([net_h4, net_txt])
        # 243 (ndf*8 + 128 or 256) x 4 x 4
        net_h4 = Conv2d(df_dim * 8, (1, 1), (1, 1), padding='VALID', W_init=w_init, b_init=None, name='d_h3/conv2d_2')(net_h4_concat)
        net_h4 = BatchNorm(act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init,
                           name='d_h3/batch_norm_2')(net_h4)

        net_ho = Conv2d(1, (s16, s16), (s16, s16), act=tf.nn.sigmoid, padding='VALID', W_init=w_init, name='d_ho/conv2d')(net_h4)
        # 1 x 1 x 1
        net_ho = Flatten()(net_h0)


# logits = net_ho.outputs
# net_ho.outputs = tf.nn.sigmoid(net_ho.outputs)
    return tl.models.Model(inputs=[net_in,net_in2], outputs=net_ho)