import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

image_size = 128
t_dim = 256  # text feature dimension
batch_size = 10  # "The number of batch images [64]")
c_dim = 3  # for rgb

s2, s4, s8, s16, s32 = int(image_size / 2), int(image_size / 4), int(image_size / 8), int(image_size / 16), int(image_size / 32)


def get_generator(input_z, t_txt=None, is_train=True, gf_dim=128):  # Dimension of gen filters in first conv layer. [64]
    b_init = None
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)

    net_in = Input(input_z)

    net_txt = Input(t_txt)
    net_txt = Dense(n_units=t_dim, act=lambda x: tl.act.lrelu(x, 0.2), W_init=w_init)(net_txt)
    net_in = Concat(concat_dim=1)([net_in, net_txt])

    net_h0 = Dense(gf_dim * 16 * s32 * s32, act=tf.identity, W_init=w_init, b_init=b_init)(net_in)
    net_h0 = Reshape([-1, s32, s32, gf_dim * 16])(net_h0)
    net_h0 = BatchNorm(act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init)(net_h0)

    net_h1 = DeConv2d(gf_dim * 8, (5, 5), strides=(2, 2), padding='SAME', act=None, W_init=w_init, b_init=b_init)(net_h0)
    net_h1 = BatchNorm(act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init)(net_h1)

    net_h2 = DeConv2d(gf_dim * 4, (5, 5), strides=(2, 2),padding='SAME', act=None, W_init=w_init, b_init=b_init)(net_h1)
    net_h2 = BatchNorm(act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init)(net_h2)

    net_h3 = DeConv2d(gf_dim * 2, (5, 5), strides=(2, 2), padding='SAME', act=None, W_init=w_init, b_init=b_init)(net_h2)
    net_h3 = BatchNorm(act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init)(net_h3)

    net_h4 = DeConv2d(gf_dim, (5, 5), strides=(2, 2), padding='SAME', act=None, W_init=w_init, b_init=b_init)(net_h3)
    net_h4 = BatchNorm(act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init)(net_h4)

    net_h5 = DeConv2d(c_dim, (5, 5), strides=(2, 2), padding='SAME', act=tf.nn.tanh, W_init=w_init)(net_h4)

    return tl.models.Model(inputs=net_in, outputs=net_h5)


def get_discriminator(input_images, input_rnn_embed, is_train=True):  # Dimension of discrim filters in first conv layer. [64]
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None  # tf.constant_initializer(value=0.0)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    df_dim = 128

    net_in = Input(input_images)
    nn = Conv2d(df_dim, (5, 5), (2, 2), act=lambda x: tl.act.lrelu(x, 0.2), padding='SAME', W_init=w_init)(net_in)

    nn = Conv2d(df_dim * 2, (5, 5), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init)(nn)
    nn = BatchNorm2d(act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init)(nn)

    nn = Conv2d(df_dim * 4, (5, 5), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init)(nn)
    nn = BatchNorm2d(act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init)(nn)

    nn = Conv2d(df_dim * 8, (5, 5), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init)(nn)
    nn = BatchNorm2d(act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init)(nn)

    # adding a layer for 128*128 input images
    nn = Conv2d(df_dim * 16, (5, 5), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init)(nn)
    nn = BatchNorm2d(act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init)(nn)

    net_in2 = Input(input_rnn_embed)
    net_txt = Dense(n_units=t_dim, act=lambda x: tl.act.lrelu(x, 0.2), W_init=w_init, b_init=None)(net_in2)
    net_txt = ExpandDims(1)(net_txt)
    net_txt = ExpandDims(1)(net_txt)
    net_txt = Tile(multiples=[1, 4, 4, 1])(net_txt)

    net_concat = Concat(3)([nn, net_txt])
    nn = Conv2d(df_dim * 16, (1, 1), (1, 1), padding='SAME', W_init=w_init, b_init=b_init)(net_concat)
    nn = BatchNorm2d(act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init)(nn)
    nn = Flatten()(nn)
    nn = Dense(n_units=1, act=tf.nn.sigmoid, W_init=w_init)(nn)

    return tl.models.Model(inputs=[net_in, net_in2], outputs=nn, name='discriminator')
