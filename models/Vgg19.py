import tensorflow as tf
import numpy as np
import scipy,scipy.io
class VGG19(object):
    def __init__(self, vgg_weights_path, im1, im2, name='vgg'):
        self.mean = np.array([123.6800, 116.7790, 103.9390]
                             ).reshape((1, 1, 1, 3))
        self.vgg_weights = vgg_weights_path

        self.vgg_real = self.build_vgg19(im1, name)
        self.vgg_fake = self.build_vgg19(im2, name,reuse=True)

    def l1(self,img1, img2):
        return tf.reduce_mean(tf.abs(img1 - img2))
    def build_net(self, ntype, nin, nwb=None, name=None):
        if ntype == 'conv':
            return tf.nn.relu(tf.nn.conv2d(nin, nwb[0], strides=[1, 1, 1, 1], padding='SAME', name=name) + nwb[1])
        elif ntype == 'pool':
            return tf.nn.avg_pool(nin, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def get_weight_bias(self, vgg_layers, i):
        weights = vgg_layers[i][0][0][2][0][0]
        weights = tf.constant(weights)
        bias = vgg_layers[i][0][0][2][0][1]
        bias = tf.constant(np.reshape(bias, (bias.size)))
        return weights, bias

    def build_vgg19(self, input, name, reuse=False):
        # if reuse:
        #     tf.get_variable_scope().reuse_variables()
        with tf.variable_scope(name, reuse=reuse):
            net = {}
            vgg_rawnet = scipy.io.loadmat(self.vgg_weights)
            vgg_layers = vgg_rawnet['layers'][0]
            net['input'] = input - self.mean
            net['conv1_1'] = self.build_net('conv', net['input'], self.get_weight_bias(vgg_layers, 0),
                                            name='vgg_conv1_1')
            net['conv1_2'] = self.build_net('conv', net['conv1_1'], self.get_weight_bias(vgg_layers, 2),
                                            name='vgg_conv1_2')
            net['pool1'] = self.build_net('pool', net['conv1_2'])
            net['conv2_1'] = self.build_net('conv', net['pool1'], self.get_weight_bias(vgg_layers, 5),
                                            name='vgg_conv2_1')
            net['conv2_2'] = self.build_net('conv', net['conv2_1'], self.get_weight_bias(vgg_layers, 7),
                                            name='vgg_conv2_2')
            net['pool2'] = self.build_net('pool', net['conv2_2'])
            net['conv3_1'] = self.build_net('conv', net['pool2'], self.get_weight_bias(vgg_layers, 10),
                                            name='vgg_conv3_1')
            net['conv3_2'] = self.build_net('conv', net['conv3_1'], self.get_weight_bias(vgg_layers, 12),
                                            name='vgg_conv3_2')
            net['conv3_3'] = self.build_net('conv', net['conv3_2'], self.get_weight_bias(vgg_layers, 14),
                                            name='vgg_conv3_3')
            net['conv3_4'] = self.build_net('conv', net['conv3_3'], self.get_weight_bias(vgg_layers, 16),
                                            name='vgg_conv3_4')
            net['pool3'] = self.build_net('pool', net['conv3_4'])
            net['conv4_1'] = self.build_net('conv', net['pool3'], self.get_weight_bias(vgg_layers, 19),
                                            name='vgg_conv4_1')
            net['conv4_2'] = self.build_net('conv', net['conv4_1'], self.get_weight_bias(vgg_layers, 21),
                                            name='vgg_conv4_2')
            net['conv4_3'] = self.build_net('conv', net['conv4_2'], self.get_weight_bias(vgg_layers, 23),
                                            name='vgg_conv4_3')
            net['conv4_4'] = self.build_net('conv', net['conv4_3'], self.get_weight_bias(vgg_layers, 25),
                                            name='vgg_conv4_4')
            net['pool4'] = self.build_net('pool', net['conv4_4'])
            net['conv5_1'] = self.build_net('conv', net['pool4'], self.get_weight_bias(vgg_layers, 28),
                                            name='vgg_conv5_1')
            net['conv5_2'] = self.build_net('conv', net['conv5_1'], self.get_weight_bias(vgg_layers, 30),
                                            name='vgg_conv5_2')
            net['conv5_3'] = self.build_net('conv', net['conv5_2'], self.get_weight_bias(vgg_layers, 32),
                                            name='vgg_conv5_3')
            net['conv5_4'] = self.build_net('conv', net['conv5_3'], self.get_weight_bias(vgg_layers, 34),
                                            name='vgg_conv5_4')
            net['pool5'] = self.build_net('pool', net['conv5_4'])
        return net

    # def perceptual_loss(self, im1, im2, reuse = False):
    #     self.vgg_real = self.build_vgg19(im1,reuse = reuse)
    #     self.vgg_fake = self.build_vgg19(im2, reuse = True)
    #     _, sp, _, _ = im1.get_shape().as_list()
    #     #p1 = self.l1(self.vgg_real['conv1_2'], self.vgg_fake['conv1_2']) 
    #     #p2 = self.l1(self.vgg_real['conv2_2'], self.vgg_fake['conv2_2']) 
    #     #p3 = self.l1(self.vgg_real['conv3_2'], self.vgg_fake['conv3_2']) 
    #     p4 = self.l1(self.vgg_real['conv4_2'], self.vgg_fake['conv4_2'])
    #     #p5 = self.l1(self.vgg_real['conv5_2'], self.vgg_fake['conv5_2'])
    #     # print(p1,p2,p3,p4,p5)
    #     #ploss = [p1,p2,p3,p4,p5]
    #     ploss = p4
    #     return ploss

    def perceptual_loss(self):

        # p0 = self.l1(self.vgg_real['input'],   self.vgg_fake['input']) / 2.6
        p1 = self.l1(self.vgg_real['conv1_2'], self.vgg_fake['conv1_2']) / 3.2
        p2 = self.l1(self.vgg_real['conv2_2'], self.vgg_fake['conv2_2']) / 4.8
        p3 = self.l1(self.vgg_real['conv3_2'], self.vgg_fake['conv3_2']) / 3.7
        p4 = self.l1(self.vgg_real['conv4_2'], self.vgg_fake['conv4_2']) / 5.6
        p5 = self.l1(self.vgg_real['conv5_2'], self.vgg_fake['conv5_2']) * 10 / 1.5
        p_loss = p3#p2 + p3 + p4 + p5
        return p_loss

    def gram_matrix(self, layer):
        bs, height, width, filters = map(lambda i:i.value,layer.get_shape())  
        size = height * width * filters
        feats = tf.reshape(layer, (bs, height * width, filters))
        feats_T = tf.transpose(feats, perm=[0,2,1])
        grams = tf.matmul(feats_T, feats) / size

        return grams


    def style_loss(self):

        s1 = self.l1(self.gram_matrix(self.vgg_real['conv1_1']), self.gram_matrix(self.vgg_fake['conv1_1'])) 
        s2 = self.l1(self.gram_matrix(self.vgg_real['conv2_1']), self.gram_matrix(self.vgg_fake['conv2_1']))
        s3 = self.l1(self.gram_matrix(self.vgg_real['conv3_1']), self.gram_matrix(self.vgg_fake['conv3_1'])) 
        s4 = self.l1(self.gram_matrix(self.vgg_real['conv4_1']), self.gram_matrix(self.vgg_fake['conv4_1']))
        s5 = self.l1(self.gram_matrix(self.vgg_real['conv5_1']), self.gram_matrix(self.vgg_fake['conv5_1'])) 
        
        s_loss = s1 + s2 + s3 + s4 + s5
        
        return s_loss


    def similarity(self, feat1, feat2):
        batch_size, sp, _, channel = feat1.get_shape().as_list()
        feat1_tmp = tf.reshape(feat1, [batch_size, sp*sp, 1, channel])
        feat2_tmp = tf.reshape(feat2, [batch_size, sp*sp, 1, channel])
        print(feat1_tmp)
        feat1_tmp = tf.tile(feat1_tmp, [1, 1, sp*sp, 1])
        feat2_tmp = tf.tile(feat2_tmp, [1, 1, sp*sp, 1])
        print(feat1_tmp)
        feat1_tmp = tf.abs(
            feat1_tmp - tf.transpose(feat1_tmp, perm=[0, 2, 1, 3]))
        feat2_tmp = tf.abs(
            feat2_tmp - tf.transpose(feat2_tmp, perm=[0, 2, 1, 3]))
        return tf.reduce_mean(tf.abs(feat1_tmp - feat2_tmp))

    def similarity_loss(self, im1, im2):
        s4 = self.similarity(
            self.vgg_real['conv4_2'], self.vgg_fake['conv4_2']) / 4.0
        s5 = self.similarity(
            self.vgg_real['conv5_2'], self.vgg_fake['conv5_2']) / 2.0
        s6 = self.similarity(self.vgg_real['pool5'], self.vgg_fake['pool5'])
        sloss = s4 + s5 + s6
        return sloss


