import tensorflow as tf
import numpy as np

from tensorflow.contrib.slim.nets import resnet_v2
from tensorflow.contrib import layers as layers_lib
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.slim.python.slim.nets import resnet_utils


_BATCH_NORM_DECAY = 0.9
_WEIGHT_DECAY = 5e-4


class Resnet50():
    def __init__(self, path, opt):
        self.path = path
        self.opt = opt

    def l1(self,img1, img2):
        return tf.reduce_mean(tf.abs(tf.to_float(img1) - tf.to_float(img2)))

    def l1_mask(self,img1, img2, mask1, mask2):
        h,w = img1.get_shape().as_list()[1:3]
        mask1 = tf.image.resize_nearest_neighbor(mask1, [h,w])
        mask2 = tf.image.resize_nearest_neighbor(mask2, [h,w])
        return tf.reduce_mean(tf.abs(tf.to_float(img1 * mask1) - tf.to_float(img2 * mask2)))

    #def calc_loss_mask(self, inputs1, inputs2):
    #    inputs1 = tf.image.resize_bilinear(inputs1, [512, 512])
    #    inputs2 = tf.image.resize_bilinear(inputs2, [512, 512])

    #    preds, results, end_points = self.built_network(inputs1, inputs2)

    #    tf.train.init_from_checkpoint(self.path, {"Seg/": "Seg/"})
    #    print("Seg restore success")

    #    batch_size = inputs1.get_shape().as_list()[0]
    #    if batch_size == 1:
    #        print("batch_size == 1")
    #        preds1, results1 = tf.expand_dims(preds[0],axis=0), tf.expand_dims(results[0], axis=0)
    #        preds2, results2 = tf.expand_dims(preds[1],axis=0), tf.expand_dims(results[1], axis=0)

    #    else:
    #        print("batch_size != 1")
    #        preds1, results1 = preds[:batch_size], results[:batch_size]
    #        preds2, results2 = preds[batch_size:], results[batch_size:]

    #    results1_onehot = tf.one_hot(results1, 7, axis=-1)
    #    results2_onehot = tf.one_hot(results2, 7, axis=-1)

    #    mask1 = tf.expand_dims(results1_onehot[:,:,:,1] + results1_onehot[:,:,:,2] + results1_onehot[:,:,:,3] + results1_onehot[:,:,:,4] + results1_onehot[:,:,:,5], axis=-1)
    #    mask2 = tf.expand_dims(results2_onehot[:,:,:,1] + results2_onehot[:,:,:,2] + results2_onehot[:,:,:,3] + results2_onehot[:,:,:,4] + results2_onehot[:,:,:,5], axis=-1)


    #    p0 = self.l1_mask(preds1, preds2, mask1, mask2)
    #    p1 = self.l1_mask(end_points["Seg/resnet_v2_50/block1"][:batch_size],
    #                 end_points["Seg/resnet_v2_50/block1"][batch_size:], mask1, mask2)
    #    p2 = self.l1_mask(end_points["Seg/resnet_v2_50/block2"][:batch_size],
    #                 end_points["Seg/resnet_v2_50/block2"][batch_size:], mask1, mask2)
    #    p3 = self.l1_mask(end_points["Seg/resnet_v2_50/block3"][:batch_size],
    #                 end_points["Seg/resnet_v2_50/block3"][batch_size:], mask1, mask2)
    #    p4 = self.l1_mask(end_points["Seg/resnet_v2_50/block4"][:batch_size],
    #                 end_points["Seg/resnet_v2_50/block4"][batch_size:], mask1, mask2)

    #    p5 = self.l1_mask(end_points["Seg/resnet_v2_50/conv1"][:batch_size],
    #                 end_points["Seg/resnet_v2_50/conv1"][batch_size:], mask1, mask2)

    #    p_loss = 10 * p5  # + p1 + p2 + p3 + p4

    #    tf.summary.image("train/seg_pred", self._decode_label(results), collections=[self.opt.train_collection])
    #    tf.summary.image("train/seg_mask1", mask1, collections=[self.opt.train_collection])
    #    tf.summary.image("train/seg_mask2", mask2, collections=[self.opt.train_collection])

    #    return p_loss


    def calc_loss(self, inputs1, inputs2):
        # inputs1 = tf.image.resize_bilinear(inputs1, [512, 512])
        # inputs2 = tf.image.resize_bilinear(inputs2, [512, 512])

        end_points = self.built_network(inputs1, inputs2)

        tf.train.init_from_checkpoint(self.path,{"resnet_v2_50/": "Seg/resnet_v2_50/"})
        print("resnet restore success")

        batch_size = inputs1.get_shape().as_list()[0]
        #preds1, results1  = preds[:batch_size], results[:batch_size]
        #preds2, results2  = preds[batch_size:], results[batch_size:]
        #p0 = self.l1(preds1, preds2)

        p1 = self.l1(end_points["Seg/resnet_v2_50/block1"][:batch_size], end_points["Seg/resnet_v2_50/block1"][batch_size:])
        p2 = self.l1(end_points["Seg/resnet_v2_50/block2"][:batch_size], end_points["Seg/resnet_v2_50/block2"][batch_size:])
        p3 = self.l1(end_points["Seg/resnet_v2_50/block3"][:batch_size], end_points["Seg/resnet_v2_50/block3"][batch_size:])
        p4 = self.l1(end_points["Seg/resnet_v2_50/block4"][:batch_size], end_points["Seg/resnet_v2_50/block4"][batch_size:])

        p5 = self.l1(end_points["Seg/resnet_v2_50/conv1"][:batch_size], end_points["Seg/resnet_v2_50/conv1"][batch_size:])
        
        p_loss = p2 #+ p5   #+ p1 + p2 + p3 + p4

        # tf.summary.image("train/seg_pred", self._decode_label(results), collections=[self.opt.train_collection])

        return p_loss

    def built_network(self, inputs1, inputs2, is_training=False):
        inputs = tf.concat([inputs1, inputs2], axis=0)
        with tf.variable_scope("Seg"):
            with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=_BATCH_NORM_DECAY)):
                logits, end_points = resnet_v2.resnet_v2_50(inputs, is_training=is_training)

        return end_points

