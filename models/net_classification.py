from __future__ import division
import os
import math
import tensorflow as tf
import numpy as np
from .stn import spatial_transformer_network as transformer
from .Vgg19 import  VGG19

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import vgg
from tensorflow.contrib.slim.nets import resnet_v2
    
class BaseModel():
    def __init__(self, opt):
        self.opt = opt
    def decay_weight(self, global_step, w_init, w_end, start_step, decay_steps, name='decay_param'):
        """ Adam optimizer with learning rate 0.0002 for the first 100k steps (~100 epochs)
            and a linearly decaying rate that goes to zero over the next 100k steps
        """
        starter_w = w_init
        end_w = w_end
        start_decay_step = start_step
        decay_steps = decay_steps
        w_cur = (
                 tf.where(
                    tf.greater_equal(global_step, start_decay_step), tf.train.polynomial_decay(starter_w, global_step - start_decay_step,
                    decay_steps, end_w, power=1.0), starter_w)
        )

        tf.summary.scalar('param_decay/{}'.format(name), w_cur, collections=[self.opt.train_collection])

        return w_cur

    def calc_loss(self, predictions, labels):
        classification_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predictions, labels=labels))
        regularization_loss = tf.add_n(tf.losses.get_regularization_losses())

        total_loss = classification_loss + regularization_loss
        return total_loss

    def update_learning_rate(self, lr, global_step):
        start_step = int(self.opt.start_epoch_decay * self.train_dataset_size / self.opt.batch_size)
        decay_steps = int(self.opt.end_epoch_decay  * self.train_dataset_size / self.opt.batch_size)
        cur_lr = self.decay_weight(global_step, lr, 0, start_step = start_step, decay_steps = decay_steps, name="lr")
        
        # tf.summary.scalar("lr", lr, collections=[self.opt.train_collection])
        return cur_lr

    def make_optimizer(self, loss, global_step, t_vars):
        cur_lr = self.update_learning_rate(self.opt.lr, global_step)
        cls_optimizer = tf.train.AdamOptimizer(cur_lr, beta1=self.opt.beta1, name="adam").minimize(loss, var_list=t_vars)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            with tf.control_dependencies([cls_optimizer]):
                return tf.no_op(name='optimizers')


class VGG16(BaseModel):
    def __init__(self, opt):
        super(VGG16, self).__init__(opt)
        #pass

    def norm(self, inputs, is_training, norm="instance"):
        with  tf.variable_scope(name):
            if norm == "instance":
                print("instance norm")
                out = tf.contrib.layers.instance_norm(inputs)
            elif norm == "batch_norm":
                print("batch norm")
                out = tf.layers.batch_normalization(inputs, momentum=0.9, scale=True, training=is_training)
            else:
                print("no normaliziation")
                out = inputs
                raise NotImplementedError
            return out


    def built_network(self, inputs, is_training, dropout_rate):

        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            net, end_points = resnet_v2.resnet_v2_50(inputs, self.opt.num_classes, is_training=is_training)
        net = tf.squeeze(net, axis=[1,2])
        return net

        # with slim.arg_scope(vgg.vgg_arg_scope()):
        #     outputs, end_points = vgg.vgg_16(inputs, self.opt.num_classes, is_training=is_training, dropout_keep_prob=dropout_rate)
        #
        # return outputs
        # with slim.arg_scope([slim.conv2d, slim.fully_connected],
        #                     activation_fn=tf.nn.relu,
        #                     weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
        #                     weights_regularizer=slim.l2_regularizer(0.0005)):
        #     net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        #     net = slim.max_pool2d(net, [2, 2], scope='pool1')
        #     net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        #     net = slim.max_pool2d(net, [2, 2], scope='pool2')
        #     net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        #     net = slim.max_pool2d(net, [2, 2], scope='pool3')
        #     net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        #     net = slim.max_pool2d(net, [2, 2], scope='pool4')
        #     net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        #     net = slim.max_pool2d(net, [2, 2], scope='pool5')
        #     net = slim.fully_connected(net, 4096, scope='fc6')
        #     net = slim.dropout(net, 0.5, scope='dropout6')
        #     net = slim.fully_connected(net, 4096, scope='fc7')
        #     net = slim.dropout(net, 0.5, scope='dropout7')
        #     net = slim.fully_connected(net, opt.class_num, activation_fn=None, scope='fc8')
        #
        #     return net

    def train(self, global_step, images, labels, is_training, dropout_rate, dataset_size):
        self.global_step = global_step
        self.train_dataset_size = dataset_size
        with tf.variable_scope("vgg16") as scope:
            preds = self.built_network(images,is_training, dropout_rate)

        t_vars = tf.trainable_variables()
        self.loss = self.calc_loss(preds, labels)
        self.train_op = self.make_optimizer(self.loss, global_step, t_vars)

        correct_prediction = tf.equal(tf.argmax(preds, 1), labels)
        self.acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.summary.scalar("train/loss", self.loss, collections=[self.opt.train_collection])
        tf.summary.scalar("train/acc", self.acc, collections=[self.opt.train_collection])
        tf.summary.image("train/_inputs", images, collections=[self.opt.train_collection])

        # validation
        self.loss_val = self.loss
        self.acc_val = self.acc

        tf.summary.scalar("val/loss", self.loss, collections=[self.opt.val_collection])
        tf.summary.scalar("val/acc", self.acc, collections=[self.opt.val_collection])
        tf.summary.image("val/_inputs", images, collections=[self.opt.val_collection])



    def evalute(self, images, labels, reuse=True):
        with tf.variable_scope("vgg16", reuse=reuse) as scope:
            preds = self.built_network(images)

        # t_vars = tf.trainable_variables()
        self.loss_val = self.calc_loss(preds, labels)

        correct_prediction = tf.equal(tf.argmax(preds, 1), labels)
        self.acc_val = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.summary.scalar("val/loss", self.loss_val, collections=[self.opt.val_collection])
        tf.summary.scalar("train/acc", self.acc_val, collections=[self.opt.train_collection])
        tf.summary.image("val/_inputs", images, collections=[self.opt.val_collection])



    def test(self, images):
        with tf.variable_scope("vgg16") as scope:
            preds = self.built_network(images)
            self.preds = tf.argmax(preds, axis=1)


