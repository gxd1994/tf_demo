import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import architectures.alexnet
import architectures.resnet
import architectures.vgg
import architectures.googlenet
import architectures.nin
import architectures.densenet
import architectures.common as common

color_map = {0: [255, 255, 255], 1: [0, 255, 0], 2: [0, 0, 255], 3: [255, 0, 0], 4: [0, 128, 128],
             5: [128, 0, 128], 6: [128, 128, 128], 7: [128, 255, 0], 8: [128, 0, 255], 9: [0, 128, 255],
             10: [0, 255, 128]}

class BaseModel():
    def __init__(self, opt):
        self.opt = opt
        # pass

    def _prepare_label(self,label_batch,new_size):
        # label_map = tf.image.resize_nearest_neighbor(label_batch, new_size,name='reisze_label_op')
        label_map = tf.cast(label_batch, tf.int32)
        label_map = tf.squeeze(label_map, axis=-1)
        label_map = tf.one_hot(label_map, depth=self.opt.class_num)
        label_batch_final = tf.reshape(label_map,[-1, self.opt.class_num])

        return label_map, label_batch_final

    def calc_loss(self, preds, labels):

        labels_map, labels_final = self._prepare_label(labels, tf.stack(preds.get_shape()[1:3]))

        preds_final = tf.reshape(preds, [-1, self.opt.class_num])

        print('final_shape labels preds',labels_final, preds_final)

        cross_entroy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels_final,logits=preds_final,name='cross_entroy'))

        return cross_entroy

    def _calculate_miou(self,logits, label_batch):
        with tf.variable_scope('MIOU_CAL'):
            confusion_matrix = tf.confusion_matrix(labels=tf.reshape(label_batch,[-1]),predictions=tf.reshape(logits,[-1]),num_classes=self.opt.class_num,dtype=tf.float32)
            def cal_miou(matrix):
                sum_col = np.zeros(shape = [self.opt.class_num],dtype=np.float32)
                sum_row = np.zeros(shape = [self.opt.class_num],dtype=np.float32)
                miou = np.zeros(shape = [],dtype=np.float32)
                for i in range(self.opt.class_num):
                    for j in range(self.opt.class_num):
                        sum_row[i] += matrix[i][j]
                        sum_col[j] += matrix[i][j]
                for i in range(self.opt.class_num):
                    if sum_col[i]+sum_row[i]-matrix[i][i] != 0:
                        miou += matrix[i][i]/(sum_col[i]+sum_row[i]-matrix[i][i])
                miou = (miou/self.opt.class_num).astype(np.float32)
                return miou

            miou = tf.py_func(cal_miou, [confusion_matrix], tf.float32)
        return miou  #,miou1

    def _decode_label(self, label_batch):
        with tf.variable_scope('decode_label'):
            def decode_label(label_batch):
                label_batch_rgb = np.zeros(shape=(label_batch.shape[0],label_batch.shape[1],label_batch.shape[2], 3), dtype=np.uint8)
                for i, label in enumerate(label_batch):
                    img = label
                    h,w = img.shape[0:2]
                    label_mask = np.zeros(shape=(h,w,3), dtype=np.uint8)
                    r = label_mask[:, :, 0]
                    g = label_mask[:, :, 1]
                    b = label_mask[:, :, 2]
                    for ll in range(11):
                        index = img == ll
                        r[index] = color_map[ll][0]
                        g[index] = color_map[ll][1]
                        b[index] = color_map[ll][2]
                    label_batch_rgb[i] = np.concatenate((b[:, :, np.newaxis], r[:, :, np.newaxis], g[:, :, np.newaxis]), axis=2)
                return  label_batch_rgb

            label_batch_rgb = tf.py_func(decode_label, [label_batch], tf.uint8)
        return label_batch_rgb

    def calc_acc(self, preds, labels):

        # label_final = tf.image.resize_nearest_neighbor(labels, tf.stack(preds.get_shape()[1:3]))
        label_final = tf.cast(tf.squeeze(labels, axis=3), tf.int32)
        preds_final = tf.cast(tf.argmax(preds, axis=3), tf.int32)

        print("label_final, preds_final", label_final, preds_final)
        miou = self._calculate_miou(preds_final, label_final)

        # tf.summary.scalar('miou_value_train',miou,collections=[SEG_COLLECTION])
        # tf.summary.scalar('miou_value_val', miou, collections=[SEG_VAL_COLLECTION])

        acc = tf.reduce_mean(tf.cast(tf.equal(label_final, preds_final), tf.float32))

        # tf.summary.scalar('accuracy_train', acc, collections=[SEG_COLLECTION])
        # tf.summary.scalar('accuracy_val', acc, collections=[SEG_VAL_COLLECTION])
        return acc, miou

    def update_learning_rate(self, lr, global_step):
        tf.summary.scalar("lr", lr, collections=[self.opt.train_collection])
        return lr

    def make_optimizer(self, global_step, loss, t_vars):
        cur_lr = self.update_learning_rate(self.opt.lr, global_step)

        cls_optimizer = tf.train.AdamOptimizer(cur_lr, beta1=self.opt.beta1, name="adam").minimize(loss,var_list=t_vars)

        with tf.control_dependencies([cls_optimizer]):
            return tf.no_op(name='optimizers')

    def test(self, images):
        with tf.variable_scope("Seg") as scope:
            preds = self.built_network(images, 1.0, is_training=False)
            preds_lambda, preds_cls, recons = preds

        preds_hair_s, preds_lips, preds_lipscolor, preds_eyebrow = preds_cls[:, :22], preds_cls[:, 22:41], preds_cls[:,
                                                                                                           41:67], preds_cls[
                                                                                                                   :,
                                                                                                                   67:]
        # self.preds = [preds_lambda + 0.5] + map(lambda x:tf.argmax(x, axis=1), [preds_hair_s, preds_lips, preds_lipscolor, preds_eyebrow])
        self.preds = preds_lambda, preds_hair_s, preds_lips, preds_lipscolor, preds_eyebrow


# Converted to TensorFlow .caffemodel
# with the DeepLab-ResNet configuration.
# The batch normalisation layer is provided by
# the slim library (https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim).

from kaffe.tensorflow import Network

class DeepLabResNetModel(Network):
    def setup(self, is_training, num_classes):
        '''Network definition.

        Args:
          is_training: whether to update the running mean and variance of the batch normalisation layer.
                       If the batch size is small, it is better to keep the running mean and variance of
                       the-pretrained model frozen.
          num_classes: number of classes to predict (including background).
        '''
        (self.feed('data')
         .conv(7, 7, 64, 2, 2, biased=False, relu=False, name='conv1')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv1')
         .max_pool(3, 3, 2, 2, name='pool1')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2a_branch1')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn2a_branch1'))

        (self.feed('pool1')
         .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2a_branch2a')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn2a_branch2a')
         .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2a_branch2b')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn2a_branch2b')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2a_branch2c')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn2a_branch2c'))

        (self.feed('bn2a_branch1',
                   'bn2a_branch2c')
         .add(name='res2a')
         .relu(name='res2a_relu')
         .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2b_branch2a')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn2b_branch2a')
         .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2b_branch2b')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn2b_branch2b')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2b_branch2c')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn2b_branch2c'))

        (self.feed('res2a_relu',
                   'bn2b_branch2c')
         .add(name='res2b')
         .relu(name='res2b_relu')
         .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2c_branch2a')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn2c_branch2a')
         .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2c_branch2b')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn2c_branch2b')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2c_branch2c')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn2c_branch2c'))

        (self.feed('res2b_relu',
                   'bn2c_branch2c')
         .add(name='res2c')
         .relu(name='res2c_relu')
         .conv(1, 1, 512, 2, 2, biased=False, relu=False, name='res3a_branch1')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn3a_branch1'))

        (self.feed('res2c_relu')
         .conv(1, 1, 128, 2, 2, biased=False, relu=False, name='res3a_branch2a')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3a_branch2a')
         .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3a_branch2b')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3a_branch2b')
         .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3a_branch2c')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn3a_branch2c'))

        (self.feed('bn3a_branch1',
                   'bn3a_branch2c')
         .add(name='res3a')
         .relu(name='res3a_relu')
         .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res3b1_branch2a')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3b1_branch2a')
         .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3b1_branch2b')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3b1_branch2b')
         .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3b1_branch2c')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn3b1_branch2c'))

        (self.feed('res3a_relu',
                   'bn3b1_branch2c')
         .add(name='res3b1')
         .relu(name='res3b1_relu')
         .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res3b2_branch2a')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3b2_branch2a')
         .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3b2_branch2b')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3b2_branch2b')
         .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3b2_branch2c')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn3b2_branch2c'))

        (self.feed('res3b1_relu',
                   'bn3b2_branch2c')
         .add(name='res3b2')
         .relu(name='res3b2_relu')
         .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res3b3_branch2a')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3b3_branch2a')
         .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3b3_branch2b')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3b3_branch2b')
         .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3b3_branch2c')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn3b3_branch2c'))

        (self.feed('res3b2_relu',
                   'bn3b3_branch2c')
         .add(name='res3b3')
         .relu(name='res3b3_relu')
         .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4a_branch1')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn4a_branch1'))

        (self.feed('res3b3_relu')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4a_branch2a')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4a_branch2a')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4a_branch2b')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4a_branch2b')
         .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4a_branch2c')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn4a_branch2c'))

        (self.feed('bn4a_branch1',
                   'bn4a_branch2c')
         .add(name='res4a')
         .relu(name='res4a_relu')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b1_branch2a')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b1_branch2a')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b1_branch2b')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b1_branch2b')
         .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b1_branch2c')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b1_branch2c'))

        (self.feed('res4a_relu',
                   'bn4b1_branch2c')
         .add(name='res4b1')
         .relu(name='res4b1_relu')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b2_branch2a')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b2_branch2a')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b2_branch2b')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b2_branch2b')
         .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b2_branch2c')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b2_branch2c'))

        (self.feed('res4b1_relu',
                   'bn4b2_branch2c')
         .add(name='res4b2')
         .relu(name='res4b2_relu')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b3_branch2a')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b3_branch2a')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b3_branch2b')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b3_branch2b')
         .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b3_branch2c')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b3_branch2c'))

        (self.feed('res4b2_relu',
                   'bn4b3_branch2c')
         .add(name='res4b3')
         .relu(name='res4b3_relu')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b4_branch2a')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b4_branch2a')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b4_branch2b')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b4_branch2b')
         .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b4_branch2c')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b4_branch2c'))

        (self.feed('res4b3_relu',
                   'bn4b4_branch2c')
         .add(name='res4b4')
         .relu(name='res4b4_relu')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b5_branch2a')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b5_branch2a')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b5_branch2b')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b5_branch2b')
         .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b5_branch2c')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b5_branch2c'))

        (self.feed('res4b4_relu',
                   'bn4b5_branch2c')
         .add(name='res4b5')
         .relu(name='res4b5_relu')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b6_branch2a')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b6_branch2a')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b6_branch2b')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b6_branch2b')
         .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b6_branch2c')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b6_branch2c'))

        (self.feed('res4b5_relu',
                   'bn4b6_branch2c')
         .add(name='res4b6')
         .relu(name='res4b6_relu')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b7_branch2a')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b7_branch2a')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b7_branch2b')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b7_branch2b')
         .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b7_branch2c')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b7_branch2c'))

        (self.feed('res4b6_relu',
                   'bn4b7_branch2c')
         .add(name='res4b7')
         .relu(name='res4b7_relu')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b8_branch2a')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b8_branch2a')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b8_branch2b')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b8_branch2b')
         .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b8_branch2c')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b8_branch2c'))

        (self.feed('res4b7_relu',
                   'bn4b8_branch2c')
         .add(name='res4b8')
         .relu(name='res4b8_relu')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b9_branch2a')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b9_branch2a')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b9_branch2b')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b9_branch2b')
         .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b9_branch2c')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b9_branch2c'))

        (self.feed('res4b8_relu',
                   'bn4b9_branch2c')
         .add(name='res4b9')
         .relu(name='res4b9_relu')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b10_branch2a')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b10_branch2a')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b10_branch2b')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b10_branch2b')
         .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b10_branch2c')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b10_branch2c'))

        (self.feed('res4b9_relu',
                   'bn4b10_branch2c')
         .add(name='res4b10')
         .relu(name='res4b10_relu')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b11_branch2a')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b11_branch2a')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b11_branch2b')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b11_branch2b')
         .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b11_branch2c')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b11_branch2c'))

        (self.feed('res4b10_relu',
                   'bn4b11_branch2c')
         .add(name='res4b11')
         .relu(name='res4b11_relu')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b12_branch2a')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b12_branch2a')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b12_branch2b')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b12_branch2b')
         .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b12_branch2c')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b12_branch2c'))

        (self.feed('res4b11_relu',
                   'bn4b12_branch2c')
         .add(name='res4b12')
         .relu(name='res4b12_relu')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b13_branch2a')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b13_branch2a')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b13_branch2b')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b13_branch2b')
         .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b13_branch2c')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b13_branch2c'))

        (self.feed('res4b12_relu',
                   'bn4b13_branch2c')
         .add(name='res4b13')
         .relu(name='res4b13_relu')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b14_branch2a')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b14_branch2a')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b14_branch2b')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b14_branch2b')
         .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b14_branch2c')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b14_branch2c'))

        (self.feed('res4b13_relu',
                   'bn4b14_branch2c')
         .add(name='res4b14')
         .relu(name='res4b14_relu')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b15_branch2a')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b15_branch2a')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b15_branch2b')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b15_branch2b')
         .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b15_branch2c')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b15_branch2c'))

        (self.feed('res4b14_relu',
                   'bn4b15_branch2c')
         .add(name='res4b15')
         .relu(name='res4b15_relu')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b16_branch2a')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b16_branch2a')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b16_branch2b')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b16_branch2b')
         .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b16_branch2c')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b16_branch2c'))

        (self.feed('res4b15_relu',
                   'bn4b16_branch2c')
         .add(name='res4b16')
         .relu(name='res4b16_relu')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b17_branch2a')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b17_branch2a')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b17_branch2b')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b17_branch2b')
         .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b17_branch2c')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b17_branch2c'))

        (self.feed('res4b16_relu',
                   'bn4b17_branch2c')
         .add(name='res4b17')
         .relu(name='res4b17_relu')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b18_branch2a')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b18_branch2a')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b18_branch2b')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b18_branch2b')
         .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b18_branch2c')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b18_branch2c'))

        (self.feed('res4b17_relu',
                   'bn4b18_branch2c')
         .add(name='res4b18')
         .relu(name='res4b18_relu')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b19_branch2a')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b19_branch2a')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b19_branch2b')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b19_branch2b')
         .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b19_branch2c')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b19_branch2c'))

        (self.feed('res4b18_relu',
                   'bn4b19_branch2c')
         .add(name='res4b19')
         .relu(name='res4b19_relu')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b20_branch2a')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b20_branch2a')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b20_branch2b')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b20_branch2b')
         .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b20_branch2c')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b20_branch2c'))

        (self.feed('res4b19_relu',
                   'bn4b20_branch2c')
         .add(name='res4b20')
         .relu(name='res4b20_relu')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b21_branch2a')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b21_branch2a')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b21_branch2b')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b21_branch2b')
         .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b21_branch2c')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b21_branch2c'))

        (self.feed('res4b20_relu',
                   'bn4b21_branch2c')
         .add(name='res4b21')
         .relu(name='res4b21_relu')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b22_branch2a')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b22_branch2a')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b22_branch2b')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b22_branch2b')
         .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b22_branch2c')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b22_branch2c'))

        (self.feed('res4b21_relu',
                   'bn4b22_branch2c')
         .add(name='res4b22')
         .relu(name='res4b22_relu')
         .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5a_branch1')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn5a_branch1'))

        (self.feed('res4b22_relu')
         .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res5a_branch2a')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn5a_branch2a')
         .atrous_conv(3, 3, 512, 4, padding='SAME', biased=False, relu=False, name='res5a_branch2b')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn5a_branch2b')
         .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5a_branch2c')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn5a_branch2c'))

        (self.feed('bn5a_branch1',
                   'bn5a_branch2c')
         .add(name='res5a')
         .relu(name='res5a_relu')
         .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res5b_branch2a')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn5b_branch2a')
         .atrous_conv(3, 3, 512, 4, padding='SAME', biased=False, relu=False, name='res5b_branch2b')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn5b_branch2b')
         .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5b_branch2c')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn5b_branch2c'))

        (self.feed('res5a_relu',
                   'bn5b_branch2c')
         .add(name='res5b')
         .relu(name='res5b_relu')
         .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res5c_branch2a')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn5c_branch2a')
         .atrous_conv(3, 3, 512, 4, padding='SAME', biased=False, relu=False, name='res5c_branch2b')
         .batch_normalization(activation_fn=tf.nn.relu, name='bn5c_branch2b', is_training=is_training)
         .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5c_branch2c')
         .batch_normalization(is_training=is_training, activation_fn=None, name='bn5c_branch2c'))

        (self.feed('res5b_relu',
                   'bn5c_branch2c')
         .add(name='res5c')
         .relu(name='res5c_relu')
         .atrous_conv(3, 3, num_classes, 6, padding='SAME', relu=False, name='fc1_voc12_c0'))

        (self.feed('res5c_relu')
         .atrous_conv(3, 3, num_classes, 12, padding='SAME', relu=False, name='fc1_voc12_c1'))

        (self.feed('res5c_relu')
         .atrous_conv(3, 3, num_classes, 18, padding='SAME', relu=False, name='fc1_voc12_c2'))

        (self.feed('res5c_relu')
         .atrous_conv(3, 3, num_classes, 24, padding='SAME', relu=False, name='fc1_voc12_c3'))

        (self.feed('fc1_voc12_c0',
                   'fc1_voc12_c1',
                   'fc1_voc12_c2',
                   'fc1_voc12_c3')
         .add(name='fc1_voc12'))


class Seg(BaseModel):
    def __init__(self, opt):
        super(Seg, self).__init__(opt)
        # pass

    def norm(self, inputs, is_training, norm="instance"):
        if norm == "instance":
            out = tf.contrib.layers.instance_norm(inputs)
        elif norm == "batch_norm":
            out = tf.contrib.layers.batch_norm(inputs, decay=0.9, scale=True, updates_collections=None,
                                               is_training=is_training)
        else:
            print("no normaliziation")
            out = inputs
        return out

    def built_network(self, inputs, dropout_rate, is_training=True):

        net = DeepLabResNetModel({'data': inputs}, is_training=is_training, num_classes=self.opt.class_num)
        preds = net.layers['fc1_voc12']
        # print("preds", preds)

        preds = tf.image.resize_bilinear(preds, tf.stack(inputs.get_shape()[1:3]))

        results = tf.argmax(preds, axis=3)

        return preds, results

    def train(self, global_step, images, labels):
        # images_crop = tf.image.crop_to_bounding_box(images, 90, 80, 70, 60)
        images_crop = images  # tf.image.crop_to_bounding_box(images, 70, 90, 50, 40)
        # images_crop = tf.image.crop_to_bounding_box(images, 40, 60, 150, 100)
        # images_crop = tf.image.resize_images(images_crop, [224 ,224])
        with tf.variable_scope("Seg") as scope:
            preds, results = self.built_network(images_crop, 0.5, is_training=True)

        t_vars = tf.trainable_variables()


        self.loss = self.calc_loss(preds, labels)

        self.train_op = self.make_optimizer(global_step, self.loss, t_vars)

        self.acc, miou = self.calc_acc(preds, labels)

        tf.summary.scalar("train/loss", self.loss, collections=[self.opt.train_collection])
        tf.summary.scalar("train/acc", self.acc, collections=[self.opt.train_collection])
        tf.summary.scalar("train/miou", miou, collections=[self.opt.train_collection])

        tf.summary.image("train/_inputs", images, collections=[self.opt.train_collection])

        tf.summary.image("train/map_label", self._decode_label(tf.squeeze(labels, axis=3)), collections=[self.opt.train_collection])
        tf.summary.image("train/map_pred", self._decode_label(results), collections=[self.opt.train_collection])


    def evalute(self, images, labels, reuse=True):

        # images_crop = tf.image.crop_to_bounding_box(images, 90, 80, 70, 60)
        images_crop = images  # tf.image.crop_to_bounding_box(images, 70, 90, 50, 40)
        # images_crop = tf.image.resize_images(images_crop, [224 ,224])

        with tf.variable_scope("Seg", reuse=reuse) as scope:
            preds, results = self.built_network(images_crop, 1.0, is_training=False)

        self.preds = results

        self.loss_val = self.calc_loss(preds, labels)

        self.acc_val, miou_val = self.calc_acc(preds, labels)

        tf.summary.scalar("val/loss", self.loss_val, collections=[self.opt.val_collection])
        tf.summary.scalar("val/acc", self.acc_val, collections=[self.opt.val_collection])
        tf.summary.scalar("val/miou", miou_val, collections=[self.opt.val_collection])

        tf.summary.image("val/_inputs", images, collections=[self.opt.val_collection])

        tf.summary.image("val/map_label", self._decode_label(tf.squeeze(labels, axis=3)), collections=[self.opt.val_collection])
        tf.summary.image("val/map_pred", self._decode_label(results), collections=[self.opt.val_collection])

