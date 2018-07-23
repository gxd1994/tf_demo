import tensorflow as tf
import tensorflow.contrib.slim as slim
import architectures.alexnet
import architectures.resnet
import architectures.vgg
import architectures.googlenet
import architectures.nin
import architectures.densenet



class VGG16():
    def __init__(self, opt):
        self.opt = opt
        # pass

    def calc_loss(self, predictions, labels):
        classification_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predictions, labels=labels) \
                                                ,name="cross_entropy")
        #regularization_loss = tf.add(tf.losses.get_regularization_losses())
        total_loss = classification_loss  #+ regularization_loss
        return total_loss

    def update_learning_rate(self, lr, global_step):

        tf.summary.scalar("lr", lr, collections=[self.opt.train_collection])
        return lr

    def make_optimizer(self, global_step, loss, t_vars):
        cur_lr = self.update_learning_rate(self.opt.lr, global_step)

        cls_optimizer = tf.train.AdamOptimizer(cur_lr, beta1=self.opt.beta1, name="adam").minimize(loss, var_list=t_vars)

        with tf.control_dependencies([cls_optimizer]):
            return tf.no_op(name='optimizers')


    def built_network(self, inputs):
        # return architectures.googlenet.inference(inputs, self.opt.class_num, 0.0005, 0.5, True, transfer_mode=False)

        # return architectures.resnet.inference(inputs, 34,self.opt.class_num, 0.0005, is_training=True, transfer_mode=False)

        # with slim.arg_scope([slim.conv2d, slim.fully_connected],
        #                     activation_fn=tf.nn.relu,
        #                     weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
        #                     weights_regularizer=slim.l2_regularizer(0.0005)):
        #
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
        #     net = slim.flatten(net, scope="flatten")
        #     net = slim.fully_connected(net, 4096, scope='fc6')
        #     net = slim.dropout(net, 0.5, scope='dropout6')
        #     net = slim.fully_connected(net, 4096, scope='fc7')
        #     net = slim.dropout(net, 0.5, scope='dropout7')
        #     net = slim.fully_connected(net, self.opt.class_num, activation_fn=None, scope='fc8')
        #
        #     return net

        conv1 = tf.layers.conv2d(inputs, 64, 7, 1, padding="SAME", activation=tf.nn.relu, name="conv1")
        pool1 = tf.layers.max_pooling2d(conv1, 2, 2, padding="SAME", name="pool1")
        conv2 = tf.layers.conv2d(pool1, 64, 3, 1, padding="SAME", activation=tf.nn.relu, name="conv2")
        pool2 = tf.layers.max_pooling2d(conv2, 2, 2, padding="SAME", name="pool2")
        conv3 = tf.layers.conv2d(pool2, 64, 3, 1, padding="SAME", activation=tf.nn.relu, name="conv3")
        pool3 = tf.layers.max_pooling2d(conv3, 2, 2, padding="SAME", name="pool3")

        flatten = tf.layers.flatten(pool3, name = "flatten")
        fc1 = tf.layers.dense(flatten, 64, activation=tf.nn.relu, name="fc1")
        fc2 = tf.layers.dense(fc1, self.opt.class_num, activation=None, name="fc2")

        return fc2


    def train(self, global_step, images, labels):
        with tf.variable_scope("vgg16") as scope:
            preds = self.built_network(images)

        t_vars = tf.trainable_variables()
        self.loss = self.calc_loss(preds, labels)
        self.train_op = self.make_optimizer(global_step, self.loss, t_vars)

        correct_prediction = tf.equal(tf.argmax(preds, 1), labels)
        self.acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.summary.scalar("train/loss", self.loss, collections=[self.opt.train_collection])
        tf.summary.scalar("train/acc", self.acc, collections=[self.opt.train_collection])
        tf.summary.image("train/_inputs", images, collections=[self.opt.train_collection])


    def evalute(self, images, labels, reuse=True):
        with tf.variable_scope("vgg16", reuse=reuse) as scope:
            preds = self.built_network(images)

        # labels = tf.Print(labels, [labels], message="lables")
        # t_vars = tf.trainable_variables()
        self.loss_val = self.calc_loss(preds, labels)

        correct_prediction = tf.equal(tf.argmax(preds, 1), labels)
        self.acc_val = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.summary.scalar("val/loss", self.loss_val, collections=[self.opt.val_collection])
        tf.summary.scalar("val/acc", self.acc_val, collections=[self.opt.val_collection])
        tf.summary.image("val/_inputs", images, collections=[self.opt.val_collection])

    def test(self, images):
        with tf.variable_scope("vgg16") as scope:
            preds = self.built_network(images)
            self.preds = tf.argmax(preds, axis=1)




