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

    def calc_loss(self, preds, labels):
        preds_lambda, preds_hair_s = preds
        labels_lambda, labels_hair_s = labels
        w_loss_hair_s = 1.0
        w_lambda = 1.0
        loss_hair_s = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=preds_hair_s, labels=labels_hair_s) \
                                                ,name="cross_entropy_hair_s")

        loss_lambda = 0 #= tf.reduce_mean(tf.squared_difference(preds_lambda, labels_lambda,name="mse_lambda"))

        #regularization_loss = tf.add(tf.losses.get_regularization_losses())
        total_loss = w_loss_hair_s * loss_hair_s #+ w_lambda * loss_lambda  #+ regularization_loss

        return total_loss, loss_lambda, loss_hair_s

    def calc_acc(self, preds, labels):
        preds_lambda, preds_hair_s = preds
        labels_lambda, labels_hair_s = labels
        correct_prediction = tf.equal(tf.argmax(preds_hair_s, 1), labels_hair_s)
        acc_hair_s = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return  acc_hair_s

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

        logits = architectures.resnet.inference(inputs, 34, self.opt.hair_style_num, 0.0005, is_training=True, transfer_mode=False)
        return None, logits        

        #with slim.arg_scope([slim.conv2d, slim.fully_connected],
        #                    activation_fn=tf.nn.relu,
        #                    weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
        #                    weights_regularizer=slim.l2_regularizer(0.0005)):

        #    net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        #    net = slim.max_pool2d(net, [2, 2], scope='pool1')
        #    net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        #    net = slim.max_pool2d(net, [2, 2], scope='pool2')
        #    net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        #    net = slim.max_pool2d(net, [2, 2], scope='pool3')
        #    net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        #    net = slim.max_pool2d(net, [2, 2], scope='pool4')
        #    net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        #    net = slim.max_pool2d(net, [2, 2], scope='pool5')
        #    flatten = slim.flatten(net, scope="flatten")

        #    fc6_hair_s = slim.fully_connected(flatten, 2048, scope='fc6_hair_s')
        #    drop6_hair_s = slim.dropout(fc6_hair_s, 0.5, scope='dropout6_hair_s')
        #    fc7_hair_s = slim.fully_connected(drop6_hair_s, 512, scope='fc7_hair_s')
        #    drop7_hair_s = slim.dropout(fc7_hair_s, 0.5, scope='dropout7_hair_s')
        #    fc8_hair_s = slim.fully_connected(drop7_hair_s, self.opt.hair_style_num, activation_fn=None, scope='fc8_hair_s')


        #    fc6_lambda = slim.fully_connected(flatten, 4096, scope='fc6_lambda')
        #    drop6_lambda = slim.dropout(fc6_lambda, 0.5, scope='dropout6_lambda')
        #    fc7_lambda = slim.fully_connected(drop6_lambda,1024, scope='fc7_lambda')
        #    drop7_lambda = slim.dropout(fc7_lambda, 0.5, scope='dropout7_lambda')
        #    fc8_lambda = slim.fully_connected(drop7_lambda, self.opt.face_lambda_len, activation_fn=None, scope='fc8_lambda')

        #    return fc8_lambda, fc8_hair_s

    def train(self, global_step, images, labels):
        with tf.variable_scope("vgg16") as scope:
            preds_lambda, preds_hair_s = self.built_network(images)
            preds = preds_lambda, preds_hair_s

        t_vars = tf.trainable_variables()

        self.loss, self.loss_lambda, self.loss_hair_s = self.calc_loss(preds, labels)

        self.train_op = self.make_optimizer(global_step, self.loss, t_vars)

        self.acc = self.calc_acc(preds, labels)


        tf.summary.scalar("train/loss_hair_s", self.loss_hair_s, collections=[self.opt.train_collection])
        tf.summary.scalar("train/loss_lambda", self.loss_lambda, collections=[self.opt.train_collection])

        tf.summary.scalar("train/loss", self.loss, collections=[self.opt.train_collection])
        tf.summary.scalar("train/acc", self.acc, collections=[self.opt.train_collection])
        tf.summary.image("train/_inputs", images, collections=[self.opt.train_collection])


    def evalute(self, images, labels, reuse=True):
        with tf.variable_scope("vgg16", reuse=reuse) as scope:
            preds_lambda, preds_hair_s = self.built_network(images)
            preds = preds_lambda, preds_hair_s

        # labels = tf.Print(labels, [labels], message="lables")
        # t_vars = tf.trainable_variables()
        self.loss_val, self.loss_lambda_val, self.loss_hair_s_val = self.calc_loss(preds, labels)

        self.acc_val = self.calc_acc(preds, labels)


        tf.summary.scalar("val/loss_hair_s", self.loss_hair_s_val, collections=[self.opt.val_collection])
        tf.summary.scalar("val/loss_lambda", self.loss_lambda_val, collections=[self.opt.val_collection])

        tf.summary.scalar("val/loss", self.loss_val, collections=[self.opt.val_collection])
        tf.summary.scalar("val/acc", self.acc_val, collections=[self.opt.val_collection])
        tf.summary.image("val/_inputs", images, collections=[self.opt.val_collection])

    def test(self, images):
        with tf.variable_scope("vgg16") as scope:
            preds = self.built_network(images)
            self.preds = tf.argmax(preds, axis=1)




