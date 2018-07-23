import tensorflow as tf
import tensorflow.contrib.slim as slim
import architectures.alexnet
import architectures.resnet
import architectures.vgg
import architectures.googlenet
import architectures.nin
import architectures.densenet
import architectures.common as common


class VGG16():
    def __init__(self, opt):
        self.opt = opt
        # pass

    def calc_loss(self, preds, labels):
        preds_lambda, preds_hair_s = preds
        labels_lambda, labels_hair_s = labels
        w_loss_hair_s = self.opt.w_hair_style
        w_lambda = self.opt.w_lambda 
        loss_hair_s = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=preds_hair_s, labels=labels_hair_s) \
                                                ,name="cross_entropy_hair_s")

        loss_lambda = tf.reduce_mean(tf.abs(preds_lambda + 0.5 - labels_lambda ,name="l1_lambda"))
        #loss_lambda = tf.reduce_mean(tf.squared_difference(preds_lambda, labels_lambda,name="mse_lambda"))

        #regularization_loss = tf.add(tf.losses.get_regularization_losses())
        #total_loss = w_loss_hair_s * loss_hair_s + w_lambda * loss_lambda  #+ regularization_loss
        total_loss = w_lambda * loss_lambda  #+ regularization_loss

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

    # a helper function to have a more organized code
    def getModel(self, x, num_output, wd, is_training, num_blocks=[3, 4, 6, 3],  # defaults to 50-layer network
                  bottleneck= True, transfer_mode= False):
        conv_weight_initializer = tf.truncated_normal_initializer(stddev= 0.1)
        fc_weight_initializer = tf.truncated_normal_initializer(stddev= 0.01)
        with tf.variable_scope('scale1'):
            x = common.spatialConvolution(x, 7, 2, 64, weight_initializer= conv_weight_initializer, wd= wd)
            x = common.batchNormalization(x, is_training= is_training)
            x = tf.nn.relu(x)
    
        with tf.variable_scope('scale2'):
            x = common.maxPool(x, 3, 2)
            x = common.resnetStack(x, num_blocks[0], 1, 64, bottleneck, wd= wd, is_training= is_training)
    
        with tf.variable_scope('scale3'):
            x = common.resnetStack(x, num_blocks[1], 2, 128, bottleneck, wd= wd, is_training= is_training)
    
        with tf.variable_scope('scale4'):
            x = common.resnetStack(x, num_blocks[2], 2, 256, bottleneck, wd= wd, is_training= is_training)
    
        with tf.variable_scope('scale5'):
            x = common.resnetStack(x, num_blocks[3], 2, 512, bottleneck, wd= wd, is_training= is_training)
        
        return x 
       # # post-net
       # x = tf.reduce_mean(x, reduction_indices= [1, 2], name= "avg_pool")
    
       # if not transfer_mode:
       #   with tf.variable_scope('output'):
       #     x = common.fullyConnected(x, num_output, weight_initializer= fc_weight_initializer, bias_initializer= tf.zeros_initializer, wd= wd)
       # else:
       #   with tf.variable_scope('transfer_output'):
       #     x = common.fullyConnected(x, num_output, weight_initializer= fc_weight_initializer, bias_initializer= tf.zeros_initializer, wd= wd)
    
       # return x

    def built_network(self, inputs, dropout_rate, is_training=True):
        
        #logits = architectures.vgg.inference(inputs, self.opt.hair_style_num, 0.0005, 0.5, True, transfer_mode=False)
        #logits = architectures.googlenet.inference(inputs, self.opt.hair_style_num, 0.0005, 0.5, True, transfer_mode=False)

        #logits = architectures.resnet.inference(inputs, 34, self.opt.hair_style_num, 0.0005, is_training=True, transfer_mode=False)
        #return None, logits        
        num_blockes= []
        bottleneck= False
        depth = 18
        x = inputs
        wd = 0.0005
        if depth == 18:
          num_blocks= [2, 2, 2, 2]
        elif depth == 34:
          num_blocks= [3, 4, 6, 3]
        elif depth == 50:
          num_blocks= [3, 4, 6, 3]
          bottleneck= True
        elif depth == 101:
          num_blocks= [3, 4, 23, 3]
          bottleneck= True
        elif depth == 152:
          num_blocks= [3, 8, 36, 3]
          bottleneck= True

        with tf.variable_scope('convs_comm'): 
            network = self.getModel(x, 0, wd, is_training, num_blocks= num_blocks, bottleneck= bottleneck, transfer_mode= False)
                        
        print("network_comm", network)
        print("inputs", inputs)
        network_comm = common.flatten(network)
    
        with tf.variable_scope('fc1_hair_s'): 
            network = common.fullyConnected(network_comm, 4096, wd= wd)
            network = tf.nn.relu(network)
            #network = common.batchNormalization(network, is_training= is_training)
            network = tf.nn.dropout(network, dropout_rate)
        with tf.variable_scope('fc2_hair_s'):
            network = common.fullyConnected(network, 4096, wd= wd)
            network = tf.nn.relu(network)
            #network = common.batchNormalization(network, is_training= is_training)
            network = tf.nn.dropout(network, dropout_rate)
        with tf.variable_scope('output_hair_s'):
            preds_hair_s = common.fullyConnected(network, self.opt.hair_style_num, wd= wd)


        with tf.variable_scope('fc1_lambda'): 
            network = common.fullyConnected(network_comm, 2048, wd= wd)
            network = tf.nn.relu(network)
            #network = common.batchNormalization(network, is_training= is_training)
            network = tf.nn.dropout(network, dropout_rate)
        with tf.variable_scope('fc2_lambda'):
            network = common.fullyConnected(network, 512, wd= wd)
            network = tf.nn.relu(network)
            #network = common.batchNormalization(network, is_training= is_training)
            network = tf.nn.dropout(network, dropout_rate)
        with tf.variable_scope('output_lambda'):
            preds_lambda = common.fullyConnected(network, self.opt.face_lambda_len, wd= wd)
            #preds_lambda = tf.sigmoid(preds_lambda, name="sigmoid") 
        
        
        return preds_lambda, preds_hair_s
    
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
        #images_crop = tf.image.crop_to_bounding_box(images, 40, 60, 150, 100)                
        images_crop = images#tf.image.crop_to_bounding_box(images, 40, 60, 150, 100)                

        with tf.variable_scope("vgg16") as scope:
            preds_lambda, preds_hair_s = self.built_network(images_crop, 0.5)
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
        tf.summary.image("train/_inputs_crop", images_crop, collections=[self.opt.train_collection])


    def evalute(self, images, labels, reuse=True):

        #images_crop = tf.image.crop_to_bounding_box(images, 40, 60, 150, 100)                
        images_crop = images#tf.image.crop_to_bounding_box(images, 40, 60, 150, 100)                
        
        with tf.variable_scope("vgg16", reuse=reuse) as scope:
            preds_lambda, preds_hair_s = self.built_network(images_crop, 1.0, is_training=False)
            preds = preds_lambda, preds_hair_s
            self.preds = preds

        # labels = tf.Print(labels, [labels], message="lables")
        # t_vars = tf.trainable_variables()
        self.loss_val, self.loss_lambda_val, self.loss_hair_s_val = self.calc_loss(preds, labels)

        self.acc_val = self.calc_acc(preds, labels)


        tf.summary.scalar("val/loss_hair_s", self.loss_hair_s_val, collections=[self.opt.val_collection])
        tf.summary.scalar("val/loss_lambda", self.loss_lambda_val, collections=[self.opt.val_collection])

        tf.summary.scalar("val/loss", self.loss_val, collections=[self.opt.val_collection])
        tf.summary.scalar("val/acc", self.acc_val, collections=[self.opt.val_collection])
        tf.summary.image("val/_inputs", images, collections=[self.opt.val_collection])
        tf.summary.image("val/_inputs_crop", images_crop, collections=[self.opt.val_collection])

    def test(self, images):
        with tf.variable_scope("vgg16") as scope:
            preds_lambda, preds_hair_s = self.built_network(images, 1.0, is_training=False)
            #self.preds_hair_s = tf.argmax(preds_hair_s, axis=1)
            self.preds = preds_lambda, preds_hair_s




