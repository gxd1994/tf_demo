import tensorflow as tf
import tensorflow.contrib.slim as slim
import architectures.alexnet
import architectures.resnet
import architectures.vgg
import architectures.googlenet
import architectures.nin
import architectures.densenet
import architectures.common as common

Total_Num_Cls = 103

class BaseModel():
    def __init__(self, opt):
        self.opt = opt
        # pass

    def calc_loss(self, preds, labels):
        preds_lambda, preds_cls, recons = preds
        (labels_lambda, labels_hair_s, labels_lips, labels_lipscolor, labels_eyebrow), images = labels
        preds_hair_s, preds_lips, preds_lipscolor, preds_eyebrow = preds_cls[:,:22], preds_cls[:,22:41], preds_cls[:,41:67], preds_cls[:,67:]
        
        w_cls = self.opt.w_cls
        w_lambda = self.opt.w_lambda 
        
        #cls loss
        loss_hair_s = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=preds_hair_s, labels=labels_hair_s), name="cross_entropy_hair_s")
        loss_lips = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=preds_lips, labels=labels_lips), name="cross_entropy_lips")
        loss_lipscolor = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=preds_lipscolor, labels=labels_lipscolor), name="cross_entropy_lipscolor")
        loss_eyebrow = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=preds_eyebrow, labels=labels_eyebrow), name="cross_entropy_eyebrow")
       
        #lambda loss
        #labels_lambda = tf.Print(labels_lambda, [labels_lambda], message="labels_lambda")
        #loss_lambda = tf.reduce_mean(tf.squared_difference(preds_lambda, labels_lambda,name="mse_lambda"))
        loss_lambda = tf.reduce_mean(tf.abs(preds_lambda + 0.5 - labels_lambda ,name="l1_lambda"))
        
        #reconstruction loss
        print("recons image", recons, images)
        loss_recons = tf.reduce_mean(tf.abs(recons - images,name="l1_recons"))

        #regularization_loss = tf.add(tf.losses.get_regularization_losses())
        #total_loss = w_loss_hair_s * loss_hair_s + w_lambda * loss_lambda  #+ regularization_loss
        #total_loss = w_lambda * loss_lambda + loss_recons + w_cls * (loss_hair_s + loss_lips + loss_lipscolor + loss_eyebrow)  #+ regularization_loss
        total_loss = w_lambda * loss_lambda + w_cls * (loss_hair_s + loss_lips + loss_lipscolor + loss_eyebrow)  #+ regularization_loss

        return total_loss, loss_lambda, loss_recons, loss_hair_s, loss_lips, loss_lipscolor, loss_eyebrow

    def calc_acc(self, preds, labels):
        ppreds_lambda, preds_cls, recons = preds
        (labels_lambda, labels_hair_s, labels_lips, labels_lipscolor, labels_eyebrow), images = labels
        preds_hair_s, preds_lips, preds_lipscolor, preds_eyebrow = preds_cls[:,:22], preds_cls[:,22:41], preds_cls[:,41:67], preds_cls[:,67:]
        
        correct_prediction = tf.equal(tf.argmax(preds_hair_s, 1), labels_hair_s)
        acc_hair_s = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        correct_prediction = tf.equal(tf.argmax(preds_lips, 1), labels_lips)
        acc_lips = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        correct_prediction = tf.equal(tf.argmax(preds_lipscolor, 1), labels_lipscolor)
        acc_lipscolor = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        correct_prediction = tf.equal(tf.argmax(preds_eyebrow, 1), labels_eyebrow)
        acc_eyebrow = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return  acc_hair_s, acc_lips, acc_lipscolor, acc_eyebrow

    def update_learning_rate(self, lr, global_step):

        tf.summary.scalar("lr", lr, collections=[self.opt.train_collection])
        return lr

    def make_optimizer(self, global_step, loss, t_vars):
        cur_lr = self.update_learning_rate(self.opt.lr, global_step)

        cls_optimizer = tf.train.AdamOptimizer(cur_lr, beta1=self.opt.beta1, name="adam").minimize(loss, var_list=t_vars)

        with tf.control_dependencies([cls_optimizer]):
            return tf.no_op(name='optimizers')

    def test(self, images):
        with tf.variable_scope("vgg16") as scope:
            preds = self.built_network(images, 1.0, is_training=False)
            preds_lambda, preds_cls, recons = preds

        preds_hair_s, preds_lips, preds_lipscolor, preds_eyebrow = preds_cls[:,:22], preds_cls[:,22:41], preds_cls[:,41:67], preds_cls[:,67:]
        #self.preds = [preds_lambda + 0.5] + map(lambda x:tf.argmax(x, axis=1), [preds_hair_s, preds_lips, preds_lipscolor, preds_eyebrow])
        self.preds = preds_lambda, preds_hair_s, preds_lips, preds_lipscolor, preds_eyebrow

       
 

class VGG16(BaseModel):
    def __init__(self, opt):
        super(VGG16, self).__init__(opt)
        #pass
    def norm(self, inputs, is_training, norm="instance"):
        if norm == "instance":
            out = tf.contrib.layers.instance_norm(inputs)
        elif norm == "batch_norm":
            out = tf.contrib.layers.batch_norm(inputs, decay=0.9, scale=True, updates_collections=None, is_training=is_training)
        else:
            print("no normaliziation")
            out = inputs
        return out
        
    #def calc_loss(self, preds, labels):
    #    preds_lambda, preds_hair_s = preds
    #    labels_lambda, labels_hair_s = labels
    #    w_loss_hair_s = self.opt.w_hair_style
    #    w_lambda = self.opt.w_lambda 
    #    loss_hair_s = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=preds_hair_s, labels=labels_hair_s) \
    #                                            ,name="cross_entropy_hair_s")

    #    loss_lambda = tf.reduce_mean(tf.squared_difference(preds_lambda, labels_lambda,name="mse_lambda"))

    #    #regularization_loss = tf.add(tf.losses.get_regularization_losses())
    #    #total_loss = w_loss_hair_s * loss_hair_s + w_lambda * loss_lambda  #+ regularization_loss
    #    total_loss = w_lambda * loss_lambda  #+ regularization_loss

    #    return total_loss, loss_lambda, loss_hair_s

    #def calc_acc(self, preds, labels):
    #    preds_lambda, preds_hair_s = preds
    #    labels_lambda, labels_hair_s = labels
    #    correct_prediction = tf.equal(tf.argmax(preds_hair_s, 1), labels_hair_s)
    #    acc_hair_s = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #    return  acc_hair_s
    

    ## a helper function to have a more organized code
    #def getModel(self, x, num_output, wd, is_training, num_blocks=[3, 4, 6, 3],  # defaults to 50-layer network
    #              bottleneck= True, transfer_mode= False):
    #    conv_weight_initializer = tf.truncated_normal_initializer(stddev= 0.1)
    #    fc_weight_initializer = tf.truncated_normal_initializer(stddev= 0.01)
    #    with tf.variable_scope('scale1'):
    #        x = common.spatialConvolution(x, 7, 2, 64, weight_initializer= conv_weight_initializer, wd= wd)
    #        x = common.batchNormalization(x, is_training= is_training)
    #        x = tf.nn.relu(x)
    #
    #    with tf.variable_scope('scale2'):
    #        x = common.maxPool(x, 3, 2)
    #        x = common.resnetStack(x, num_blocks[0], 1, 64, bottleneck, wd= wd, is_training= is_training)
    #
    #    with tf.variable_scope('scale3'):
    #        x = common.resnetStack(x, num_blocks[1], 2, 128, bottleneck, wd= wd, is_training= is_training)
    #
    #    with tf.variable_scope('scale4'):
    #        x = common.resnetStack(x, num_blocks[2], 2, 256, bottleneck, wd= wd, is_training= is_training)
    #
    #    with tf.variable_scope('scale5'):
    #        x = common.resnetStack(x, num_blocks[3], 2, 512, bottleneck, wd= wd, is_training= is_training)
    #    
    #    return x 
    #   # # post-net
    #   # x = tf.reduce_mean(x, reduction_indices= [1, 2], name= "avg_pool")
    #
    #   # if not transfer_mode:
    #   #   with tf.variable_scope('output'):
    #   #     x = common.fullyConnected(x, num_output, weight_initializer= fc_weight_initializer, bias_initializer= tf.zeros_initializer, wd= wd)
    #   # else:
    #   #   with tf.variable_scope('transfer_output'):
    #   #     x = common.fullyConnected(x, num_output, weight_initializer= fc_weight_initializer, bias_initializer= tf.zeros_initializer, wd= wd)
    #
    #   # return x

    #def built_network(self, inputs, dropout_rate, is_training=True):
    #    
    #    #logits = architectures.vgg.inference(inputs, self.opt.hair_style_num, 0.0005, 0.5, True, transfer_mode=False)
    #    #logits = architectures.googlenet.inference(inputs, self.opt.hair_style_num, 0.0005, 0.5, True, transfer_mode=False)

    #    #logits = architectures.resnet.inference(inputs, 34, self.opt.hair_style_num, 0.0005, is_training=True, transfer_mode=False)
    #    #return None, logits        
    #    num_blockes= []
    #    bottleneck= False
    #    depth = 18
    #    x = inputs
    #    wd = 0.0005
    #    if depth == 18:
    #      num_blocks= [2, 2, 2, 2]
    #    elif depth == 34:
    #      num_blocks= [3, 4, 6, 3]
    #    elif depth == 50:
    #      num_blocks= [3, 4, 6, 3]
    #      bottleneck= True
    #    elif depth == 101:
    #      num_blocks= [3, 4, 23, 3]
    #      bottleneck= True
    #    elif depth == 152:
    #      num_blocks= [3, 8, 36, 3]
    #      bottleneck= True

    #    with tf.variable_scope('convs_comm'): 
    #        network = self.getModel(x, 0, wd, is_training, num_blocks= num_blocks, bottleneck= bottleneck, transfer_mode= False)
    #                    
    #    network = tf.Print(network, [network[0,4:,:,0]],"net_comm feature", -1, 50)
    #    print("network_comm", network)
    #    print("inputs", inputs)
    #    network_comm = common.flatten(network)
    #
    #    with tf.variable_scope('fc1_hair_s'): 
    #        network = common.fullyConnected(network_comm, 4096, wd= wd)
    #        network = tf.nn.relu(network)
    #        #network = common.batchNormalization(network, is_training= is_training)
    #        network = tf.nn.dropout(network, dropout_rate)
    #    with tf.variable_scope('fc2_hair_s'):
    #        network = common.fullyConnected(network, 4096, wd= wd)
    #        network = tf.nn.relu(network)
    #        #network = common.batchNormalization(network, is_training= is_training)
    #        network = tf.nn.dropout(network, dropout_rate)
    #    with tf.variable_scope('output_hair_s'):
    #        preds_hair_s = common.fullyConnected(network, self.opt.hair_style_num, wd= wd)


    #    with tf.variable_scope('fc1_lambda'): 
    #        network = common.fullyConnected(network_comm, 2048, wd= wd)
    #        network = common.batchNormalization(network, is_training= is_training)
    #        network = tf.nn.relu(network)
    #        network = tf.nn.dropout(network, dropout_rate)
    #    with tf.variable_scope('fc2_lambda'):
    #        network = common.fullyConnected(network, 512, wd= wd)
    #        network = common.batchNormalization(network, is_training= is_training)
    #        network = tf.nn.relu(network)
    #        network = tf.nn.dropout(network, dropout_rate)
    #    with tf.variable_scope('output_lambda'):
    #        preds_lambda = common.fullyConnected(network, self.opt.face_lambda_len, wd= wd)
    #        #preds_lambda = tf.sigmoid(preds_lambda, name="sigmoid") 
    #    
    #    
    #    return preds_lambda, preds_hair_s
    


    #def built_network(self, inputs, dropout_rate, is_training=True):
    #    
    #    #logits = architectures.vgg.inference(inputs, self.opt.hair_style_num, 0.0005, 0.5, True, transfer_mode=False)
    #    #logits = architectures.googlenet.inference(inputs, self.opt.hair_style_num, 0.0005, 0.5, True, transfer_mode=False)

    #    #logits = architectures.resnet.inference(inputs, 34, self.opt.hair_style_num, 0.0005, is_training=True, transfer_mode=False)
    #    #return None, logits        
 

    #    # The VGG architecture, the default type is 'A'
    #    wd = 0.0005
    #    model_type = 'a' 
    #    print("model_type:",model_type)
    #    # Create tables describing VGG configurations A, B, D, E
    #    if model_type == 'a':
    #       config = [64, 'M', 128, 'M', 256, 'M']
    #    elif model_type == 'A':
    #       config = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    #    elif model_type == 'B':
    #       config = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    #    elif model_type == 'D':
    #       config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    #    elif model_type == 'E':
    #       config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
    #    else:
    #       print('Unknown model type: ' + model_type + ' | Please specify a modelType A or B or D or E')
    #    
    #    network= inputs
    #    

    #    with tf.variable_scope('convs_comm'): 
    #        for k,v in enumerate(config):
    #            if v == 'M':
    #                network= common.maxPool(network, 2, 2)
    #            else:  
    #                with tf.variable_scope('conv'+str(k)):
    #                    network = common.spatialConvolution(network, 3, 1, v, wd=wd)
    #                    network = common.batchNormalization(network, is_training=is_training)
    #                    network = tf.nn.relu(network)
    #       
    #         
    #        #network = tf.image.crop_to_bounding_box(network, 1,2,4,3)
    #        network = tf.Print(network, [network[0,14:,:,0]],"net_comm feature", -1, 50)
    #        print("network_comm", network)
    #        print("inputs", inputs)
    #        network_comm = common.flatten(network)
    #
    #    with tf.variable_scope('fc1_hair_s'): 
    #        network = common.fullyConnected(network_comm, 4096, wd= wd)
    #        #network = common.batchNormalization(network, is_training= is_training)
    #        network = tf.nn.dropout(network, dropout_rate)
    #    with tf.variable_scope('fc2_hair_s'):
    #        network = common.fullyConnected(network, 4096, wd= wd)
    #        network = tf.nn.relu(network)
    #        #network = common.batchNormalization(network, is_training= is_training)
    #        network = tf.nn.dropout(network, dropout_rate)
    #    with tf.variable_scope('output_hair_s'):
    #        preds_hair_s = common.fullyConnected(network, self.opt.hair_style_num, wd= wd)


    #    with tf.variable_scope('fc1_lambda'): 
    #        network = common.fullyConnected(network_comm, 2048, wd= wd)
    #        network = common.batchNormalization(network, is_training= is_training)
    #        network = tf.nn.relu(network)
    #        network = tf.nn.dropout(network, dropout_rate)
    #    with tf.variable_scope('fc2_lambda'):
    #        network = common.fullyConnected(network, 256, wd= wd)
    #        network = common.batchNormalization(network, is_training= is_training)
    #        network = tf.nn.relu(network)
    #        network = tf.nn.dropout(network, dropout_rate)
    #    with tf.variable_scope('output_lambda'):
    #        preds_lambda = common.fullyConnected(network, self.opt.face_lambda_len, wd= wd)
    #        #preds_lambda = tf.sigmoid(preds_lambda, name="sigmoid") 
    #    
    #    
    #    return preds_lambda, preds_hair_s
    
    def built_network(self, inputs, dropout_rate, is_training=True):
        
        #logits = architectures.vgg.inference(inputs, self.opt.hair_style_num, 0.0005, 0.5, True, transfer_mode=False)
        #logits = architectures.googlenet.inference(inputs, self.opt.hair_style_num, 0.0005, 0.5, True, transfer_mode=False)

        #logits = architectures.resnet.inference(inputs, 34, self.opt.hair_style_num, 0.0005, is_training=True, transfer_mode=False)
        #return None, logits        
 

        # The VGG architecture, the default type is 'A'
        wd = 0.0005
        model_type = 'a' 
        print("model_type:",model_type)
        # Create tables describing VGG configurations A, B, D, E
        if model_type == 'a':
           config = [64, 'M', 128, 'M', 256, 'M', 512, 'M', 512, 'M']
        elif model_type == 'A':
           config = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
        elif model_type == 'B':
           config = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
        elif model_type == 'D':
           config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        elif model_type == 'E':
           config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
        else:
           print('Unknown model type: ' + model_type + ' | Please specify a modelType A or B or D or E')
        
        network= inputs
        norm_type = "batch_norm" 
         #kernel_initializer = tf.contrib.layers.xavier_initializer()  #variance_scaling_initializer()
        kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
        with tf.variable_scope('convs_comm'): 
            for k,v in enumerate(config):
                if v == 'M':
                    network= tf.layers.max_pooling2d(network, 2, 2, padding="SAME")
                else:  
                    with tf.variable_scope('conv'+str(k)):
                        network = tf.layers.conv2d(network, v, 3, activation=None, padding="SAME", kernel_initializer=kernel_initializer)
                        network = self.norm(network, is_training=is_training, norm=norm_type)
                        network = tf.nn.relu(network)

           
            network_comm_map = network 
            #network = tf.image.crop_to_bounding_box(network, 1,2,4,3)
            network_comm_map = tf.Print(network_comm_map, [network_comm_map[0,4:,:,0]],"net_comm feature", -1, 50)
            print("network_comm", network_comm_map)
            print("inputs", inputs)
            network_comm = common.flatten(network_comm_map)
             
        with tf.variable_scope('decode'):
            config = [512, 256, 128, 64]
            for k,v in enumerate(config):
                with tf.variable_scope('deconv'+str(k)):
                    network = tf.layers.conv2d_transpose(network, v, 3, (2, 2), padding="SAME") 
                    network = self.norm(network, is_training=is_training, norm=norm_type)
                    network = tf.nn.relu(network)
            with tf.variable_scope('pred'):
                network = tf.layers.conv2d_transpose(network, self.opt.num_channels, 3, (2,2), padding="SAME")
                #recons = tf.nn.relu(network) 
                recons = tf.sigmoid(network) 
                #recons = network 
            print("recons", recons)

                 
        with tf.variable_scope('fc1_cls'): 
            network = tf.layers.dense(network_comm, 2048, activation=None, kernel_initializer = kernel_initializer)
            network = self.norm(network, is_training=is_training, norm=norm_type)
            network = tf.nn.relu(network)
            network = tf.nn.dropout(network, dropout_rate)
        with tf.variable_scope('fc2_cls'):
            network = tf.layers.dense(network, 256, activation=None, kernel_initializer = kernel_initializer)
            network = self.norm(network, is_training=is_training, norm=norm_type)
            network = tf.nn.relu(network)
            network = tf.nn.dropout(network, dropout_rate)
        with tf.variable_scope('output_cls'):
            preds_cls = tf.layers.dense(network, Total_Num_Cls, activation=None, kernel_initializer = kernel_initializer, use_bias=False)
            #preds_cls = tf.layers.dense(network, Total_Num_Cls - 22, activation=None, kernel_initializer = kernel_initializer, use_bias=False)
    
        #with tf.variable_scope('output_cls_hair'):
        #    network = tf.reduce_mean(network_comm_map, [1, 2], keep_dims=True, name="global_pooling")
        #    network = tf.nn.dropout(network, dropout_rate)
        #    network = tf.layers.conv2d(network, 22, 1, activation=None, padding="SAME", kernel_initializer=kernel_initializer)
        #    pred_hair_s = tf.squeeze(network, [1, 2], name="squeeze_hair_s")
        #
        #preds_cls = tf.concat([pred_hair_s, preds_cls], axis=1)
        
        # face lambda 
        with tf.variable_scope('fc1_lambda'): 
            network = tf.layers.dense(network_comm, 2048, activation=None, kernel_initializer = kernel_initializer)
            network = self.norm(network, is_training=is_training, norm=norm_type)
            network = tf.nn.relu(network)
            network = tf.nn.dropout(network, dropout_rate)
        with tf.variable_scope('fc2_lambda'):
            network = tf.layers.dense(network, 256, activation=None, kernel_initializer = kernel_initializer)
            network = self.norm(network, is_training=is_training, norm=norm_type)
            network = tf.nn.relu(network)
            network = tf.nn.dropout(network, dropout_rate)
        with tf.variable_scope('output_lambda'):
            preds_lambda = tf.layers.dense(network, self.opt.face_lambda_len, activation=None, kernel_initializer = kernel_initializer, use_bias=False)
            #preds_lambda = tf.sigmoid(preds_lambda, name="sigmoid") 
            preds_lambda = tf.nn.tanh(preds_lambda, name="tanh") 
            #preds_lambda = preds_lambda #tf.sigmoid(preds_lambda, name="sigmoid") 
        
        
        return preds_lambda, preds_cls, recons
    
    def train(self, global_step, images, labels):
        #images_crop = tf.image.crop_to_bounding_box(images, 90, 80, 70, 60)                
        images_crop = images#tf.image.crop_to_bounding_box(images, 70, 90, 50, 40)                
        #images_crop = tf.image.crop_to_bounding_box(images, 40, 60, 150, 100)                
        #images_crop = tf.image.resize_images(images_crop, [224 ,224])
        with tf.variable_scope("vgg16") as scope:
            preds = self.built_network(images_crop, 0.5, is_training=True)
            preds_lambda, preds_cls, recons = preds
        
        t_vars = tf.trainable_variables()
        
        preds_lambda, preds_cls, recons = preds
        
        labels = labels, images_crop
        self.loss, loss_lambda, loss_recons, loss_hair_s, loss_lips, loss_lipscolor, loss_eyebrow  = self.calc_loss(preds, labels)

        self.train_op = self.make_optimizer(global_step, self.loss, t_vars)

        acc_hair_s, acc_lips, acc_lipscolor, acc_eyebrow  = self.calc_acc(preds, labels)
        self.acc = (acc_hair_s + acc_lips + acc_lipscolor + acc_eyebrow) / 4.0

        tf.summary.scalar("train/loss_lambda", loss_lambda, collections=[self.opt.train_collection])
        tf.summary.scalar("train/loss_hair_s", loss_hair_s, collections=[self.opt.train_collection])
        tf.summary.scalar("train/loss_lips", loss_lips, collections=[self.opt.train_collection])
        tf.summary.scalar("train/loss_lipscolor", loss_lipscolor, collections=[self.opt.train_collection])
        tf.summary.scalar("train/loss_eyebrow", loss_eyebrow, collections=[self.opt.train_collection])

        tf.summary.scalar("train/acc_hair_s", acc_hair_s, collections=[self.opt.train_collection])
        tf.summary.scalar("train/acc_lips", acc_lips, collections=[self.opt.train_collection])
        tf.summary.scalar("train/acc_lipscolor", acc_lips, collections=[self.opt.train_collection])
        tf.summary.scalar("train/acc_eyebrow", acc_eyebrow, collections=[self.opt.train_collection])
        
        tf.summary.scalar("train/loss_recons", loss_recons, collections=[self.opt.train_collection])

        tf.summary.scalar("train/loss", self.loss, collections=[self.opt.train_collection])
        tf.summary.scalar("train/acc", self.acc, collections=[self.opt.train_collection])
        #tf.summary.image("train/_inputs", images, collections=[self.opt.train_collection])
        #tf.summary.image("train/_inputs_crop", images_crop, collections=[self.opt.train_collection])
        tf.summary.image("train/_inputs", images, collections=[self.opt.train_collection])
        #tf.summary.image("train/_inputs_temp", tf.expand_dims(images[:,:,:,1], -1), collections=[self.opt.train_collection])
        tf.summary.image("train/_inputs_crop", images_crop, collections=[self.opt.train_collection])
        tf.summary.image("train/recons", recons, collections=[self.opt.train_collection])


    def evalute(self, images, labels, reuse=True):

        #images_crop = tf.image.crop_to_bounding_box(images, 90, 80, 70, 60)                
        images_crop = images#tf.image.crop_to_bounding_box(images, 70, 90, 50, 40)                
        #images_crop = tf.image.resize_images(images_crop, [224 ,224])
        
        with tf.variable_scope("vgg16", reuse=reuse) as scope:
            preds = self.built_network(images_crop, 1.0, is_training=False)
            #preds_lambda, preds_hair_s = self.built_network(images_crop, 1.0, is_training=False)
            preds_lambda, preds_cls, recons = preds

        labels = labels, images_crop
        # t_vars = tf.trainable_variables()
        # labels = tf.Print(labels, [labels], message="lables")
        
        preds_hair_s, preds_lips, preds_lipscolor, preds_eyebrow = preds_cls[:,:22], preds_cls[:,22:41], preds_cls[:,41:67], preds_cls[:,67:]
        self.preds = preds_lambda, preds_hair_s, preds_lips, preds_lipscolor, preds_eyebrow

        self.loss_val, loss_lambda, loss_recons, loss_hair_s, loss_lips, loss_lipscolor, loss_eyebrow  = self.calc_loss(preds, labels)
        self.lambda_loss = loss_lambda 
       
        acc_hair_s, acc_lips, acc_lipscolor, acc_eyebrow  = self.calc_acc(preds, labels)
        self.acc_val = (acc_hair_s + acc_lips + acc_lipscolor + acc_eyebrow) / 4.0
    

        tf.summary.scalar("val/loss_lambda", loss_lambda, collections=[self.opt.val_collection])
        tf.summary.scalar("val/loss_hair_s", loss_hair_s, collections=[self.opt.val_collection])
        tf.summary.scalar("val/loss_lips", loss_lips, collections=[self.opt.val_collection])
        tf.summary.scalar("val/loss_lipscolor", loss_lipscolor, collections=[self.opt.val_collection])
        tf.summary.scalar("val/loss_eyebrow", loss_eyebrow, collections=[self.opt.val_collection])

        tf.summary.scalar("val/acc_hair_s", acc_hair_s, collections=[self.opt.val_collection])
        tf.summary.scalar("val/acc_lips", acc_lips, collections=[self.opt.val_collection])
        tf.summary.scalar("val/acc_lipscolor", acc_lips, collections=[self.opt.val_collection])
        tf.summary.scalar("val/acc_eyebrow", acc_eyebrow, collections=[self.opt.val_collection])
   
        tf.summary.scalar("val/loss_recons", loss_recons, collections=[self.opt.val_collection])

        tf.summary.scalar("val/loss", self.loss_val, collections=[self.opt.val_collection])
        tf.summary.scalar("val/acc", self.acc_val, collections=[self.opt.val_collection])
        #tf.summary.image("val/_inputs", images, collections=[self.opt.val_collection])
        #tf.summary.image("val/_inputs_crop", images_crop, collections=[self.opt.val_collection])
        tf.summary.image("val/_inputs", images, collections=[self.opt.val_collection])
        tf.summary.image("val/_inputs_crop", images_crop, collections=[self.opt.val_collection])
        tf.summary.image("val/recons", recons, collections=[self.opt.val_collection])



