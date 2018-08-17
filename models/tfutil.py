def generator(self, z, is_training, norm_type, name, reuse):
    self.gf_dim = 64
    s_h, s_w = self.opt.load_size, self.opt.load_size
    assert s_h == 256
    kernel_initializer = tf.truncated_normal_initializer(stddev=0.02)

    with tf.variable_scope(name, reuse=reuse):
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

        # project `z` and reshape
        z1 = tf.layers.dense(z, self.gf_dim * 8 * s_h16 * s_w16, activation=None, kernel_initializer=kernel_initializer,
                             name="linear")

        network = tf.reshape(z1, [-1, s_h16, s_w16, self.gf_dim * 8])
        network = self.norm(network, is_training=is_training, norm=norm_type)
        network = tf.nn.relu(network)

        # deconv
        network = tf.layers.conv2d_transpose(network, self.gf_dim * 4, 5, (2, 2), padding="SAME",
                                             kernel_initializer=kernel_initializer)
        network = self.norm(network, is_training=is_training, norm=norm_type)
        network = tf.nn.relu(network)

        network = tf.layers.conv2d_transpose(network, self.gf_dim * 2, 5, (2, 2), padding="SAME",
                                             kernel_initializer=kernel_initializer)
        network = self.norm(network, is_training=is_training, norm=norm_type)
        network = tf.nn.relu(network)

        network = tf.layers.conv2d_transpose(network, self.gf_dim, 5, (2, 2), padding="SAME",
                                             kernel_initializer=kernel_initializer)
        network = self.norm(network, is_training=is_training, norm=norm_type)
        network = tf.nn.relu(network)

        network = tf.layers.conv2d_transpose(network, self.gf_dim // 2, 5, (2, 2), padding="SAME",
                                             kernel_initializer=kernel_initializer)
        network = self.norm(network, is_training=is_training, norm=norm_type)
        network = tf.nn.relu(network)

        network = tf.layers.conv2d(network, self.opt.num_channels, 5, (1, 1), padding="SAME",
                                   kernel_initializer=kernel_initializer)
        print("generator  output", network)

        return tf.nn.tanh(network)


def discriminator(self, inputs, is_training, norm_type, name, reuse):
    kernel_initializer = tf.truncated_normal_initializer(stddev=0.02)
    with tf.variable_scope(name, reuse=reuse):
        network = tf.layers.conv2d(inputs, 64, 7, (2, 2), activation=None, padding="SAME",
                                   kernel_initializer=kernel_initializer)
        # network = self.norm(network, is_training=is_training, norm=norm_type)
        network = tf.nn.leaky_relu(network)

        network = tf.layers.conv2d(network, 128, 4, (2, 2), activation=None, padding="SAME",
                                   kernel_initializer=kernel_initializer)
        network = self.norm(network, is_training=is_training, norm=norm_type)
        network = tf.nn.leaky_relu(network)

        network = tf.layers.conv2d(network, 256, 4, (2, 2), activation=None, padding="SAME",
                                   kernel_initializer=kernel_initializer)
        network = self.norm(network, is_training=is_training, norm=norm_type)
        network = tf.nn.leaky_relu(network)

        network = tf.layers.conv2d(network, 512, 4, (2, 2), activation=None, padding="SAME",
                                   kernel_initializer=kernel_initializer)
        network = self.norm(network, is_training=is_training, norm=norm_type)
        network = tf.nn.leaky_relu(network)

        network = tf.layers.conv2d(network, 1, 4, (1, 1), activation=None, padding="SAME",
                                   kernel_initializer=kernel_initializer)
        print("discriminator  output", network)
        return network


def encoder(self, inputs, is_training, norm_type, dropout_rate, name, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        # The VGG architecture, the default type is 'A'
        wd = 0.0005
        model_type = 'a'
        print("model_type:", model_type)
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
            config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512,
                      'M']
        else:
            print('Unknown model type: ' + model_type + ' | Please specify a modelType A or B or D or E')

        network = inputs
        # kernel_initializer = tf.contrib.layers.xavier_initializer()  #variance_scaling_initializer()
        kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
        with tf.variable_scope('convs_comm'):
            for k, v in enumerate(config):
                if v == 'M':
                    network = tf.layers.max_pooling2d(network, 2, 2, padding="SAME")
                else:
                    with tf.variable_scope('conv' + str(k)):
                        network = tf.layers.conv2d(network, v, 3, activation=None, padding="SAME",
                                                   kernel_initializer=kernel_initializer)
                        network = self.norm(network, is_training=is_training, norm=norm_type)
                        network = tf.nn.relu(network)

            network_comm_map = network
            network_comm_map_flatten = tf.layers.flatten(network_comm_map)

        return network_comm_map, network_comm_map_flatten


def decoder(self, inputs, is_training, norm_type, name, reuse):
    network = inputs
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
    with tf.variable_scope(name, reuse=reuse):
        config = [512, 256, 128, 64]
        for k, v in enumerate(config):
            with tf.variable_scope('deconv' + str(k)):
                network = tf.layers.conv2d_transpose(network, v, 3, (2, 2), padding="SAME")
                network = self.norm(network, is_training=is_training, norm=norm_type)
                network = tf.nn.relu(network)
        with tf.variable_scope('pred'):
            network = tf.layers.conv2d_transpose(network, self.opt.num_channels, 3, (2, 2), padding="SAME")
            # recons = tf.nn.relu(network)
            recons = tf.sigmoid(network)
            # recons = network
            # print("recons", recons)

    return recons


def estimator_param(self, inputs, is_training, norm_type, dropout_rate, name, reuse):
    network_comm_flatten = inputs
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
    with tf.variable_scope(name, reuse=reuse):
        with tf.variable_scope('fc1_cls'):
            network = tf.layers.dense(network_comm_flatten, 2048, activation=None,
                                      kernel_initializer=kernel_initializer)
            network = self.norm(network, is_training=is_training, norm=norm_type)
            network = tf.nn.relu(network)
            network = tf.nn.dropout(network, dropout_rate)
        with tf.variable_scope('fc2_cls'):
            network = tf.layers.dense(network, 256, activation=None, kernel_initializer=kernel_initializer)
            network = self.norm(network, is_training=is_training, norm=norm_type)
            network = tf.nn.relu(network)
            network = tf.nn.dropout(network, dropout_rate)
        with tf.variable_scope('output_cls'):
            preds_cls = tf.layers.dense(network, Total_Num_Cls, activation=None, kernel_initializer=kernel_initializer,
                                        use_bias=False)
            # preds_cls = tf.layers.dense(network, Total_Num_Cls - 22, activation=None, kernel_initializer = kernel_initializer, use_bias=False)

        # with tf.variable_scope('output_cls_hair'):
        #    network = tf.reduce_mean(network_comm_map, [1, 2], keep_dims=True, name="global_pooling")
        #    network = tf.nn.dropout(network, dropout_rate)
        #    network = tf.layers.conv2d(network, 22, 1, activation=None, padding="SAME", kernel_initializer=kernel_initializer)
        #    pred_hair_s = tf.squeeze(network, [1, 2], name="squeeze_hair_s")
        #
        # preds_cls = tf.concat([pred_hair_s, preds_cls], axis=1)

        # face lambda
        with tf.variable_scope('fc1_lambda'):
            network = tf.layers.dense(network_comm_flatten, 2048, activation=None,
                                      kernel_initializer=kernel_initializer)
            network = self.norm(network, is_training=is_training, norm=norm_type)
            network = tf.nn.relu(network)
            network = tf.nn.dropout(network, dropout_rate)
        with tf.variable_scope('fc2_lambda'):
            network = tf.layers.dense(network, 512, activation=None, kernel_initializer=kernel_initializer)
            network = self.norm(network, is_training=is_training, norm=norm_type)
            network = tf.nn.relu(network)
            network = tf.nn.dropout(network, dropout_rate)
        with tf.variable_scope('output_lambda'):
            preds_lambda = tf.layers.dense(network, self.opt.face_lambda_len, activation=None,
                                           kernel_initializer=kernel_initializer, use_bias=False)
            preds_lambda = tf.sigmoid(preds_lambda, name="sigmoid")
            # preds_lambda = tf.nn.tanh(preds_lambda, name="tanh")
            # preds_lambda = preds_lambda #tf.sigmoid(preds_lambda, name="sigmoid")

    return preds_lambda, preds_cls
