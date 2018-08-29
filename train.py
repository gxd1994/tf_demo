import tensorflow as tf
import util,os
from options import TrainOptions
from data_loader import DataLoader, DataLoader_Val
from tensorflow.contrib.slim.nets import resnet_v2




def calc_size(batch_size):
    if batch_size == 16:
        return 4,4
    elif batch_size == 32:
        return 4,8
    elif batch_size == 64:
        return 8,8
    elif batch_size == 8:
        return 2,4
    elif batch_size == 1:
        return 1,1
    else:
        print("no win size according to batch_size")
        raise NotImplementedError



def restore_model(sess, t_vars, opt):
    global_step_val = 0
    # restore_saver = tf.train.Saver(var_list=t_vars)
    vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    # g_vars =  [var for var in vars_list if "generator" in var.op.name]
    print("restore vars", vars_list)
    restore_saver = tf.train.Saver(vars_list)
    #restore_saver = tf.train.Saver()
    restore_path = opt.restore_spec_model if  opt.restore_spec_model else tf.train.latest_checkpoint(
        opt.checkpoints_dir)
    print("restore_path", restore_path, opt.checkpoints_dir)

    if restore_path:
        global_step_val = int(restore_path.split("-")[1].split(".")[0])
        restore_saver.restore(sess, restore_path)
        print("restore form : step:%d" % (global_step_val))
    else:
        print("restore fail ")

    return global_step_val


def train(opt):
    # pass
    graph = tf.Graph()
    with graph.as_default():
        global_step = tf.placeholder_with_default(0,[],name="global_step")
        is_validation = tf.placeholder(dtype=tf.bool, shape=[], name="is_validation")
        is_training = tf.placeholder(dtype=tf.bool, shape=[], name="is_training")
        dropout_rate = tf.placeholder(dtype=tf.float32, shape=[], name="dropout_rate")

        dataloader = DataLoader(opt)
        dataloader_val = DataLoader_Val(opt)
        imgs_train, labels_train, infos_train = dataloader.read_inputs(opt.dataroot)
        imgs_val, labels_val, infos_val = dataloader_val.read_inputs(opt.dataroot_val)
        images_tensor, labels_tensor, infos_tensor = tf.cond(is_validation, \
                                                             lambda: (imgs_val, labels_val, infos_val), \
                                                             lambda: (imgs_train, labels_train, infos_train))


        model = util.parse_attr(opt.model)(opt)
        model.train(global_step, images_tensor, labels_tensor, is_training, dropout_rate, dataloader.dataset_size))
        t_vars = tf.trainable_variables()
        #print("t_vars", t_vars)
        summary_train_op = tf.summary.merge_all(opt.train_collection)
        summary_val_op = tf.summary.merge_all(opt.val_collection)

    with tf.Session(graph=graph) as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        global_step_val = restore_model(sess, t_vars, opt)

        #saver = tf.train.Saver(var_list=t_vars)
        saver = tf.train.Saver()
        train_writer = tf.summary.FileWriter(opt.checkpoints_dir, graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            while not coord.should_stop():
                start_epoch = global_step_val // max(1, dataloader.dataset_size // opt.batch_size) + 1
                for epoch in range(start_epoch, opt.epochs + 1):
                    for i in range(max(1,dataloader.dataset_size // opt.batch_size)):
                        global_step_val += 1
                        # train
                        _, loss_val, acc_val, summary_train = sess.run(
                                [model.train_op, model.loss, model.acc, summary_train_op],
                                feed_dict={is_validation: False, global_step:global_step_val,
                                             is_training: True, dropout_rate: 0.5})
                        if global_step_val % opt.print_freq == 0:
                            print('-----------Step %d:-------------' % global_step_val)
                            print("Epoch:{},  loss:{},  acc:{}".format(epoch, loss_val, acc_val))

                        if global_step_val % opt.display_freq == 0:
                            train_writer.add_summary(summary_train, global_step_val)
                            train_writer.flush()
                        # validation
                        if global_step_val % opt.eval_freq == 0:
                            eval_loss_val_sum = 0
                            eval_acc_val_sum = 0
                            val_count = max(1, dataloader_val.dataset_size // opt.batch_size)
                            for j in range(val_count):
                                eval_loss_val, eval_acc_val, eval_summary_val = sess.run(
                                                [model.loss_val, model.acc_val, summary_val_op],
                                                feed_dict={is_validation: True,
                                                           is_training: False, dropout_rate: 1.0})
                                eval_loss_val_sum += eval_loss_val
                                eval_acc_val_sum += eval_acc_val
                                print('----------- Step %d:-------------' % global_step_val)
                                print("EVAL Batch, total count:{}  cur:{}  Epoch:{},  loss:{},  acc:{}".format(val_count, j, epoch, eval_loss_val, eval_acc_val))

                                # util.save_images(sample_imgs, calc_size(opt.batch_size),
                                #                  os.path.join(opt.save_results_path,
                                                              "epoch_%d_count_%d_val_%d.jpg" % (epoch, global_step_val, j)))

                                train_writer.add_summary(eval_summary_val, global_step_val)
                                train_writer.flush()
                            # val average
                            print('----------- Step %d:-------------' % global_step_val)
                            print("EVAL ALL:   Epoch:{},  loss:{},  acc:{}".format(epoch, eval_loss_val_sum / val_count , eval_acc_val_sum / val_count))

                    if epoch % opt.save_epoch_freq == 0:
                        save_path = saver.save(sess, opt.checkpoints_dir + "/model.ckpt", global_step=global_step_val)
                        print("Model saved in file: %s" % save_path)

                print("achieve maximum epoch")
                coord.request_stop()

        except KeyboardInterrupt:
            print('Interrupted')
            coord.request_stop()
        except Exception as e:
            coord.request_stop(e)
        finally:
            save_path = saver.save(sess, opt.checkpoints_dir + "/model.ckpt", global_step=global_step_val)
            print("Model saved in file: %s" % save_path)
            # When done, ask the threads to stop.
            coord.request_stop()
            coord.join(threads)




def main():
    opt = TrainOptions().parse()
    train(opt)


if __name__ == '__main__':
    main()
