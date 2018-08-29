import tensorflow as tf
import util
from options import TrainOptions
from data_loader import DataLoader, DataLoader_Val

def restore_model(sess, t_vars, opt):
    global_step_val = 0
    restore_saver = tf.train.Saver(var_list=t_vars)
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
        #dataloader = DataLoader(opt)
        dataloader_val = DataLoader_Val(opt)
        #imgs_train, labels_train, infos_train = dataloader.read_inputs()
        imgs_val, labels_val, infos_val = dataloader_val.read_inputs()
        #images_tensor, labels_tensor, infos_tensor = tf.cond(is_validation, \
        #                                                     lambda: (imgs_val, labels_val, infos_val), \
        #                                                     lambda: (imgs_train, labels_train, infos_train))
        images_tensor, labels_tensor, infos_tensor =  imgs_val, labels_val, infos_val

         #tf.cond(tf.equal(tf.mod(global_step, opt.eval_freq), 0),

        model = util.parse_attr(opt.model)(opt)
        #model.train(global_step, images_tensor, labels_tensor)
        model.evalute(images_tensor, labels_tensor, reuse=False)
        t_vars = tf.trainable_variables()
        #summary_train_op = tf.summary.merge_all(opt.train_collection)
        summary_val_op = tf.summary.merge_all(opt.val_collection)

    with tf.Session(graph=graph) as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        global_step_val = restore_model(sess, t_vars, opt)

        saver = tf.train.Saver(var_list=t_vars)
        train_writer = tf.summary.FileWriter(opt.checkpoints_dir, graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            while not coord.should_stop():
                epoch = global_step_val // opt.batch_size + 1
                eval_loss_val_sum = 0
                eval_acc_val_sum = 0
                val_count = dataloader_val.dataset_size // opt.batch_size
                for j in range(val_count):
                    eval_loss_val, eval_acc_val, eval_summary_val = sess.run(
                                    [model.loss_val, model.acc_val, summary_val_op],
                                    feed_dict={is_validation: True})
                    eval_loss_val_sum += eval_loss_val
                    eval_acc_val_sum += eval_acc_val
                    print('----------- Step %d:-------------' % global_step_val)
                    print("EVAL Batch, total count:{}  cur:{}  Epoch:{},  loss:{},  acc:{}".format(val_count, j, epoch, eval_loss_val, eval_acc_val))

                    train_writer.add_summary(eval_summary_val, global_step_val)
                    train_writer.flush()
                # val average
                print('----------- Step %d:-------------' % global_step_val)
                print("EVAL ALL:   Epoch:{},  loss:{},  acc:{}".format(epoch, eval_loss_val_sum / val_count , eval_acc_val_sum / val_count))


                coord.request_stop()

        except KeyboardInterrupt:
            print('Interrupted')
            coord.request_stop()
        except Exception as e:
            coord.request_stop(e)
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
            coord.join(threads)




def main():
    opt = TrainOptions().parse()
    train(opt)


if __name__ == '__main__':
    main()
