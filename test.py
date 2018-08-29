import tensorflow as tf
import util
from options import TestOptions
import cv2, glob,os,json
import numpy as np
from tqdm import tqdm
from data_loader import Data_dict, Data_to_index_dict, Data_to_ID_dict 

JSON_TEMP = "./template_json/template.json"
ext = 'jpg'
def write_json(path, save_path, pred_val):
    preds_lambda, preds_hair_s, preds_lips, preds_lipscolor, preds_eyebrow = pred_val
    preds_hair_s = np.argmax(preds_hair_s, axis=0)
    preds_lips = np.argmax(preds_lips, axis=0)
    preds_lipscolor = np.argmax(preds_lipscolor, axis=0)
    preds_eyebrow = np.argmax(preds_eyebrow, axis=0)
 
    with open(path, "r") as fp:
        context = json.load(fp)
        face_lambda_raw = context["FaceControllerLambda"]
        face_lambda_pred = preds_lambda.tolist()        
        for i in range(len(Data_to_ID_dict["lambda"])):
            face_lambda_raw[Data_to_ID_dict["lambda"][i]] = face_lambda_pred[i] + 0.5
        
        context["FaceControllerLambda"] = face_lambda_raw

        context["HairStyleID"] = Data_to_ID_dict["HairStyleID"][int(preds_hair_s)]
        context["LipsID"] = Data_to_ID_dict["LipsID"][int(preds_lips)]
        context["LipsColorID"] = Data_to_ID_dict["LipsColorID"][int(preds_lipscolor)]
        context["EyebrowID"] = Data_to_ID_dict["EyebrowID"][int(preds_eyebrow)]
        # print(type(context))
        # print(type(face_lambda))
        # print(type(hair_style_id))
    
    with open(save_path, "w") as fp:
        json.dump(context, fp)    

def restore_model(sess, t_vars, opt):
    global_step_val = 0
    #restore_saver = tf.train.Saver(var_list=t_vars)
    restore_saver = tf.train.Saver()
    restore_path = opt.restore_spec_model if  opt.restore_spec_model else tf.train.latest_checkpoint(
        opt.checkpoints_dir)
    print("restore_path", restore_path, opt.restore_spec_model, opt.checkpoints_dir)

    if restore_path:
        global_step_val = int(restore_path.split("-")[1].split(".")[0])
        restore_saver.restore(sess, restore_path)
        print("restore form : step:%d" % (global_step_val))
    else:
        print("restore fail ")

    return global_step_val


def write_json_test(preds_val, save_path):
    preds_lambda, preds_hair_s, preds_lips, preds_lipscolor, preds_eyebrow = preds_val
    
    write_json(JSON_TEMP, save_path,(preds_lambda[0], preds_hair_s[0], preds_lips[0], preds_lipscolor[0], preds_eyebrow[0]))
 

def test(opt):
    # pass
    graph = tf.Graph()
    with graph.as_default():
        images_tensors = tf.placeholder(dtype=tf.float32, shape= [1, opt.load_size, opt.load_size, 3], name="images_ph")
        model = util.parse_attr(opt.model)(opt)
        model.test(images_tensors)
        t_vars = tf.trainable_variables()

    with tf.Session(graph=graph) as sess:
        restore_model(sess, t_vars, opt)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            while not coord.should_stop():
                for file_path in tqdm(glob.glob(os.path.join(opt.test_dir, "*.%s"%ext))):
                    file_name = os.path.basename(file_path)
                    save_path = os.path.join(opt.save_results_dir, file_name)
                    save_path = save_path.replace(".%s"%ext, ".json") 
                    
                    img = cv2.imread(file_path)
                    #img = (img - np.mean(img)) / np.std(img)
                    img = cv2.resize(img, (opt.load_size, opt.load_size))
                    img = img.astype(np.float32) / 255.0
                    preds_val, = sess.run([model.preds], feed_dict={images_tensors: img[np.newaxis, :, :, :]})
                    print('processing: %s' % file_path)
                    write_json_test(preds_val, save_path)
                
                coord.request_stop()

        except KeyboardInterrupt:
            print('Interrupted')
            coord.request_stop()
        except Exception as e:
            coord.request_stop(e)
        finally:
            coord.request_stop()
            coord.join(threads)

def main():
    opt = TestOptions().parse()
    save_root = opt.save_results_dir
    util.mkdir(save_root)
    test(opt)


if __name__ == '__main__':
    main()
