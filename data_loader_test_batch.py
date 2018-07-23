import os,cv2
import sys
from six.moves import xrange
import tensorflow as tf
import json,glob


SELECTED_INDEX = [0,1,9,10,18,19,27,28,37,46,55,63,64,72,73,82,99,109,127,144,162,171,172,226,234,243,252]
CROP = [100, 225, 700, 700]
SIZE = 224, 224
#SIZE = 256, 256
CROP_SIZE = 224


Data_dict = {"lambda": [0, 1, 9, 10, 18, 19, 27, 28, 37, 46, 55, 63, 64, 72, 73, 82, 99, 109, 127, 144, 162, 171, 172, 226, 234, 243, 252], 
"HairStyleID": [20001, 20003, 20004, 20005, 20006, 20007, 20008, 20009, 20010, 20011, 20012, 20013, 20014, 20016, 20017, 20018, 20019, 20027, 20028, 28000, 28002, 28003],"EyebrowID": [0, 20001, 20002, 20003, 20004, 20005, 20006, 20007, 20008, 20009, 20010, 20011, 20012, 20013, 20014, 20015, 20016, 20017, 20018, 20019, 20020, 20021, 20022,20023, 20024, 20025, 20026, 20027, 20028, 20029, 20030, 20031, 20032, 20033, 20034, 20035], 
"LipsID": [0, 20001, 20002, 20003, 20004, 20005, 20006, 20007, 20008, 20009, 20010, 20011, 20012, 20013, 20014, 20015, 20016, 20017, 20018], 
"LipsColorID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]}
Data_to_index_dict = {}
Data_to_ID_dict = {}
for k,v in Data_dict.items():
    d = {sub_v:sub_k for sub_k, sub_v in enumerate(v)}
    d_invese = {sub_k:sub_v for sub_k, sub_v in enumerate(v)}
    Data_to_index_dict[k] = d
    Data_to_ID_dict[k] = d_invese

print(Data_to_index_dict)
print(Data_to_ID_dict)


class BaseDataLoader():
    def __init__(self, opt):
        self.args = opt
        self.dataset_size = None

    def _read_files(self, dataroot):
        ext = "jpg"
        labels_json = glob.glob(os.path.join(dataroot,"*.json"))
        #labels_json.remove(os.path.join(dataroot,"temp.json")[0])
        #len_jsons = len(labels_json)
        #print("labels_json len", len_jsons)
        #templet_path = glob.glob(os.path.join(dataroot,"temp.path")) * len_jsons
        #labels_json += templet_json
        #print("labels_json len after", len(labels_json))
        
        # print("read labels_json", labels_json)
        imgs_path = []
        for file in labels_json:
            img_path = file.replace("json", ext)
            assert os.path.exists(img_path), img_path
            imgs_path.append(img_path)
        # print("read imgs_path", imgs_path)
        face_lambda_list = []
        hair_style_list = []
        lipsID_list = []
        lipscolorID_list = []
        eyebrowID_list = []
        for file in labels_json:
            with open(file, "r") as fp:
                data_dict = json.load(fp)
            #face lambda
            face_lambda_all = data_dict["FaceControllerLambda"]
            selected_index = Data_dict["lambda"] 
            #print("selected_index", selected_index)
            face_lambda_selected = [face_lambda_all[e] for e in selected_index]
            face_lambda_list.append(face_lambda_selected)
            
            #hair style ID
            hair_style_list.append(Data_to_index_dict["HairStyleID"][data_dict["HairStyleID"]])
            #print("hair style", hair_style_list)
            
            #lipsID
            lipsID_list.append(Data_to_index_dict["LipsID"][data_dict["LipsID"]])
            #print("lipsid", lipsID_list)
            
            #lipscolorID
            lipscolorID_list.append(Data_to_index_dict["LipsColorID"][data_dict["LipsColorID"]])
            #print("lipscolorID_list style", lipscolorID_list)
            
            #eyecolorID
            eyebrowID_list.append(Data_to_index_dict["EyebrowID"][data_dict["EyebrowID"]])
            #print("eyebrowid", eyebrowID_list)
            
        labels_list = [face_lambda_list, hair_style_list, lipsID_list, lipscolorID_list, eyebrowID_list]
        #print(labels_list)
        return imgs_path, labels_list


    #def tf_resize_image_keep_aspect(self, image, lo_dim):
    #    # Take width/height
    #    initial_width = tf.shape(image)[0]
    #    initial_height = tf.shape(image)[1]

    #    # Take the greater value, and use it for the ratio
    #    #min_ = tf.minimum(initial_width, initial_height)
    #    min_ = tf.maximum(initial_width, initial_height)
    #    ratio = tf.to_float(min_) / tf.constant(lo_dim, dtype=tf.float32)

    #    new_width = tf.to_int32(tf.to_float(initial_width) / ratio)
    #    new_height = tf.to_int32(tf.to_float(initial_height) / ratio)
    #    print(new_width, new_height)

    #    return tf.image.resize_images(image, [new_width, new_height])

    def tf_resize_image_keep_aspect(self, image, lo_dim):
        # Take width/height
        print(image.get_shape().as_list())
        initial_width = image.get_shape().as_list()[0]
        initial_height = image.get_shape().as_list()[1]

        # Take the greater value, and use it for the ratio
        #min_ = tf.minimum(initial_width, initial_height)
        min_ = max([initial_width, initial_height])
        ratio = 1.0 * min_ / lo_dim

        new_width = int(initial_width / ratio)
        new_height = int(initial_height / ratio)
        print(new_width, new_height)

        return tf.image.resize_images(image, [new_width, new_height])

    def resize_image_keep_aspect(self, img_path, lo_dim):
        # Take width/height
        img = cv2.imread(img_path)
        initial_height = img.shape[0]
        initial_width = img.shape[1]
        
        # Take the greater value, and use it for the ratio
        #min_ = tf.minimum(initial_width, initial_height)
        #min_ = tf.maximum(initial_width, initial_height)
        min_ = max([initial_width, initial_height]) 
        ratio = 1.0 * min_ / lo_dim
        
        new_width = int(1.0 * initial_width / ratio)
        new_height = int(1.0 * initial_height / ratio)
        
        print("new h,w", new_height, new_width)
        return new_height, new_width 
        #return new_width, new_height 


    def read_inputs(self, dataroot):
        args = self.args
        imgs_path, labels_list = self._read_files(dataroot)
        self.dataset_size = len(imgs_path)
        print("dataset size", self.dataset_size)

        # Create a queue that produces the filenames to read.

        input_queue = tf.train.slice_input_producer([imgs_path] + labels_list, shuffle=True, capacity=1024)

        # Read examples from files in the filename queue.
        file_content = tf.read_file(input_queue[0])
        # Read JPEG or PNG or GIF image from file
        reshaped_image = tf.image.decode_jpeg(file_content, channels=args.num_channels)
        # Resize image to 256*256
        #reshaped_image = tf.image.resize_images(reshaped_image,  (args.load_size, args.load_size))
        #reshaped_image = self.tf_resize_image_keep_aspect(reshaped_image,  args.load_size)
        reshaped_image = tf.image.crop_to_bounding_box(reshaped_image, CROP[0], CROP[1], CROP[2], CROP[3])
        new_h, new_w = SIZE #self.resize_image_keep_aspect(imgs_path[0], args.load_size)
        reshaped_image = tf.image.resize_images(reshaped_image, (new_h, new_w))
        
        img_info = input_queue[0]
        label_face_lambda = tf.cast(input_queue[1], tf.float32)
        label_hair_style = tf.cast(input_queue[2], tf.int64)
        label_lipsID = tf.cast(input_queue[3], tf.int64)
        label_lipscolorID = tf.cast(input_queue[4], tf.int64)
        label_eyebrowID = tf.cast(input_queue[5], tf.int64)

        reshaped_image = self.preprocess(reshaped_image, args)

        # Ensure that the random shuffling has good mixing properties.
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(self.dataset_size * min_fraction_of_examples_in_queue)
        # print(batch_size)
        print('Filling queue with %d images before starting to train. '
              'This may take some times.' % min_queue_examples)
        batch_size = args.batch_size

        # Load images and labels with additional info and return batches
        image_batch, label_lambda_batch, label_hair_batch, label_lipsID_batch,label_lipscolorID_batch,label_eyebrowID_batch,info = tf.train.batch(
                [reshaped_image, label_face_lambda, label_hair_style, label_lipsID, label_lipscolorID, label_eyebrowID, img_info],
                batch_size=batch_size,
                num_threads=args.num_threads,
                capacity=min_queue_examples + 3 * batch_size)
        # print(image_batch, (label_lambda_batch, label_hair_batch), info)
        return image_batch, (label_lambda_batch, label_hair_batch, label_lipsID_batch, label_lipscolorID_batch, label_eyebrowID_batch), info

    def preprocess(self, reshaped_image, args):
        pass
 
        

class DataLoader(BaseDataLoader):
    def __init__(self, opt):
        super(DataLoader,self).__init__(opt)

    def preprocess(self, reshaped_image, args):    
        # Image processing for training the network. Note the many random
        # distortions applied to the image.

        # # Randomly crop a [height, width] section of the image.
        #reshaped_image = tf.random_crop(reshaped_image, [CROP_SIZE, CROP_SIZE, args.num_channels])

        # # Randomly flip the image horizontally.
        # reshaped_image = tf.image.random_flip_left_right(reshaped_image)

        # # Because these operations are not commutative, consider randomizing
        # # the order their operation.
        # reshaped_image = tf.image.random_brightness(reshaped_image,
        #                                             max_delta=63)
        # # Randomly changing contrast of the image
        # reshaped_image = tf.image.random_contrast(reshaped_image,
        #                                           lower=0.2, upper=1.8)

        # # Subtract off the mean and divide by the variance of the pixels.
        # reshaped_image = tf.image.per_image_standardization(reshaped_image)

        # # Set the shapes of tensors.
        # reshaped_image.set_shape([args.crop_size, args.crop_size, args.num_channels])

        # # read_input.label.set_shape([1])

        reshaped_image = tf.image.convert_image_dtype(reshaped_image, dtype=tf.float32) / 255.0

        return reshaped_image


class DataLoader_Val(BaseDataLoader):
    def __init__(self, opt):
        super(DataLoader_Val, self).__init__(opt)
    
    def preprocess(self, reshaped_image, args):  
        # # Image processing for evaluation.
        # # Crop the central [height, width] of the image.
        #reshaped_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,CROP_SIZE, CROP_SIZE)

        # # Subtract off the mean and divide by the variance of the pixels.
        # float_image = tf.image.per_image_standardization(resized_image)

        # # Set the shapes of tensors.
        # float_image.set_shape([args.crop_size, args.crop_size, args.num_channels])
        reshaped_image = tf.image.convert_image_dtype(reshaped_image, dtype=tf.float32) / 255.0

        return reshaped_image

    
def main():
    from options import TrainOptions
    opt = TrainOptions().parse()
    dataloader = DataLoader(opt)
    dataloader_val = DataLoader_Val(opt)
    imgs_train, labels_train, infos_train = dataloader.read_inputs(opt.dataroot)
    imgs_val, labels_val, infos_val = dataloader_val.read_inputs(opt.dataroot_val)
    print(imgs_train, labels_train, infos_train,imgs_val,labels_val,infos_val)

if __name__ == '__main__':
    main()
