import os,cv2
import sys
from six.moves import xrange
import tensorflow as tf
import json,glob


CROP = [100, 225, 700, 700]
R_SIZE = 286
SIZE = 64
do_crop = False 


class BaseDataLoader():
    def __init__(self, opt):
        self.args = opt
        self.dataset_size = None

    def _read_files(self, dataroot):
        f = open(dataroot, "r")
        imgs_path = []
        labels = []
        for line in f:
            tokens = line.split()
            imgs_path.append(os.path.join(self.args.path_prefix, tokens[0]))
            labels.append(int(tokens[1]))

        return imgs_path, labels


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
        input_queue = tf.train.slice_input_producer([imgs_path, labels_list], shuffle=True, capacity=1024)

        # Read examples from files in the filename queue.
        file_content = tf.read_file(input_queue[0])
        # Read JPEG or PNG or GIF image from file
        reshaped_image = tf.image.decode_jpeg(file_content, channels=args.num_channels)
        # Resize image to 256*256
        #reshaped_image = tf.image.resize_images(reshaped_image,  (args.load_size, args.load_size))
        #reshaped_image = self.tf_resize_image_keep_aspect(reshaped_image,  args.load_size)
        #reshaped_image = tf.image.crop_to_bounding_box(reshaped_image, CROP[0], CROP[1], CROP[2], CROP[3])
        new_h, new_w = SIZE, SIZE #self.resize_image_keep_aspect(imgs_path[0], args.load_size)
        reshaped_image = tf.image.resize_images(reshaped_image, (new_h, new_w))
        
        img_info = input_queue[0]
        labels = tf.cast(input_queue[1], tf.int64)

        reshaped_image = self.preprocess(reshaped_image, args)

        # Ensure that the random shuffling has good mixing properties.
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(self.dataset_size * min_fraction_of_examples_in_queue)
        # print(batch_size)
        print('Filling queue with %d images before starting to train. '
              'This may take some times.' % min_queue_examples)
        batch_size = args.batch_size

        # Load images and labels with additional info and return batches
        image_batch, label_batch,info = tf.train.batch(
                [reshaped_image, labels, img_info],
                batch_size=batch_size,
                num_threads=args.num_threads,
                capacity=min_queue_examples + 3 * batch_size)
        # print(image_batch, (label_lambda_batch, label_hair_batch), info)
        return image_batch, label_batch, info

    def preprocess(self, reshaped_image, args):
        pass
 
        

class DataLoader(BaseDataLoader):
    def __init__(self, opt):
        super(DataLoader,self).__init__(opt)

    def random_rotate(self, image, label, max_angle=10, seed=123456):
        print("*****************random rotate******************")
        max_angle = max_angle
        angle = tf.random_uniform([], minval=-1, maxval=1, seed=seed) * max_angle
        angle = angle * 3.1415926 / 180
        image = tf.contrib.image.rotate(image, angle, interpolation="BILINEAR")
        label = tf.contrib.image.rotate(label, angle, interpolation="NEAREST")
        return image, label

    def random_rescale(self, image, label, min_scale=0.5, max_scale=2.0, seed=123456):
        if min_scale <= 0:
            raise ValueError('\'min_scale\' must be greater than 0.')
        elif max_scale <= 0:
            raise ValueError('\'max_scale\' must be greater than 0.')
        elif min_scale >= max_scale:
            raise ValueError('\'max_scale\' must be greater than \'min_scale\'.')

        shape = tf.shape(image)
        height = tf.to_float(shape[0])
        width = tf.to_float(shape[1])
        scale = tf.random_uniform(
            [], minval=min_scale, maxval=max_scale, dtype=tf.float32, seed=seed)
        new_height = tf.to_int32(height * scale)
        new_width = tf.to_int32(width * scale)
        image = tf.image.resize_images(image, [new_height, new_width],
                                       method=tf.image.ResizeMethod.BILINEAR)
        # Since label classes are integers, nearest neighbor need to be used.
        label = tf.image.resize_images(label, [new_height, new_width],
                                       method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        return image, label

    def random_crop_or_pad_image_and_label(self, image, label, crop_height, crop_width, seed=123456):

        label = tf.to_float(label)
        image_height = tf.shape(image)[0]
        image_width = tf.shape(image)[1]
        image_and_label = tf.concat([image, label], axis=2)
        image_and_label_pad = tf.image.pad_to_bounding_box(
            image_and_label, 0, 0,
            tf.maximum(crop_height, image_height),
            tf.maximum(crop_width, image_width))
        image_and_label_crop = tf.random_crop(
            image_and_label_pad, [crop_height, crop_width, 4], seed=seed)

        image_crop = image_and_label_crop[:, :, :3]
        label_crop = image_and_label_crop[:, :, 3:]
        label_crop = tf.cast(label_crop, tf.uint8)

        return image_crop, label_crop

    def preprocess(self, reshaped_image, args):    

        if do_crop:
            print("************************crop******************")
            reshaped_image = tf.image.resize_images(reshaped_image, (R_SIZE, R_SIZE))
            reshaped_image = tf.random_crop(reshaped_image, [SIZE, SIZE, args.num_channels])

            # # Randomly flip the image horizontally.
            reshaped_image = tf.image.random_flip_left_right(reshaped_image)

        reshaped_image = reshaped_image / 127.5 - 1.0

        return reshaped_image


class DataLoader_Val(BaseDataLoader):
    def __init__(self, opt):
        super(DataLoader_Val, self).__init__(opt)

    def preprocess(self, reshaped_image, args):  

        reshaped_image = reshaped_image / 127.5 - 1.0

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
