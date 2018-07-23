import os,cv2
import sys
from six.moves import xrange
import tensorflow as tf
import glob


# CROP = [100, 225, 700, 700]
R_SIZE = 600
SIZE = 512
do_crop =  True


class BaseDataLoader():
    def __init__(self, opt):
        self.args = opt
        self.dataset_size = None

    def _read_files(self, dataroot):
        imgs_path = glob.glob(os.path.join(dataroot, 'images', "*.jpg"))
        labels_path = glob.glob(os.path.join(dataroot, 'labels', "*.png"))
        # print(imgs_path, labels_path)

        assert len(imgs_path) == len(labels_path), "labels  imgs num not equal"

        return imgs_path, labels_path


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

        input_queue = tf.train.slice_input_producer([imgs_path,labels_list], shuffle=True, capacity=1024)

        # Read examples from files in the filename queue.
        file_content = tf.read_file(input_queue[0])
        # Read JPEG or PNG or GIF image from file
        reshaped_image = tf.image.decode_jpeg(file_content, channels=args.num_channels)


        # Read labels from files in the filename queue.
        file_content = tf.read_file(input_queue[1])
        # Read JPEG or PNG or GIF image from file
        label_image = tf.image.decode_png(file_content, channels=1)

        new_h, new_w = SIZE, SIZE #self.resize_image_keep_aspect(imgs_path[0], args.load_size)

        reshaped_image = tf.image.resize_images(reshaped_image, (new_h, new_w))
        label_image = tf.image.resize_images(label_image, (new_h, new_w))

        img_info = input_queue[0]

        reshaped_image, label_image = self.preprocess(reshaped_image, label_image, args)

        # Ensure that the random shuffling has good mixing properties.
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(self.dataset_size * min_fraction_of_examples_in_queue)
        # print(batch_size)
        print('Filling queue with %d images before starting to train. '
              'This may take some times.' % min_queue_examples)
        batch_size = args.batch_size

        # Load images and labels with additional info and return batches
        image_batch, label_batch, info = tf.train.batch(
                [reshaped_image, label_image, img_info],
                batch_size=batch_size,
                num_threads=args.num_threads,
                capacity=min_queue_examples + 3 * batch_size)
        #print(image_batch, label_batch, info)
        return image_batch, label_batch, info

    def preprocess(self, reshaped_image, args):
        pass
 
        

class DataLoader(BaseDataLoader):
    def __init__(self, opt):
        super(DataLoader,self).__init__(opt)

    def preprocess(self, reshaped_image, label_image, args):
        # Image processing for training the network. Note the many random
        # distortions applied to the image.

        # # Randomly crop a [height, width] section of the image.
        #reshaped_image = tf.random_crop(reshaped_image, [CROP_SIZE, CROP_SIZE, args.num_channels])
        if do_crop:
            print("************************crop******************")
            # # Randomly crop a [height, width] section of the image.
            reshaped_image = tf.image.resize_images(reshaped_image, (R_SIZE, R_SIZE))
            reshaped_image = tf.random_crop(reshaped_image, [SIZE, SIZE, args.num_channels], seed = 1234)

            # # Randomly flip the image horizontally.
            reshaped_image = tf.image.random_flip_left_right(reshaped_image, seed = 789)

            # # Randomly crop a [height, width] section of the image.
            label_image = tf.image.resize_images(label_image, (R_SIZE, R_SIZE))
            label_image = tf.random_crop(label_image, [SIZE, SIZE, 1], seed=1234)

            # # Randomly flip the image horizontally.
            label_image = tf.image.random_flip_left_right(label_image, seed=789)

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
        # label_image = tf.image.convert_image_dtype(label_image, dtype=tf.float32) / 255.0

        return reshaped_image, label_image


class DataLoader_Val(BaseDataLoader):
    def __init__(self, opt):
        super(DataLoader_Val, self).__init__(opt)
    
    def preprocess(self, reshaped_image, label_image, args):
        # # Image processing for evaluation.
        # # Crop the central [height, width] of the image.
        #reshaped_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,CROP_SIZE, CROP_SIZE)

        # # Subtract off the mean and divide by the variance of the pixels.
        # float_image = tf.image.per_image_standardization(resized_image)

        # # Set the shapes of tensors.
        # float_image.set_shape([args.crop_size, args.crop_size, args.num_channels])
        reshaped_image = tf.image.convert_image_dtype(reshaped_image, dtype=tf.float32) / 255.0
        # label_image = tf.image.convert_image_dtype(label_image, dtype=tf.float32) / 255.0

        return reshaped_image, label_image

    
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
