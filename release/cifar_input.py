import tensorflow as tf
import os


IMAGE_SIZE = 24
NUM_CLASSES = 10
BATCH_TRAIN = 50000
BATCH_EVAL = 10000

def read_cifar10(filename_queue):
    class CIFAR10Record(object):
        pass
    result = CIFAR10Record()
    label_bytes = 1
    result.height = 32
    result.width = 32
    result.dept = 3
    image_bytes = result.height * result.width * result.dept
    record_bytes = label_bytes + image_bytes

    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key,value = reader.read(filename_queue)

    record_bytes = tf.decode_raw(value,tf.uint8)
    # result.label = tf.cast(tf.strided_slice(record_bytes,[0],[label_bytes]),tf.int32)
    result.label = tf.cast(tf.slice(record_bytes,[0],[label_bytes]),tf.int32)
    # dept_major = tf.reshape(tf.strided_slice(record_bytes,[label_bytes],
    #                                          [label_bytes + image_bytes]),
    #                             [result.dept,result.height,result.width])
    dept_major = tf.reshape(tf.slice(record_bytes, [label_bytes],
                                     [image_bytes]),
                            [result.dept, result.height, result.width])

    result.uint8image = tf.transpose(dept_major,[1,2,0])
    print(result.label,result.uint8image)
    return result

def _generate_image_and_label_batch(image,label,min_queue_examples,batch_size,shuffle):
    num_preprocess_threads = 8
    if shuffle:
        images,label_batch = tf.train.shuffle_batch([image,label],
                                                    batch_size=batch_size,
                                                    num_threads=num_preprocess_threads,
                                                    capacity=min_queue_examples + 3 * batch_size,
                                                    min_after_dequeue=min_queue_examples)
    else:
        images,label_batch = tf.train.batch([image,label],
                                            batch_size=batch_size,
                                            num_threads=num_preprocess_threads,
                                            capacity=min_queue_examples)
    tf.summary.image("images",images)
    return images,tf.reshape(label_batch,[batch_size])

def distorted_inputs(data_dir,batch_size):
    filenames = [os.path.join(data_dir,"data_batch_%d.bin" % i) for i in range(1,6)]
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError("Failed to find file: "+f)

    filename_queue = tf.train.string_input_producer(filenames)
    read_input = read_cifar10(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image,tf.float32)
    height = IMAGE_SIZE
    width = IMAGE_SIZE
    distorted_image = tf.random_crop(reshaped_image,[height,width,3])
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    distorted_image = tf.image.random_brightness(distorted_image,max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image,lower=0.2,upper=1.8)
    float_image = tf.image.per_image_standardization(distorted_image)

    float_image.set_shape([height,width,3])
    read_input.label.set_shape([1])

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(BATCH_TRAIN * min_fraction_of_examples_in_queue)

    print("Filling queue with %d CIFAR images before starting to train. This will take a few minutes." % min_queue_examples)
    print("distorted_inputs ",read_input.label,read_input.label.get_shape())
    return _generate_image_and_label_batch(float_image,read_input.label,min_queue_examples,batch_size,shuffle=True)

def inputs(eval_data,data_dir,batch_size):
    if not eval_data:
        filenames = [os.path.join(data_dir,"data_batch_%d.bin" % i) for i in range(1,6)]
        num_examples_per_epoch = BATCH_TRAIN
    else:
        filenames = [os.path.join(data_dir,"test_batch.bin")]
        num_examples_per_epoch = BATCH_EVAL

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError("Failed to find file : "+f)

    filename_queue = tf.train.string_input_producer(filenames)

    read_input = read_cifar10(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image,tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,width,height)
    float_image = tf.image.per_image_standardization(resized_image)

    float_image.set_shape([height, width, 3])
    read_input.label.set_shape([1])

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch*min_fraction_of_examples_in_queue)
    return _generate_image_and_label_batch(float_image,read_input.label,min_queue_examples,batch_size,shuffle=False)


