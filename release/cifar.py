import tensorflow as tf
import re
import os
import sys
import tarfile
from six.moves import urllib
from release import imgInput


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', 20,
                            """Number of images to process in a batch.""")
# tf.app.flags.DEFINE_string('data_dir', 'E:/PWORKSPACE\houseUtil\resized',
#                            """Path to the data directory.""")

tf.app.flags.DEFINE_string('filename', 'E:/PWORKSPACE/houseUtil/resized/train.tfc',
                           """record data file name.""")

tf.app.flags.DEFINE_string('eval_filename', 'E:/PWORKSPACE/houseUtil/resized/eval.tfc',
                           """record data file name.""")

NUM_CLASSES = 3
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 10000

MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 40.0  # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.01  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.01  # Initial learning rate.

def _activation_summary(x):
    tf.summary.histogram("/activations",x)
    tf.summary.scalar("/sparsity",tf.nn.zero_fraction(x))

def _variable_on_cpu(name,shape,initializer):
    with tf.device("/cpu:0"):
        var = tf.get_variable(name,shape,initializer=initializer)
    return var

def _variable_with_weight_decay(name,shape,stddev,wd):
    dtype = tf.float32
    var = _variable_on_cpu(name,shape,tf.truncated_normal_initializer(stddev=stddev,dtype=dtype))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var),wd,name="weight_loss")
        tf.add_to_collection("loss",weight_decay)
    return var

def inputs():
    # data_dir = os.path.join(FLAGS.data_dir, '')
    # images, labels = imgInput.createBatch(FLAGS.filename,FLAGS.batch_size)
    images, labels = imgInput.createBatch(FLAGS.filename,FLAGS.batch_size)
    return images, labels

def one_input(filename,label):
    img = imgInput.one_read(filename)
    label = tf.convert_to_tensor(label)
    print("one_input--label:",label)
    label = tf.reshape(label,[1,])
    return img,label

def inference(images):
    with tf.variable_scope("conv1") as scope:
        kernel = _variable_with_weight_decay("weights",shape=[5,5,3,64],stddev=1e-4,wd=None)
        conv = tf.nn.conv2d(images,kernel,[1,1,1,1],padding="SAME")
        biases = _variable_on_cpu("biases",[64],tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv,biases)
        conv1 = tf.nn.leaky_relu(bias,name=scope.name)
        _activation_summary(conv1)

    pool1 = tf.nn.max_pool(conv1,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME",name="pool1")
    norm1 = tf.nn.lrn(pool1,4,bias=1.0,alpha=0.001/9.0,beta=0.75,name="norm1")

    with tf.variable_scope("conv2") as scope:
        kernel = _variable_with_weight_decay("weights",[5,5,64,64],stddev=1e-4,wd=None)
        conv = tf.nn.conv2d(norm1,kernel,[1,1,1,1],padding="SAME")
        biases = _variable_on_cpu("biases",[64],tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv,biases)
        conv2 = tf.nn.leaky_relu(bias,name=scope.name)
        _activation_summary(conv2)
    norm2 = tf.nn.lrn(conv2,4,bias=1.0,alpha=0.001/9.0,beta=0.75,name="norm2")
    pool2 = tf.nn.max_pool(norm2,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME",name="pool2")

    with tf.variable_scope("local3") as scope:
        reshape = tf.reshape(pool2, [images.get_shape().as_list()[0], -1])
        dim = reshape.get_shape()[1].value

        weights = _variable_with_weight_decay("weights",[dim,384],stddev=0.04,wd=0.004)
        biases  = _variable_on_cpu("biases",[384],tf.constant_initializer(0.1))
        local3 = tf.nn.leaky_relu(tf.matmul(reshape,weights)+biases,name=scope.name)
        _activation_summary(local3)

    with tf.variable_scope("local4") as scope:
        weights = _variable_with_weight_decay("weights",[384,192],stddev=0.04,wd=0.004)
        biases = _variable_on_cpu("biases",[192],tf.constant_initializer(0.1))
        local4 = tf.nn.leaky_relu(tf.matmul(local3,weights) + biases,name=scope.name)
        _activation_summary(local4)

    with tf.variable_scope("softmax_linear") as scope:
        weights = _variable_with_weight_decay("weights",[192,NUM_CLASSES],stddev=1/192.0,wd=None)
        biases = _variable_on_cpu("biases",[NUM_CLASSES],tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(local4,weights),biases,name=scope.name)
        _activation_summary(softmax_linear)
    return softmax_linear

def loss(logits,labels):
    # sparse_labels = tf.reshape(labels,[FLAGS.batch_size,1])
    # indices = tf.reshape(tf.range(FLAGS.batch_size),[FLAGS.batch_size,1])
    # concated = tf.concat([indices,sparse_labels],1)
    labels = tf.cast(labels,tf.int64)
    print("loss : ",labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits,name="cross_entropy_example")
    cross_entropy_mean = tf.reduce_mean(cross_entropy,name="cross_entry")
    tf.add_to_collection("loss",cross_entropy_mean)

    return tf.add_n(tf.get_collection("loss"),name="total_loss")

def _add_loss_summaries(total_loss):
    loss_averages = tf.train.ExponentialMovingAverage(0.9,name="avg")
    losses = tf.get_collection("loss")
    loss_averages_op = loss_averages.apply(losses + [total_loss])
    for l in losses + [total_loss]:
        tf.summary.scalar(l.op.name + " (raw)",l)
        tf.summary.scalar(l.op.name,loss_averages.average(l))
    return loss_averages_op

def train(total_loss,global_step):
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.summary.scalar("learning_rate",lr)
    loss_averages_op = _add_loss_summaries(total_loss)

    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)
    apply_gradient_op = opt.apply_gradients(grads,global_step=global_step)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name,var)

    for grad,var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + "/gradients",grad)

    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    with tf.control_dependencies([apply_gradient_op]):
        variable_averages_op = variable_averages.apply(tf.trainable_variables())

    return variable_averages_op
