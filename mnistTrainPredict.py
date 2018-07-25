import tensorflow as tf
from tensorflow.examples.tutorials.mnist  import input_data
import cv2

INPUT_NODE = 784
OUTPUT_NODE  = 10

LAYER1_NODE  = 500
BATCH_SIZE = 100

LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY  = 0.99

REGULARIZATION_RATE  =  0.0001
TRAINING_STEPTS = 5000
MOVING_AVERAGE_DECAY = 0.99

def inference(input_tensor,avg_class,weights1,biases1,weights2,biases2):
    if avg_class==None:
       layer1 = tf.nn.relu(tf.matmul(input_tensor,weights1)+biases1)
       return tf.matmul(layer1,weights2)+biases2
    else:
        layer1 = tf.nn.relu(tf.matmul(input_tensor,avg_class.average(weights1))+avg_class.average(biases1))
        return tf.matmul(layer1,avg_class.average(weights2))+avg_class.average(biases2)

def train(mnist):
    x = tf.placeholder(tf.float32,[None,INPUT_NODE],name="x-input")
    y_ = tf.placeholder(tf.float32,[None,OUTPUT_NODE],name="y-input")

    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE,LAYER1_NODE],stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1,shape=[LAYER1_NODE]))

    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE,OUTPUT_NODE],stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1,shape=[OUTPUT_NODE]))

    y = inference(x,None,weights1,biases1,weights2,biases2)

    global_step = tf.Variable(0,trainable=False)

    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    average_y = inference(x,variable_averages,weights1,biases1,weights2,biases2)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=average_y,labels=tf.argmax(y_,1))

    cross_entropy_mean =  tf.reduce_mean(cross_entropy)
    # cross_entropy_mean1 =  tf.reduce_mean(cross_entropy1)

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    regularization = regularizer(weights1)+regularizer(weights2)

    loss =  cross_entropy_mean+regularization
    # loss1 =  cross_entropy_mean1+regularization

    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,mnist.train.num_examples/BATCH_SIZE,LEARNING_RATE_DECAY)

    train_stept = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    # train_stept1 = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss1,global_step=global_step)

    with tf.control_dependencies([train_stept,variable_averages_op]):
        train_op = tf.no_op(name="train")

    correct_prediction = tf.equal(tf.argmax(average_y,1),tf.argmax(y_,1))
    # correct_prediction1 = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction ,tf.float32))
    # accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1 ,tf.float32))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        validate_feed = {x:mnist.validation.images,y_:mnist.validation.labels}
        test_feed = {x:mnist.test.images,y_:mnist.test.labels}

        ckpt = tf.train.get_checkpoint_state(r"D:\workspace\cifar10Demo\models\mnist")
        if ckpt and ckpt.model_checkpoint_path:
            print("ckpt.mode_checkpoint_path ",ckpt.model_checkpoint_path)
            saver.restore(sess,ckpt.model_checkpoint_path)
        else:
            for i in range(TRAINING_STEPTS):
                if i%100 ==0:
                    validate_acc =  sess.run(accuracy,feed_dict=validate_feed)
                    print("after %d step,validation accuracy using average  model is %g "%(i,validate_acc))
                    # validate_acc1 =  sess.run(accuracy1,feed_dict=validate_feed)
                    # print("after %d step,validation accuracy1 *** using average  model is %g "%(i,validate_acc1))

                xs,ys = mnist.train.next_batch(BATCH_SIZE)
                sess.run(train_op,feed_dict={x:xs,y_:ys})
                # print("after steps,test accuracy using average model is ",(sess.run(y,feed_dict={x:xs[:2]})))
            saver.save(sess,r"D:\workspace\cifar10Demo\models\mnist\mnist.chpt")

        test_acc = sess.run(accuracy,feed_dict=test_feed)
        print("after %d steps,test accuracy using average model is %g "%(TRAINING_STEPTS,test_acc))
        # test_acc1 = sess.run(accuracy1,feed_dict=test_feed)
        # print("after %d steps,test accuracy1 *** using average model is %g "%(TRAINING_STEPTS,test_acc1))

        image = cv2.imread(r"D:\workspace\cifar10Demo\data\mnist\7.png")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # ret,img = cv2.threshold(image, 125, 1, cv2.THRESH_BINARY)
        img = image.reshape([1,784])
        img=1-img/255
        # print("*"*8,img)
        predict_list = sess.run(y,feed_dict={x:img})
        print("predict is  ",sess.run(tf.argmax(predict_list,1)))

def main(argv=None):
    mnist = input_data.read_data_sets("D:/workspace/cifar10Demo/data/mnist",one_hot=True)
    train(mnist)

if __name__=="__main__":
    tf.app.run()