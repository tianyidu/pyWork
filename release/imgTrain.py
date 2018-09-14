from datetime import datetime
import time
import tensorflow as tf
from release import imgInference
from release import cifar_input
# from release import cifar
import cv2
from matplotlib import pyplot as plt
import os

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', 'E:/PWORKSPACE/houseUtil/ckpt',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 50,
                            """How often to log results to the console.""")

def train():

    with tf.Graph().as_default():
        input_image = tf.placeholder(dtype=tf.float32, shape=[None, 300, 300, 3], name="input_image")
        isEval = tf.placeholder(dtype=tf.bool, name="isEval")

        global_step = tf.train.get_or_create_global_step()
        with tf.device("/cpu:0"):
            # images,labels = cifar.inputs()
            images,labels = imgInference.inputs()
            # images,labels = cifar_input.distorted_inputs()
        # logits = cifar.inference(images)
        print("type : ",images)
        logits,conv = imgInference.inference(input_image,isEval)

        print("train:",labels,images,logits)
        # loss = cifar.loss(logits=logits, labels=labels)
        # train_op = cifar.train(loss, global_step)
        loss = imgInference.loss(logits=logits, labels=labels)
        train_op = imgInference.train(loss, global_step)

        class _LoggerHook(tf.train.SessionRunHook):
            def begin(self):
                self._step = -1
                self._start_time = time.time()

            def before_run(self,run_context):
                self._step += 1
                return tf.train.SessionRunArgs(loss,feed_dict={input_image:images,isEval:True})

            def after_run(self,run_context,  # pylint: disable=unused-argument
                            run_values):
                if self._step % FLAGS.log_frequency == 0:
                    current_time = time.time()
                    duration = current_time - self._start_time
                    self._start_time = current_time

                    # with tf.variable_scope("conv1_1",reuse=True) as scope:
                        # pic = run_context.session.run(conv)
                        # for i in range(100,105):
                        #     plt.imshow(pic[0,:,:,i])
                        #     plt.show()
                    loss_value = run_values.results
                    examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
                    sec_per_batch = float(duration / FLAGS.log_frequency)
                    format_str = ("%s: step %d, loss=%.2f (%.1f examples/sec; %.3f sec/batch）")
                    print(format_str % (datetime.now(),self._step,loss_value,examples_per_sec,sec_per_batch))

        with tf.train.MonitoredTrainingSession(checkpoint_dir=FLAGS.train_dir,
                                                   hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),tf.train.NanTensorHook(loss),_LoggerHook()],
                                                    config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)) as mon_sess:
            while not mon_sess.should_stop():
                # with tf.variable_scope("conv1_1", reuse=True) as scope:
                #     with tf.device("/cpu:0"):
                #         print("weights :", scope.name, mon_sess.run(tf.get_variable("weights",[3,3,3,32])),"=====",mon_sess.run(tf.get_variable("biases")))
                # print("logits ",mon_sess.run(logits))
                # print("loss ",mon_sess.run(loss))
                # print("conv ",mon_sess.run(conv))
                # images = mon_sess.run(images)
                mon_sess.run(train_op)
                # print("image_tmp :",label_tmp,label_tmp.shape,len(image_tmp),image_tmp.shape)
                # for i in range(len(image_tmp)):
                #     # print("len :",i)
                #     plt.imshow(image_tmp[i])
                #     plt.show()
                #     plt.imsave(fname=os.path.join("E:/PWORKSPACE/houseUtil/test/midtmp",str(int(datetime.now().timestamp()))+str(i)+"_"+str(label_tmp[i])+".jpg"),arr=image_tmp[i])
                    # cv2.imwrite(os.path.join("E:/PWORKSPACE/houseUtil/test/midtmp",str(int(datetime.now().timestamp()))+"_"+str(i)+".jpg"),image_tmp[0][i])

def main(arvg=None):
    # cifar.maybe_download_and_extract()
    # if tf.gfile.Exists(FLAGS.train_dir):
    #     tf.gfile.DeleteRecursively(FLAGS.train_dir)
    # tf.gfile.MakeDirs(FLAGS.train_dir)
    train()

if __name__ == "__main__":
    tf.app.run()