#coding:utf8
from datetime import datetime
import math
import time
import numpy as np
import tensorflow as tf
import cv2

from release import imgInference
from release import cifar_input
# from release import cifar

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', 'E:/PWORKSPACE/houseUtil/summary',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'summary',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', 'E:/PWORKSPACE/houseUtil/ckpt/ckpt',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 650,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 1,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', True,
                         """Whether to run eval only once.""")

result = np.array(["厨房","卫生间","卧室"])
def eval_once(saver,summary_writer,top_k_op,summary_op,logits,labels,images):
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess,ckpt.model_checkpoint_path)
            print("model_checkpoint_path ",ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split("/")[-1].split("-")[-1]
        else:
            print("No checkpoint file found")
            return

        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess,coord=coord,daemon=True,start=True))

            # num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
            num_iter = FLAGS.num_examples
            true_count = 0
            total_sample_count = num_iter * FLAGS.batch_size
            step = 0
            while step < num_iter and not coord.should_stop():
                predictions,logitis_tmp,labels_tmp,image_tmp = sess.run([top_k_op,logits,labels,images])
                if step % 2 == 0:
                    print("prediction: ",predictions,logitis_tmp,labels_tmp)
                # w1,w2,w3 = sess.run([saver._var_list["conv1_1/weights/ExponentialMovingAverage"],saver._var_list["local4/weights/ExponentialMovingAverage"],saver._var_list["softmax_linear/weights/ExponentialMovingAverage"]])
                # print("w ",w1,w2,w3)
                # cv2.imshow("w",np.array(255-np.abs(w[0][0][1][0])))
                # cv2.waitKey(0)
                true_count += np.sum(predictions)
                step += 1
            precision = true_count / total_sample_count
            print('%s: precision @ 1 = %.3f  @@ total_sample_count:%d, step:%d, true_count:%d' % (datetime.now(), precision,total_sample_count,step,true_count))
            summary = tf.Summary()
            if summary_op:
                summary.ParseFromString(sess.run(summary_op))
                summary.value.add(tag='Precision @ 1', simple_value=precision)
                summary_writer.add_summary(summary, global_step)

        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)
        return sess.run(tf.argmax(logitis_tmp,1))

def evaluate():
  with tf.Graph().as_default() as g:
    # eval_data = FLAGS.eval_data == 'test'
    # images, labels = imgInference.eval_inputs()
    # images,labels = cifar.inputs()
    filename = r"E:\PWORKSPACE\houseUtil\resized_2\wc\5i5j_wc.jpg"
    images, labels = imgInference.one_input(filename,[2])
    # images, labels = imgInference.inputs()
    logits,_ = imgInference.inference(images,True)
    # images, labels = cifar.one_input(r"E:\PWORKSPACE\houseUtil\resized\wc\0bf2ac91-360f-4d13-9aa2-535b24e071bd.jpg",[1])
    # logits = cifar.inference(images)
    # print("evaluate: ",logits,labels)
    top_k_op = tf.nn.in_top_k(logits, labels, 1)
    variable_averages = tf.train.ExponentialMovingAverage(imgInference.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

    while True:
      prediction = eval_once(saver, summary_writer, top_k_op, summary_op,logits,labels,images)
      print(filename," 预测值为 ",result[prediction])
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
    # cifarModel.maybe_download_and_extract()
    # if tf.gfile.Exists(FLAGS.eval_dir):
    #     tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    # tf.gfile.MakeDirs(FLAGS.eval_dir)
    evaluate()


if __name__ == '__main__':
  tf.app.run()
  #   imgInference.one_input(r"E:\PWORKSPACE\houseUtil\resized\livingroom\0b600e66-6e01-46cc-8378-693b9aa11c3a.jpg",[3])