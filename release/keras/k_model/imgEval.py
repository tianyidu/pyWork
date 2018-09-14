import tensorflow as tf
from . import readImg

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('ckpt', 'E:/PWORKSPACE/house3/cnn/coreCnn/ckpt/ckpt',"Directory where to read model checkpoints.")

def eval():
    img = readImg.Img(r"E:\PWORKSPACE\house3\test.jpg")
    image,label = img.preProcess()
    print(image)
    saver = tf.train.import_meta_graph(r"E:/PWORKSPACE/house3/cnn/coreCnn/ckpt/ckpt/model.ckpt-1.meta")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(FLAGS.ckpt)
        if ckpt and ckpt.model_checkpoint_path:
            print("restore model ",ckpt.model_checkpoint_path)
            saver.restore(sess,ckpt.model_checkpoint_path)

            print(saver,dir(saver))
            print(dir(sess))
            print(sess.graph)
            print(saver._var_list,saver.saver_def)
            print("-----")
            for var in tf.get_default_graph().get_all_collection_keys():
                print(var,tf.get_default_graph().get_collection(var))

            # print("---------")
            # for op in tf.get_default_graph().get_operations():
            #     print(op)
            softmax_linear = tf.get_default_graph().get_tensor_by_name("softmax_linear/softmax_linear:0")

            print("----",sess.run(softmax_linear,feed_dict={"input_image:0":sess.run(image)}))
            # print(sess.run(softmax_linear,feed_dict={"conv1_1/images:0":sess.run(image)}))

eval()
# flag = tf.placeholder(dtype=tf.bool,shape=[1],name="flag")
# with tf.Session() as sess:
#     t = sess.run(flag,feed_dict={flag:[True]})
#     if t:
#         print("true")
#     else:
#         print(t)
