import tensorflow as tf
from release import imgInput
import datetime
from matplotlib import pyplot as plt


class houseCnn:
    def __init__(self,imgs,labels,batch_size=32,lr=0.01,decay_stept=100,decay_rate=0.9):
        self.imgs = imgs
        self.labels = labels
        self.batch_size = batch_size
        self.lr = lr
        self.decay_stept = decay_stept
        self.decay_rate = decay_rate

    def variable_weights(self,name,shape,initializer,wd=None):
        with tf.device("/cpu:0"):
            variable = tf.get_variable(name=name,shape=shape,initializer=initializer)
        if wd is not None:
            weights_decay = tf.multiply(tf.nn.l2_loss(variable),name="weights_loss")
            tf.add_to_collection("losses",weights_decay)
        return variable

    def addLayer(self,layer_name,imgs,output_shape,initializer=tf.truncated_normal_initializer(stddev=1e-1)):
        with tf.variable_scope(layer_name) as scope:
            weights = self.variable_weights("weight",output_shape,initializer=initializer)
            conv = tf.nn.conv2d(imgs,weights,[1,1,1,1],padding="SAME")
            bias = self.variable_weights("bias",[output_shape[-1]],initializer=tf.constant_initializer(0.0))
            bias_add = tf.nn.bias_add(conv,bias)
            out = tf.nn.relu(bias_add,name=scope.name)
        return out

    def addFcLayer(self,name,imgs,shape,initializer=tf.truncated_normal_initializer(stddev=1e-1)):
        with tf.variable_scope(name) as scope:
            weight = self.variable_weights("weights",shape=shape,initializer=initializer)
            bias = self.variable_weights("bias",[shape[-1]],initializer=tf.constant_initializer(0.0))
            fc = tf.nn.relu(tf.matmul(imgs,weight)+bias,name=scope.name)
        return fc

    def network(self):
        conv1_1 = self.addLayer("conv1_1",self.imgs,[3,3,3,32])
        conv1_2 = self.addLayer("conv1_2",conv1_1,[3,3,32,32])
        pool1 = tf.nn.max_pool(conv1_2,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME",name="pool1")

        conv2_1 = self.addLayer("conv2_1",pool1,[3,3,32,48])
        conv2_2 = self.addLayer("conv2_2",conv2_1,[3,3,48,48])
        pool2 = tf.nn.max_pool(conv2_2,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME",name="pool2")

        conv3_1 = self.addLayer("conv3_1",pool2,[3,3,48,64])
        conv3_2 = self.addLayer("conv3_2",conv3_1,[3,3,64,64])
        conv3_3 = self.addLayer("conv3_3",conv3_2,[3,3,64,64])
        pool3 = tf.nn.max_pool(conv3_3,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME",name="pool3")

        conv4_1 = self.addLayer("conv4_1",pool3,[3,3,64,96])
        conv4_2 = self.addLayer("conv4_2",conv4_1,[3,3,96,96])
        conv4_3 = self.addLayer("conv4_3",conv4_2,[3,3,96,96])
        pool4 = tf.nn.max_pool(conv4_3,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME",name="pool4")

        conv5_1 = self.addLayer("conv5_1",pool4,[3,3,96,128])
        conv5_2 = self.addLayer("conv5_2",conv5_1,[3,3,128,128])
        conv5_3 = self.addLayer("conv5_3",conv5_2,[3,3,128,128])
        pool5 = tf.nn.max_pool(conv5_3,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME",name="pool5")

        reshape = tf.reshape(pool5,[self.batch_size,-1],name="reshape")
        print("reshape",reshape.get_shape().as_list()[1])

        fc1 = self.addFcLayer(name="fc1",imgs=reshape,shape=[12800,1000],initializer=tf.truncated_normal_initializer(stddev=1/1000.0))
        fc2 = self.addFcLayer(name="fc2",imgs=fc1,shape=[1000,1000],initializer=tf.truncated_normal_initializer(stddev=1/1000.0))
        fc3 = self.addFcLayer(name="fc3",imgs=fc2,shape=[1000,512],initializer=tf.truncated_normal_initializer(stddev=1/512.0))

        softmax = self.addFcLayer("softmax",fc3,[512,3])

        return softmax

    def train(self,gloable_stept):

        labels = tf.cast(self.labels,tf.int64)
        imgs = self.network()
        cross_entry = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=imgs,name="cross_entry")
        cross_entry_mean = tf.reduce_mean(cross_entry)

        tf.add_to_collection("losses",cross_entry_mean)

        losses = tf.add_n(tf.get_collection("losses"),name="total_loss")

        lr = tf.train.exponential_decay(self.lr,gloable_stept,decay_steps=self.decay_stept,decay_rate=self.decay_rate,staircase=True,name="lr")

        train_op = tf.train.GradientDescentOptimizer(lr).minimize(losses,global_step=gloable_stept)

        return train_op,losses

batch_size = 3
with tf.Session() as sess:

    global_step = tf.train.get_or_create_global_step()

    images = tf.placeholder(tf.float32,[None,300,300,3],name="images")
    label = tf.placeholder(tf.float32,[None],name="label")

    housecnn = houseCnn(images,label,batch_size=batch_size,decay_stept=10)
    train_op,losses = housecnn.train(global_step)

    print(tf.local_variables(),tf.global_variables())

    sess.run(tf.global_variables_initializer())

    imgs, ls = imgInput.createBatch("E:/PWORKSPACE/houseUtil/resized_2/train.tfc",batch_size)

    tf.train.start_queue_runners()
    for i in range(10000):
        # print("gloabl_stept",sess.run(global_step))
        x,y = sess.run([imgs,ls])
        # print("labels",label,"img",x)

        sess.run(train_op,feed_dict={housecnn.imgs:x,housecnn.labels:y})
        if i % 100 == 0:
            print("y",y)
            for j in range(len(x)):
                plt.imshow(x[j])
                plt.show()
            print(datetime.datetime.now(),"stept",i,"losses",sess.run(losses,feed_dict={housecnn.imgs:x,housecnn.labels:y}))







