import os
import numpy as np
from tqdm import tqdm
import tensorflow as tf


epochs = 25
batch_size = 4
data_zh = "data/zh.tsv"


# 数据处理&读取数据
class DealData(object):

    # 初始化
    def __init__(self, filename=data_zh):

        with open(filename, 'r', encoding='utf-8') as fout:
            data = fout.readlines()[:100]

        self.__input_data = []
        self.__label_data = []
        for i in tqdm(range(len(data))):
            key, pny, hanzi = data[i].split('\t')
            self.__input_data.append(pny.split(' '))
            self.__label_data.append(hanzi.strip('\n').split(' '))

        print('文件读取正常----->>>1')

    # 获取词汇
    def get_vocab(self, data):
        print('词汇获取正常----->>>2')
        vocab = ['<PAD>']
        for line in tqdm(data):
            for char in line:
                if char not in vocab:
                    vocab.append(char)
        return vocab

    # 获取输入值与标签值
    def pyn_han(self):
        pny2id = self.get_vocab(self.__input_data)  # 获取输入值
        han2id = self.get_vocab(self.__label_data)  # 获取标签值
        print('PNY:', pny2id[:10])
        print('HAN:', han2id[:10])
        return pny2id, han2id

    # 获取输入数量与标签数量
    def input_lable_num(self):

        print('输入与输出获取正常----->>>3')
        pny2id, han2id = self.pyn_han()
        input_num = [[pny2id.index(pny) for pny in line] for line in tqdm(self.__input_data)]
        label_num = [[han2id.index(han) for han in line] for line in tqdm(self.__label_data)]
        return input_num, label_num

    # 文件分批读取
    def get_batch(self, input_data, label_data, batch_size):
        print('文件分批读取正常----->>>4')
        batch_num = len(input_data) // batch_size
        for k in range(batch_num):
            begin = k * batch_size
            end = begin + batch_size
            input_batch = input_data[begin:end]
            label_batch = label_data[begin:end]
            max_len = max([len(line) for line in input_batch])
            input_batch = np.array([line + [0] * (max_len - len(line)) for line in input_batch])
            label_batch = np.array([line + [0] * (max_len - len(line)) for line in label_batch])
            yield input_batch, label_batch

    # # input与labelbatch
    # def input_label_batch(self):
    #
    #     input_num, label_num = self.input_lable_num()
    #     batch = self.get_batch(input_num, label_num, 4)
    #     input_batch, label_batch = next(batch)
    #     # print(input_batch)
    #     # print(label_batch)

    # 参数设定
    def create_hparams(self):
        print('参数设定正常----->>>5')
        params = tf.contrib.training.HParams(
            num_heads=8,
            num_blocks=6,
            # vocab
            input_vocab_size=50,
            label_vocab_size=50,
            # embedding size
            max_length=100,
            hidden_units=512,
            dropout_rate=0.2,
            lr=0.0003,
            is_training=True)
        return params


# 模型各分层的实现
class BuildLayer(object):

    # layer norm层
    def normalize(self,
                  inputs,
                  epsilon = 1e-8,
                  scope="ln",
                  reuse=None):
        print('layer norm层正常----->>>6')
        with tf.variable_scope(scope, reuse=reuse):
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]

            mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
            beta= tf.Variable(tf.zeros(params_shape))
            gamma = tf.Variable(tf.ones(params_shape))
            normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
            outputs = gamma * normalized + beta

        return outputs

    # embedding层
    def embedding(self,
                  inputs,
                  vocab_size,
                  num_units,
                  zero_pad=True,
                  scale=True,
                  scope="embedding",
                  reuse=None):
        print('embedding层正常----->>>7')
        with tf.variable_scope(scope, reuse=reuse):
            lookup_table = tf.get_variable('lookup_table',
                                           dtype=tf.float32,
                                           shape=[vocab_size, num_units],
                                           initializer=tf.contrib.layers.xavier_initializer())
            if zero_pad:
                lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                          lookup_table[1:, :]), 0)
            outputs = tf.nn.embedding_lookup(lookup_table, inputs)

            if scale:
                outputs = outputs * (num_units ** 0.5)

        return outputs

    # multihead层
    def multihead_attention(self,
                            emb,
                            queries,
                            keys,
                            num_units=None,
                            num_heads=8,
                            dropout_rate=0,
                            is_training=True,
                            causality=False,
                            scope="multihead_attention",
                            reuse=None):
        print('multihead层正常----->>>8')
        with tf.variable_scope(scope, reuse=reuse):
            # Set the fall back option for num_units
            if num_units is None:
                num_units = queries.get_shape().as_list[-1]

            # Linear projections
            Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)  # (N, T_q, C)
            K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)
            V = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)

            # Split and concat
            Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
            K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
            V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

            # Multiplication
            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

            # Scale
            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

            # Key Masking
            key_masks = tf.sign(tf.abs(tf.reduce_sum(emb, axis=-1)))  # (N, T_k)
            key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
            key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

            # Causality = Future blinding
            if causality:
                diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
                tril = tf.contrib.linalg.LinearOperatorTriL(diag_vals).to_dense()  # (T_q, T_k)
                masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)

                paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
                outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

            # Activation
            outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

            # Query Masking
            query_masks = tf.sign(tf.abs(tf.reduce_sum(emb, axis=-1)))  # (N, T_q)
            query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
            query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
            outputs *= query_masks  # broadcasting. (N, T_q, C)

            # Dropouts
            outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

            # Weighted sum
            outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

            # Restore shape
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

            # Residual connection
            outputs += queries

            # Normalize
            outputs = self.normalize(outputs)  # (N, T_q, C)

        return outputs

    # feedforward
    def feedforward(self,
                    inputs,
                    num_units=[2048, 512],
                    scope="multihead_attention",
                    reuse=None):
        print('feedforward层正常----->>>9')
        with tf.variable_scope(scope, reuse=reuse):
            # Inner layer
            params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                      "activation": tf.nn.relu, "use_bias": True}
            outputs = tf.layers.conv1d(**params)
            # Readout layer
            params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                      "activation": None, "use_bias": True}
            outputs = tf.layers.conv1d(**params)
            # Residual connection
            outputs += inputs
            # Normalize
            outputs = self.normalize(outputs)

        return outputs

    # label_smoothing
    def label_smoothing(self, inputs, epsilon=0.1):
        print('label_smoothing正常----->>>9')
        K = inputs.get_shape().as_list()[-1]  # number of channels
        return ((1 - epsilon) * inputs) + (epsilon / K)


dealdat = DealData()
arg = dealdat.create_hparams()
pny2id, han2id = dealdat.pyn_han()
input_num, label_num = dealdat.input_lable_num()
arg.input_vocab_size = len(pny2id)
arg.label_vocab_size = len(han2id)


# 搭建模型
class Graph(object):
    def __init__(self, is_training=True):

        print('Graph初始化正常----->>>10')
        # self.dealdat = DealData()
        # self.arg = self.dealdat.create_hparams()
        # self.__pny2id, self.__han2id = self.dealdat.pyn_han()
        # self.__input_num, self.__label_num = self.dealdat.input_lable_num()
        # self.arg.input_vocab_size = len(self.__pny2id)
        # self.arg.label_vocab_size = len(self.__han2id)

        tf.reset_default_graph()
        self.is_training = arg.is_training
        self.hidden_units = arg.hidden_units
        self.input_vocab_size = arg.input_vocab_size
        self.label_vocab_size = arg.label_vocab_size
        self.num_heads = arg.num_heads
        self.num_blocks = arg.num_blocks
        self.max_length = arg.max_length
        self.lr = arg.lr
        self.dropout_rate = arg.dropout_rate

        self.laybuild = BuildLayer()
        # input
        self.x = tf.placeholder(tf.int32, shape=(None, None))
        self.y = tf.placeholder(tf.int32, shape=(None, None))
        # embedding
        self.emb = self.laybuild.embedding(self.x,
                                           vocab_size=self.input_vocab_size,
                                           num_units=self.hidden_units,
                                           scale=True,
                                           scope="enc_embed")
        self.enc = self.emb + self.laybuild.embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.x)[1]), 0),
                                                              [tf.shape(self.x)[0], 1]),
                                                      vocab_size=self.max_length,
                                                      num_units=self.hidden_units,
                                                      zero_pad=False,
                                                      scale=False,
                                                      scope="enc_pe")
        # Dropout
        self.enc = tf.layers.dropout(self.enc,
                                     rate=self.dropout_rate,
                                     training=tf.convert_to_tensor(self.is_training))

        # Blocks
        for i in range(self.num_blocks):
            with tf.variable_scope("num_blocks_{}".format(i)):
                # Multihead Attention
                self.enc = self.laybuild.multihead_attention(emb=self.emb,
                                                             queries=self.enc,
                                                             keys=self.enc,
                                                             num_units=self.hidden_units,
                                                             num_heads=self.num_heads,
                                                             dropout_rate=self.dropout_rate,
                                                             is_training=self.is_training,
                                                             causality=False)

        # Feed Forward
        self.outputs = self.laybuild.feedforward(self.enc, num_units=[4 * self.hidden_units, self.hidden_units])

        # Final linear projection
        self.logits = tf.layers.dense(self.outputs, self.label_vocab_size)
        self.preds = tf.to_int32(tf.argmax(self.logits, axis=-1))
        self.istarget = tf.to_float(tf.not_equal(self.y, 0))
        self.acc = tf.reduce_sum(tf.to_float(tf.equal(self.preds, self.y)) * self.istarget) / (
            tf.reduce_sum(self.istarget))
        tf.summary.scalar('acc', self.acc)

        if is_training:
            # Loss
            self.y_smoothed = self.laybuild.label_smoothing(tf.one_hot(self.y, depth=self.label_vocab_size))
            self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_smoothed)
            self.mean_loss = tf.reduce_sum(self.loss * self.istarget) / (tf.reduce_sum(self.istarget))

            # Training Scheme
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
            self.train_op = self.optimizer.minimize(self.mean_loss, global_step=self.global_step)

            # Summary
            tf.summary.scalar('mean_loss', self.mean_loss)
            self.merged = tf.summary.merge_all()


# 模型训练
def train_model():

    print('模型训练正常----->>>11')
    g = Graph(arg)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        sess.run(tf.global_variables_initializer())
        if os.path.exists('logs/model.meta'):
            saver.restore(sess, 'logs/model')
        writer = tf.summary.FileWriter('tensorboard/lm', tf.get_default_graph())
        for k in range(epochs):
            total_loss = 0
            batch_num = len(input_num) // batch_size
            batch = dealdat.get_batch(input_num, label_num, batch_size)
            for i in range(batch_num):
                input_batch, label_batch = next(batch)
                feed = {g.x: input_batch, g.y: label_batch}
                cost, _ = sess.run([g.mean_loss, g.train_op], feed_dict=feed)
                total_loss += cost
                if (k * batch_num + i) % 10 == 0:
                    rs=sess.run(merged, feed_dict=feed)
                    writer.add_summary(rs, k * batch_num + i)
            if (k+1) % 5 == 0:
                print('epochs', k+1, ': average loss = ', total_loss/batch_num)
        saver.save(sess, 'logs/model')
        writer.close()


# 模型推断
def model_predict():
    print('模型推断正常----->>>12')
    arg.is_training = False
    g = Graph(arg)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, 'logs/model')
        while True:
            line = input('输入测试拼音: ')
            if line == 'exit': break
            line = line.strip('\n').split(' ')
            print('--->', type(line))
            x = np.array([pny2id.index(pny) for pny in line])
            x = x.reshape(1, -1)
            preds = sess.run(g.preds, {g.x: x})
            got = ''.join(han2id[idx] for idx in preds[0])
            print(got)


if __name__ == '__main__':

    # train_model()
    model_predict()












