import tensorflow as tf
from func import cudnn_gru, native_gru, dot_attention, summ, dropout, ptr_net,dot_attention_rnet


class Model(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(self, config, batch, word_mat=None,char_mat=None,  filter_sizes=None, embedding_size=None,num_filters=None,trainable=True, l2_reg_lambda=0.0, keep_prob=0.9, graph=None):

        # Placeholders for input, output and dropout
        self.config = config
        self.graph = graph if graph is not None else tf.Graph()
        self.trainable = trainable
        gru = cudnn_gru if config.use_cudnn else native_gru
        self.is_train = tf.get_variable("is_train", shape=[], dtype=tf.bool, trainable=True)
        if trainable == True:
            self.input_x, self.input_x1, self.ch, self.qh, self.input_y, self.qa_id,self.alternatives_tokens = batch.get_next()  # self.y1 is (64, 3)self.alterh batch_size is[batch,3,alternative_len,chara_len]
        else:
            self.input_x, self.input_x1, self.ch, self.qh,self.alternatives_tokens= batch.get_next()  # self.y1 is (64, 3)self.alterh batch_size is[batch,3,alternative_len,chara_len]
        self.dropout_keep_prob =keep_prob
        self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                           initializer=tf.constant_initializer(0), trainable=False)
        self.dropout = tf.placeholder_with_default(0.5, (), name="dropout")
        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)
        self.c_mask = tf.cast(self.input_x, tf.bool)  # 这里是判断出每一个数据集的context对应实际句子长度的位置(64,400)
        self.q_mask = tf.cast(self.input_x1, tf.bool)  # 同上(64,50)
        self.c_len = tf.reduce_sum(tf.cast(self.c_mask, tf.int32), axis=1)  # 每一个训练数据集实际长度
        self.q_len = tf.reduce_sum(tf.cast(self.q_mask, tf.int32), axis=1)  # 每一个问题的实际长度
        self.ch_len = tf.reshape(tf.reduce_sum(tf.cast(tf.cast(self.ch, tf.bool), tf.int32), axis=2), [-1])
        self.qh_len = tf.reshape(tf.reduce_sum(tf.cast(tf.cast(self.qh, tf.bool), tf.int32), axis=2), [-1])
        # Embedding layer
        N, PL, QL, CL, d, dc,dg= config.batch_size,config.para_limit,config.ques_limit,config.char_limit, config.hidden, config.char_dim,config.char_hidden
        with tf.variable_scope("Input_Embedding_Layer"):
            self.char_mat = tf.get_variable("char_mat", initializer=tf.constant(char_mat, dtype=tf.float32),trainable=True)
            ch_emb = tf.reshape(tf.nn.embedding_lookup(self.char_mat, self.ch), [N * PL, CL, dc])
            qh_emb = tf.reshape(tf.nn.embedding_lookup(self.char_mat, self.qh), [N * QL, CL, dc])
            ch_emb = tf.nn.dropout(ch_emb, 1.0 - 0.5 * self.dropout)
            qh_emb = tf.nn.dropout(qh_emb, 1.0 - 0.5 * self.dropout)

            cell_fw = tf.contrib.rnn.GRUCell(dg)  # 按照字符有多少个gru神经单元
            cell_bw = tf.contrib.rnn.GRUCell(dg)
            _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, ch_emb, self.ch_len,
                dtype=tf.float32)  # self.ch_len表示训练数据集所有字符平摊之后，实际字符的长度,sequence_length=[bacth_size] is N * PL, because
            # char_hidden is 100 so state_fw and state_bw is [N * PL,100]
            ch_emb = tf.concat([state_fw, state_bw], axis=1)  # [N * PL,200]
            _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, qh_emb, self.qh_len,dtype=tf.float32)  # state_* [N*QL]
            qh_emb = tf.concat([state_fw, state_bw], axis=1)  # question_emd is [,200]

            qh_emb = tf.reshape(qh_emb, [N, QL, 2 * dg])  # [batch_size,que_len,200]
            ch_emb = tf.reshape(ch_emb, [N, PL, 2 * dg])  # 以上过程对应了论文里边的 the character-level embedding are generate by ...in the token
                                                   #这样就把每一个单词的字符转化为单词的字符级别embedding信息，tf.reshape(ch_emb, [N, PL, 2 * dg])
                                                    # 从这里可以看出作者最后那字符的state状态作为字符信息与原始单词embedding进行连接，那么是否可以用拼音
                                                    # 作为汉语的字符级别信息呢，可以尝试
            print(qh_emb,"llllllllllllll")
        with tf.name_scope("embedding"):

            self.W = tf.get_variable("word_mat", initializer=tf.constant(word_mat, dtype=tf.float32),
                                            trainable=True)
            if trainable:
                self.c_maxlen, self.q_maxlen, = config.para_limit, config.ques_limit,
            else:
                self.c_maxlen, self.q_maxlen = config.test_para_limit, config.test_ques_limit
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars1 = tf.nn.embedding_lookup(self.W, self.input_x1)
            c_emb = tf.concat([self.embedded_chars, ch_emb], axis=2)
            q_emb= tf.concat([self.embedded_chars1, qh_emb], axis=2)
            # self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            # self.embedded_chars_expanded1 = tf.expand_dims(self.embedded_chars1, -1)
        with tf.variable_scope('simple_lstm'):

            cell_fw = tf.contrib.rnn.GRUCell(dg)  # 按照字符有多少个gru神经单元
            cell_bw = tf.contrib.rnn.GRUCell(dg)
            c_output1, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, c_emb, self.c_len,
                dtype=tf.float32)  # self.ch_len表示训练数据集所有字符平摊之后，实际字符的长度,seq
            c_output=tf.concat(c_output1,axis=-1)
            _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, q_emb, self.q_len,\
                                                                      initial_state_bw=state_bw,initial_state_fw= \
                                                                     state_fw, dtype=tf.float32)  # state_* [N*QL]
            w_shapes = c_output.get_shape().as_list()[-1]
            qh_emb = tf.expand_dims(tf.concat([state_fw, state_bw], axis=1),axis=1)  # question_emd is [,200]
            filter_shape = [w_shapes,w_shapes]
            W = tf.expand_dims(tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W"),axis=0)
            W1=tf.tile(W,[N,1,1])
            imter=tf.matmul(qh_emb,W1)
            c_output_tran=tf.transpose(c_output,[0,2,1])
            p_weight=tf.transpose(tf.nn.softmax(tf.matmul(imter,c_output_tran)),[0,2,1])
            c_oupt_all=tf.reduce_sum(tf.multiply(p_weight,c_output),axis=1)


        with tf.variable_scope("encoding"):
            rnn = gru(num_layers=3, num_units=d, batch_size=N, input_size=c_emb.get_shape(
            ).as_list()[-1], keep_prob=config.keep_prob, is_train=self.is_train)  #input_size对应embedding的长度,此过程是初始化一个gru，双向lstm，包括他们的初始状态
            c = rnn(c_emb, seq_len=self.c_len) #上下文编码输出为batch ，c_maxlen,以及lstm输出长度 [batch_size,sequncen_length,150*3] num_layers is 3 so concat each layers
                                                    #each layer is 150 because each layers has back_forword and feed_forword(75+75)
            q = rnn(q_emb, seq_len=self.q_len) #问题编码
        with tf.variable_scope("attention"):
            qc_att = dot_attention(c, q, mask=self.q_mask, hidden=d,
                                   keep_prob=config.keep_prob, is_train=self.is_train)  # 这个函数实现的是公式（4）中的所有公式
            # print(qc_att,"33333333333333333333")
            # qc_att=dot_attention_rnet(c, q, mask=self.q_mask, hidden=d,keep_prob=config.keep_prob, is_train=self.is_train)
            print(qc_att,"yyyyyyyyyyyyyyyyyyyyyyyyyyy")
            rnn = gru(num_layers=1, num_units=d, batch_size=N, input_size=qc_att.get_shape(
            ).as_list()[-1], keep_prob=config.keep_prob, is_train=self.is_train)
            att = rnn(qc_att, seq_len=self.c_len)  # this is 公式(3) #[batch,c_maxlen,150]
        # Create a convolution + maxpool layer for each filter size
        input_shape=att.get_shape().as_list()
        print(att,"rrrr")
        att=tf.expand_dims(att,-1)
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, input_shape[-1], 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                l2_loss += tf.nn.l2_loss(W)
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                l2_loss += tf.nn.l2_loss(b)
                conv_ouput = tf.nn.conv2d(
                    att,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv_ouput, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, config.para_limit - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            c_oupt_all_shape=c_oupt_all.get_shape().as_list()
            self.h_drop_all=tf.concat([self.h_drop,c_oupt_all],axis=-1)
            W = tf.get_variable(
                "W",
                shape=[num_filters_total+c_oupt_all_shape[-1], 3],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[3]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop_all, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss
        if trainable:
            with tf.name_scope("loss"):
                print(self.scores,self.input_y, "llllllllllllllll")
                self.mse_loss=tf.reduce_mean(tf.square(self.scores-self.input_y))
                losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
                self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
            # Accuracy
            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            # if config.decay is not None:
            #     self.var_ema = tf.train.ExponentialMovingAverage(config.decay)
            #     ema_op = self.var_ema.apply(tf.trainable_variables())
            #     with tf.control_dependencies([ema_op]):
            #         self.loss = tf.identity(self.loss)
            #
            #         self.assign_vars = []
            #         for var in tf.global_variables():
            #             v = self.var_ema.average(var)
            #             if v:
            #                 self.assign_vars.append(tf.assign(var, v))
            self.lr = tf.minimum(config.init_lr,
                                 0.001 / tf.log(999.) * tf.log(tf.cast(self.global_step, tf.float32) + 1))
            self.opt = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.8, beta2=0.999, epsilon=1e-7)
            self.opt2=tf.train.GradientDescentOptimizer(learning_rate=self.lr)
            grads_mean=self.opt2.compute_gradients(self.mse_loss)
            grads = self.opt.compute_gradients(self.loss)+grads_mean
            gradients, variables = zip(*grads)
            capped_grads, _ = tf.clip_by_global_norm(
                gradients, config.grad_clip)
            self.train_op = self.opt.apply_gradients(
                zip(capped_grads, variables), global_step=self.global_step)
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=3)
