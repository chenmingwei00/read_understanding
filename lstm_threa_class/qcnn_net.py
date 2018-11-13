import tensorflow as tf
from layers import initializer, regularizer, residual_block, highway, conv, mask_logits, trilinear, total_params, \
     optimized_trilinear_for_attention
class Model(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(self, config, batch, word_mat=None,char_mat=None,filter_sizes=None, embedding_size=None,num_filters=None,trainable=True, l2_reg_lambda=0.0, keep_prob=0.9, graph=None):

        # Placeholders for input, output and dropout
        self.config = config
        self.graph = graph if graph is not None else tf.Graph()
        self.trainable = trainable
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
        # Embedding layer
        with tf.name_scope("embedding"):
            self.char_mat = tf.get_variable("char_mat", initializer=tf.constant(char_mat, dtype=tf.float32),
                                            trainable=True)
            self.W = tf.get_variable("word_mat", initializer=tf.constant(word_mat, dtype=tf.float32),
                                            trainable=True)
            self.c_mask = tf.cast(self.input_x, tf.bool)  # self.c为填充之后的长度是一致的，用0进行填充
            self.q_mask = tf.cast(self.input_x1, tf.bool)
            if trainable:
                self.c_maxlen, self.q_maxlen, = config.para_limit, config.ques_limit,
            else:
                self.c_maxlen, self.q_maxlen = config.test_para_limit, config.test_ques_limit
            self.ch_len = tf.reshape(tf.reduce_sum(tf.cast(tf.cast(self.ch, tf.bool), tf.int32), axis=2), [-1])
            self.qh_len = tf.reshape(tf.reduce_sum(tf.cast(tf.cast(self.qh, tf.bool), tf.int32), axis=2), [-1])
            N, PL, QL, CL, d, dc, nh,dg = config.batch_size, self.c_maxlen, self.q_maxlen,\
                                                      config.char_limit, config.hidden, config.char_dim, \
                                                      config.num_heads,config.char_hidden
            ch_emb = tf.reshape(tf.nn.embedding_lookup(self.char_mat, self.ch), [N * PL, CL, dc])
            qh_emb = tf.reshape(tf.nn.embedding_lookup(self.char_mat, self.qh), [N * QL, CL, dc])
            ch_emb = tf.nn.dropout(ch_emb, 1.0 - 0.5 * self.dropout)
            qh_emb = tf.nn.dropout(qh_emb, 1.0 - 0.5 * self.dropout)
            with tf.variable_scope("cnn_char_Embedding_Layer"):
                # Bidaf style conv-highway encoder
                ch_emb_cnn = conv(ch_emb, d, bias=True, activation=tf.nn.relu, kernel_size=5, name="char_conv", reuse=None)
                qh_emb_cnn = conv(qh_emb, d, bias=True, activation=tf.nn.relu, kernel_size=5, name="char_conv", reuse=True)

                ch_emb_cnn = tf.reduce_max(ch_emb_cnn, axis=1)  # 求出横向唯独的最大特征，这里可以用k_max尝试
                qh_emb_cnn = tf.reduce_max(qh_emb_cnn, axis=1)

                ch_emb_cnn = tf.reshape(ch_emb_cnn, [N, PL, ch_emb_cnn.shape[-1]])
                qh_emb_cnn = tf.reshape(qh_emb_cnn, [N, QL, qh_emb_cnn.shape[-1]])
            with tf.variable_scope('lstm_char_embedding'):
                cell_fw = tf.contrib.rnn.GRUCell(dg)  # 按照字符有多少个gru神经单元
                cell_bw = tf.contrib.rnn.GRUCell(dg)
                _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, ch_emb, self.ch_len,dtype=tf.float32)  # self.ch_len表示训练数据集所有字符平摊之后，实际字符的长度,sequence_length=[bacth_size] is N * PL, because
                # char_hidden is 100 so state_fw and state_bw is [N * PL,100]
                ch_emb_lstm = tf.concat([state_fw, state_bw], axis=1)  # [N * PL,200]
                _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, qh_emb, self.qh_len,
                                                                          dtype=tf.float32)  # state_* [N*QL]
                qh_emb_lstm = tf.concat([state_fw, state_bw], axis=1)  # question_emd is [,200]

                qh_emb_lstm = tf.reshape(qh_emb_lstm, [N, QL, 2 * dg])  # [batch_size,que_len,200]
                ch_emb_lstm = tf.reshape(ch_emb_lstm, [N, PL,2 * dg])  # 以上过程对应了论文里边的 the character-level embedding are generate by ...in the token
                # 这样就把每一个单词的字符转化为单词的字符级别embedding信息，tf.reshape(ch_emb, [N, PL, 2 * dg])
                # 从这里可以看出作者最后那字符的state状态作为字符信息与原始单词embedding进行连接，那么是否可以用拼音
                # 作为汉语的字符级别信息呢，可以尝试
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars1 = tf.nn.embedding_lookup(self.W, self.input_x1)
            ch_emb_cnn = tf.nn.dropout(ch_emb_cnn, self.dropout)
            ch_emb_lstm = tf.nn.dropout(ch_emb_lstm, self.dropout)
            qh_emb_cnn = tf.nn.dropout(qh_emb_cnn, self.dropout)
            qh_emb_lstm = tf.nn.dropout(qh_emb_lstm, self.dropout)
            with tf.variable_scope("lstm_output"):
                c_emb = tf.concat([self.embedded_chars, ch_emb_lstm], axis=2)
                q_emb = tf.concat([self.embedded_chars1,qh_emb_lstm], axis=2)
                c_emb = highway(c_emb, size=d, scope="highway", dropout=self.dropout,
                                reuse=None)  # 相当于对信息进行一次筛选并且让表示的维度降低到75
                q_emb = highway(q_emb, size=d, scope="highway", dropout=self.dropout, reuse=True)
                self.embedded_chars_expanded = tf.expand_dims(c_emb, -1)
                self.embedded_chars_expanded1 = tf.expand_dims(q_emb, -1)
                # Create a convolution + maxpool layer for each filter size
                input_shape = c_emb.get_shape().as_list()
                pooled_outputs = []
                for i, filter_size in enumerate(filter_sizes):
                    with tf.name_scope("conv-maxpool-%s" % filter_size):
                        # Convolution Layer
                        filter_shape = [filter_size, input_shape[-1], 1, num_filters]
                        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                        b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                        l2_loss += tf.nn.l2_loss(W)
                        l2_loss += tf.nn.l2_loss(b)
                        conv_ouput = tf.nn.conv2d(
                            self.embedded_chars_expanded,
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
                    W = tf.get_variable(
                        "W",
                        shape=[num_filters_total, 3],
                        initializer=tf.contrib.layers.xavier_initializer())
                    b = tf.Variable(tf.constant(0.1, shape=[3]), name="b")
                    l2_loss += tf.nn.l2_loss(W)
                    l2_loss += tf.nn.l2_loss(b)
                    self.lstm_scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            with tf.variable_scope("cnn_output"):
                c_emb = tf.concat([self.embedded_chars, ch_emb_cnn], axis=2)
                q_emb = tf.concat([self.embedded_chars1, qh_emb_cnn], axis=2)
                c_emb = highway(c_emb, size=d, scope="highway", dropout=self.dropout,reuse=None)  # 相当于对信息进行一次筛选并且让表示的维度降低到75
                q_emb = highway(q_emb, size=d, scope="highway", dropout=self.dropout, reuse=True)
                self.embedded_chars_expanded = tf.expand_dims(c_emb, -1)
                self.embedded_chars_expanded1 = tf.expand_dims(q_emb, -1)
                # Create a convolution + maxpool layer for each filter size
                input_shape=c_emb.get_shape().as_list()
                pooled_outputs = []
                for i, filter_size in enumerate(filter_sizes):
                    with tf.name_scope("conv-maxpool-%s" % filter_size):
                        # Convolution Layer
                        filter_shape = [filter_size, input_shape[-1], 1, num_filters]
                        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                        b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                        l2_loss += tf.nn.l2_loss(W)
                        l2_loss += tf.nn.l2_loss(b)
                        conv_ouput = tf.nn.conv2d(
                            self.embedded_chars_expanded,
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
                    W = tf.get_variable(
                        "W",
                        shape=[num_filters_total, 3],
                        initializer=tf.contrib.layers.xavier_initializer())
                    b = tf.Variable(tf.constant(0.1, shape=[3]), name="b")
                    l2_loss += tf.nn.l2_loss(W)
                    l2_loss += tf.nn.l2_loss(b)
                    self.cnn_scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
        self.scores=tf.add(self.lstm_scores,self.cnn_scores)/2.0
        print(self.scores)
        print(self.lstm_scores)
        print("3333333333333333333333333")
        self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss
        if trainable:
            with tf.name_scope("loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
                self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

            # Accuracy
            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            if config.decay is not None:
                self.var_ema = tf.train.ExponentialMovingAverage(config.decay)
                ema_op = self.var_ema.apply(tf.trainable_variables())
                with tf.control_dependencies([ema_op]):
                    self.loss = tf.identity(self.loss)

                    self.assign_vars = []
                    for var in tf.global_variables():
                        v = self.var_ema.average(var)
                        if v:
                            self.assign_vars.append(tf.assign(var, v))
            self.lr = tf.minimum(config.init_lr,
                                 0.001 / tf.log(999.) * tf.log(tf.cast(self.global_step, tf.float32) + 1))
            self.opt = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.8, beta2=0.999, epsilon=1e-7)
            grads = self.opt.compute_gradients(self.loss)
            gradients, variables = zip(*grads)
            capped_grads, _ = tf.clip_by_global_norm(
                gradients, config.grad_clip)
            self.train_op = self.opt.apply_gradients(
                zip(capped_grads, variables), global_step=self.global_step)
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=3)
