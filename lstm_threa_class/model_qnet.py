
import tensorflow as tf
from layers import initializer, regularizer, residual_block, highway, conv, mask_logits, trilinear, total_params, \
    optimized_trilinear_for_attention


class Model(object):
    def __init__(self, config, batch, word_mat=None, char_mat=None, filter_sizes=None, embedding_size=None,num_filters=None,trainable=True, opt=True, demo=False, graph=None):
        self.config = config
        self.demo = demo
        self.graph = graph if graph is not None else tf.Graph()
        self.trainable = trainable
        self.l2_loss = tf.constant(0.0)
        self.l2_reg_lambda = 0.7
        if trainable == True:
            self.c, self.q, self.ch, self.qh, self.input_y, self.qa_id,self.alternatives_tokens = batch.get_next()  # self.y1 is (64, 3)self.alterh batch_size is[batch,3,alternative_len,chara_len]
        else:
            self.c, self.q, self.ch, self.qh,self.alternatives_tokens= batch.get_next()  # self.y1 is (64, 3)self.alterh batch_size is[batch,3,alternative_len,chara_len]

        self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                           initializer=tf.constant_initializer(0), trainable=False)
        self.dropout = tf.placeholder_with_default(0.5, (), name="dropout")

        # self.c, self.q, self.ch, self.qh, self.y1, self.y2, self.qa_id = batch.get_next()

        # self.word_unk = tf.get_variable("word_unk", shape = [config.glove_dim], initializer=initializer())
        self.word_mat = tf.get_variable("word_mat", initializer=tf.constant(word_mat, dtype=tf.float32), trainable=True)
        self.char_mat = tf.get_variable("char_mat", initializer=tf.constant(char_mat, dtype=tf.float32),trainable=True)

        self.c_mask = tf.cast(self.c, tf.bool) #self.c为填充之后的长度是一致的，用0进行填充
        self.q_mask = tf.cast(self.q, tf.bool)

        self.c_len = tf.reduce_sum(tf.cast(self.c_mask, tf.int32), axis=1)#表示每一个句子的实际长度
        self.q_len = tf.reduce_sum(tf.cast(self.q_mask, tf.int32), axis=1)



        if opt:
            #此过程会按照batch的最大长度对扩充句子重新缩减
            N, CL= config.batch_size if not self.demo else 1, config.char_limit
            self.c_maxlen = tf.reduce_max(self.c_len)#一个batch中最大的长度
            self.q_maxlen = tf.reduce_max(self.q_len)


            self.c = tf.slice(self.c, [0, 0], [N, self.c_maxlen])
            self.q = tf.slice(self.q, [0, 0], [N, self.q_maxlen])

            self.c_mask = tf.slice(self.c_mask, [0, 0], [N, self.c_maxlen])
            self.q_mask = tf.slice(self.q_mask, [0, 0], [N, self.q_maxlen])

            self.ch = tf.slice(self.ch, [0, 0, 0], [N, self.c_maxlen, CL])
            self.qh = tf.slice(self.qh, [0, 0, 0], [N, self.q_maxlen, CL])

            # self.y1 = tf.slice(self.y1, [0, 0], [N, self.c_maxlen])
            # self.y2 = tf.slice(self.y2, [0, 0], [N, self.c_maxlen])
        else:
            if trainable:
                self.c_maxlen, self.q_maxlen,= config.para_limit, config.ques_limit,
            else:
                self.c_maxlen, self.q_maxlen=config.test_para_limit, config.test_ques_limit

        self.ch_len = tf.reshape(tf.reduce_sum(tf.cast(tf.cast(self.ch, tf.bool), tf.int32), axis=2), [-1])
        self.qh_len = tf.reshape(tf.reduce_sum(tf.cast(tf.cast(self.qh, tf.bool), tf.int32), axis=2), [-1])

        self.forward(trainable)
        total_params()


    def forward(self,trainable):
        config = self.config
        N, PL, QL, CL, d, dc, nh= config.batch_size,self.c_maxlen, self.q_maxlen,\
                                               config.char_limit, config.hidden, config.char_dim, \
                                               config.num_heads,


        with tf.variable_scope("Input_Embedding_Layer"):
            ch_emb = tf.reshape(tf.nn.embedding_lookup(self.char_mat, self.ch), [N * PL, CL, dc]) #[一个句子共有多少单词,每个单词的字符个数,每一个字符的维度]
            qh_emb = tf.reshape(tf.nn.embedding_lookup(self.char_mat, self.qh), [N * QL, CL, dc])



            ch_emb = tf.nn.dropout(ch_emb, 1.0 - 0.5 * self.dropout)
            qh_emb = tf.nn.dropout(qh_emb, 1.0 - 0.5 * self.dropout)



            # Bidaf style conv-highway encoder以下是得到卷积之后的特征输出
            ch_emb = conv(ch_emb, d,bias=True, activation=tf.nn.relu, kernel_size=5, name="char_conv", reuse=None)#[batch,feature_len,d]
            qh_emb = conv(qh_emb, d,bias=True, activation=tf.nn.relu, kernel_size=5, name="char_conv", reuse=True)

            ch_emb = tf.reduce_max(ch_emb, axis=1) #求出横向唯独的最大特征，这里可以用k_max尝试,而没有用max_pooling
            qh_emb = tf.reduce_max(qh_emb, axis=1)



            ch_emb = tf.reshape(ch_emb, [N, PL, ch_emb.shape[-1]])#最终转变为句子长度对应的维度，
            qh_emb = tf.reshape(qh_emb, [N, QL, qh_emb.shape[-1]])


            c_emb = tf.nn.dropout(tf.nn.embedding_lookup(self.word_mat, self.c), 1.0 - self.dropout)
            q_emb = tf.nn.dropout(tf.nn.embedding_lookup(self.word_mat, self.q), 1.0 - self.dropout)


            c_emb = tf.concat([c_emb, ch_emb], axis=2) #把字符与对应的特征进行连接[batch,sequence_len,对应的输出维度]
            q_emb = tf.concat([q_emb, qh_emb], axis=2)

            c_emb = highway(c_emb, size=d, scope="highway", dropout=self.dropout, reuse=None)#相当于对信息进行一次筛选,并且让表示的维度降低到75,[batch,sql_len,75]
            q_emb = highway(q_emb, size=d, scope="highway", dropout=self.dropout, reuse=True)


        with tf.variable_scope("Embedding_Encoder_Layer"):
            c = residual_block(c_emb,
                               num_blocks=1,
                               num_conv_layers=4,
                               kernel_size=7,
                               mask=self.c_mask,
                               num_filters=d,
                               num_heads=nh,
                               seq_len=self.c_len,
                               scope="Encoder_Residual_Block",
                               bias=False,
                               dropout=self.dropout)
            q = residual_block(q_emb,
                               num_blocks=1,
                               num_conv_layers=4,
                               kernel_size=7,
                               mask=self.q_mask,
                               num_filters=d,
                               num_heads=nh,
                               seq_len=self.q_len,
                               scope="Encoder_Residual_Block",
                               reuse=True,  # Share the weights between passage and question
                               bias=False,
                               dropout=self.dropout)
        with tf.variable_scope("Context_to_Query_Attention_Layer"):
            # C = tf.tile(tf.expand_dims(c,2),[1,1,self.q_maxlen,1])
            # Q = tf.tile(tf.expand_dims(q,1),[1,self.c_maxlen,1,1])
            # S = trilinear([C, Q, C*Q], input_keep_prob = 1.0 - self.dropout)
            S = optimized_trilinear_for_attention([c, q], self.c_maxlen, self.q_maxlen,
                                                  input_keep_prob=1.0 - self.dropout)
            mask_q = tf.expand_dims(self.q_mask, 1)
            S_ = tf.nn.softmax(mask_logits(S, mask=mask_q))
            mask_c = tf.expand_dims(self.c_mask, 2)
            S_T = tf.transpose(tf.nn.softmax(mask_logits(S, mask=mask_c), dim=1), (0, 2, 1))
            self.c2q = tf.matmul(S_, q)
            self.q2c = tf.matmul(tf.matmul(S_, S_T), c)
            attention_outputs = [c, self.c2q, c * self.c2q, c * self.q2c]
        with tf.variable_scope("Model_Encoder_Layer"):
            inputs = tf.concat(attention_outputs, axis=-1)
            self.enc = [conv(inputs, d, name="input_projection")]

            for i in range(3):
                if i % 2 == 0:  # dropout every 2 blocks
                    self.enc[i] = tf.nn.dropout(self.enc[i], 1.0 - self.dropout)
                self.enc.append(
                    residual_block(self.enc[i],
                                   num_blocks=7,
                                   num_conv_layers=2,
                                   kernel_size=5,
                                   mask=self.c_mask,
                                   num_filters=d,
                                   num_heads=nh,
                                   seq_len=self.c_len,
                                   scope="Model_Encoder",
                                   bias=False,
                                   reuse=True if i > 0 else None,
                                   dropout=self.dropout)
                )
        with tf.variable_scope('question_rnn'):
            self.gru = tf.contrib.rnn.GRUCell(d)
            initstate = self.gru.zero_state(batch_size=N, dtype=tf.float32)
            output,state=tf.nn.dynamic_rnn(self.gru,q,initial_state=initstate)
            # self.qandc=tf.concat([self.q2c,self.c2q],axis=2)
            # self.qandc=dense(self.qandc,d)
            # output,state=tf.nn.dynamic_rnn(self.gru,self.qandc,initial_state=initstate)#(32,?,75)

            state=tf.expand_dims(state,axis=2)
            weight1=tf.matmul(self.enc[1],state)
            weight2=tf.matmul(self.enc[2],state)
            weight3=tf.matmul(self.enc[3],state)

            weight_enc1=tf.multiply(self.enc[1],weight1)
            weight_enc1=tf.reduce_sum(weight_enc1,axis=1)

            weight_enc2 = tf.multiply(self.enc[2], weight2)
            weight_enc2 = tf.reduce_sum(weight_enc2, axis=1)

            weight_enc3 = tf.multiply(self.enc[3], weight3)
            weight_enc3 = tf.reduce_sum(weight_enc3, axis=1)


        with tf.variable_scope("Output_Layer"):
            print(weight_enc1,"ggggggggggggggggg")
            inputs_shape=weight_enc1.get_shape().as_list()
            W = tf.get_variable(
            "W",
            shape=[inputs_shape[-1], 3],
            initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[3]), name="b")
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)
            self.scores1 = tf.nn.xw_plus_b(weight_enc1, W, b, name="scores")
            self.scores2 = tf.nn.xw_plus_b(weight_enc2, W, b, name="scores")
            self.scores3 = tf.nn.xw_plus_b(weight_enc3, W, b, name="scores")
            self.scores=(self.scores1+self.scores2+self.scores3)/3.0
            print(self.scores)
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            if trainable:
                with tf.name_scope("loss"):
                    print(self.scores, self.input_y, "llllllllllllllll")
                    losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
                    self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * self.l2_loss
                    # Accuracy
                with tf.name_scope("accuracy"):
                    correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
                    self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
                # losses2 = tf.nn.softmax_cross_entropy_with_logits(
                #     logits=logits2, labels=self.y2)
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
            # self.logits = [mask_logits(start_logits, mask=self.c_mask),
            #                mask_logits(end_logits, mask=self.c_mask)]
            #
            # logits1, logits2 = [l for l in self.logits]
            #
            # outer = tf.matmul(tf.expand_dims(tf.nn.softmax(logits1), axis=2),
            #                   tf.expand_dims(tf.nn.softmax(logits2), axis=1))
            # outer = tf.matrix_band_part(outer, 0, config.ans_limit)
            # self.yp1 = tf.argmax(tf.reduce_max(outer, axis=2), axis=1)
            # self.yp2 = tf.argmax(tf.reduce_max(outer, axis=1), axis=1)




        # if config.l2_norm is not None:
        #     variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        #     l2_loss = tf.contrib.layers.apply_regularization(regularizer, variables)
        #     self.loss += l2_loss



    def get_loss(self):
        return self.loss

    def get_global_step(self):
        return self.global_step

    def cos_sine(self, all_pre, alter):
        pooled_len_spa1 = tf.sqrt(tf.reduce_sum(tf.multiply(all_pre, all_pre), 1))  # 利用余弦相似度求解
        pooled_len_spa2 = tf.sqrt(tf.reduce_sum(tf.multiply(alter, alter), 1))
        pooled_mul_spa12 = tf.reduce_sum(tf.multiply(all_pre, alter), 1)  # 计算向量的点乘Batch模式
        # pooled_mul_13 = tf.reduce_sum(tf.multiply(pooled_flat_1, pooled_flat_3), 1)

        with tf.name_scope("output_spa"):
            cos_spa12 = tf.div(pooled_mul_spa12, tf.multiply(pooled_len_spa1, pooled_len_spa2),
                               name="scores_spa")  # 最后相似
        return cos_spa12
def dense(inputs, hidden, use_bias=True, scope="dense"):
    #整个函数利用权重，把多层lstm的输出转化为一层lstm的维度
    with tf.variable_scope(scope):
        shape = tf.shape(inputs) #tf.shape()中的数据类型可以是tensor，list，array
        dim = inputs.get_shape().as_list()[-1] #每一个输出的维度，也就是就是句子长度的每一个单词的维度
        out_shape = [shape[idx] for idx in range(len(inputs.get_shape().as_list()) - 1)] + [hidden]
        # flat_inputs = tf.reshape(inputs, [-1, dim]) #把每一个单词的rnn的output输出转换为一个维度
        W = tf.get_variable("W", [dim, hidden])
        bathc=inputs.get_shape().as_list()[0]
        W1=tf.tile(W,[bathc,1])
        W1=tf.reshape(W1,[bathc,dim,-1])
        res = tf.matmul(inputs, W1)#变为[-1,hidden]维度
        print(res,"gggggggggggggggggggggggggggggggggggg")
        return res