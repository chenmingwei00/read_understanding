import tensorflow as tf
import numpy as np

class Model(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(self, config, batch, word_mat=None, filter_sizes=None, embedding_size=None,num_filters=None,trainable=True, l2_reg_lambda=0.0, keep_prob=0.9, graph=None):

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
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.get_variable("word_mat", initializer=tf.constant(word_mat, dtype=tf.float32),
                                            trainable=True)
            self.c_mask = tf.cast(self.input_x, tf.bool)  # self.c为填充之后的长度是一致的，用0进行填充
            self.q_mask = tf.cast(self.input_x1, tf.bool)
            if trainable:
                self.c_maxlen, self.q_maxlen, = config.para_limit, config.ques_limit,
            else:
                self.c_maxlen, self.q_maxlen = config.test_para_limit, config.test_ques_limit

            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars1 = tf.nn.embedding_lookup(self.W, self.input_x1)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            self.embedded_chars_expanded1 = tf.expand_dims(self.embedded_chars1, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
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
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
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
