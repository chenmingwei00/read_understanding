import tensorflow as tf

INF = 1e30


class cudnn_gru:

    def __init__(self, num_layers, num_units, batch_size, input_size, keep_prob=1.0, is_train=None, scope=None):
        self.num_layers = num_layers
        self.grus = []
        self.inits = []
        self.dropout_mask = []
        for layer in range(num_layers):
            input_size_ = input_size if layer == 0 else 2 * num_units
            gru_fw = tf.contrib.cudnn_rnn.CudnnGRU(1, num_units)
            gru_bw = tf.contrib.cudnn_rnn.CudnnGRU(1, num_units)
            init_fw = tf.tile(tf.Variable(
                tf.zeros([1, 1, num_units])), [1, batch_size, 1])
            init_bw = tf.tile(tf.Variable(
                tf.zeros([1, 1, num_units])), [1, batch_size, 1])
            mask_fw = dropout(tf.ones([1, batch_size, input_size_], dtype=tf.float32),
                              keep_prob=keep_prob, is_train=is_train, mode=None)
            mask_bw = dropout(tf.ones([1, batch_size, input_size_], dtype=tf.float32),
                              keep_prob=keep_prob, is_train=is_train, mode=None)
            self.grus.append((gru_fw, gru_bw, ))
            self.inits.append((init_fw, init_bw, ))
            self.dropout_mask.append((mask_fw, mask_bw, ))

    def __call__(self, inputs, seq_len, keep_prob=1.0, is_train=None, concat_layers=True):
        outputs = [tf.transpose(inputs, [1, 0, 2])]
        for layer in range(self.num_layers):
            gru_fw, gru_bw = self.grus[layer]
            init_fw, init_bw = self.inits[layer]
            mask_fw, mask_bw = self.dropout_mask[layer]
            with tf.variable_scope("fw_{}".format(layer)):
                out_fw, _ = gru_fw(
                    outputs[-1] * mask_fw, initial_state=(init_fw, ))
            with tf.variable_scope("bw_{}".format(layer)):
                inputs_bw = tf.reverse_sequence(
                    outputs[-1] * mask_bw, seq_lengths=seq_len, seq_dim=0, batch_dim=1)
                out_bw, _ = gru_bw(inputs_bw, initial_state=(init_bw, ))
                out_bw = tf.reverse_sequence(
                    out_bw, seq_lengths=seq_len, seq_dim=0, batch_dim=1)
            outputs.append(tf.concat([out_fw, out_bw], axis=2))
        if concat_layers:
            res = tf.concat(outputs[1:], axis=2)
        else:
            res = outputs[-1]
        res = tf.transpose(res, [1, 0, 2])
        return res


class native_gru:

    def __init__(self, num_layers, num_units, batch_size, input_size, keep_prob=1.0, is_train=None, scope="native_gru"):
        self.num_layers = num_layers
        self.grus = []
        self.inits = []
        self.dropout_mask = []
        self.scope = scope
        for layer in range(num_layers):
            input_size_ = input_size if layer == 0 else 2 * num_units
            gru_fw = tf.contrib.rnn.GRUCell(num_units) #神经元的个数
            gru_bw = tf.contrib.rnn.GRUCell(num_units)
            init_fw = tf.tile(tf.Variable(
                tf.zeros([1, num_units])), [batch_size, 1])
            init_bw = tf.tile(tf.Variable(
                tf.zeros([1, num_units])), [batch_size, 1])
            mask_fw = dropout(tf.ones([batch_size, 1, input_size_], dtype=tf.float32),
                              keep_prob=keep_prob, is_train=is_train, mode=None)
            mask_bw = dropout(tf.ones([batch_size, 1, input_size_], dtype=tf.float32),
                              keep_prob=keep_prob, is_train=is_train, mode=None)
            self.grus.append((gru_fw, gru_bw, ))
            self.inits.append((init_fw, init_bw, ))
            self.dropout_mask.append((mask_fw, mask_bw, ))

    def __call__(self, inputs, seq_len, keep_prob=1.0, is_train=None, concat_layers=True):
        #seq_len 每一个句子的实际长度
        outputs = [inputs]
        with tf.variable_scope(self.scope):
            for layer in range(self.num_layers):
                gru_fw, gru_bw = self.grus[layer] #找到第一层的初始化fw and bw
                init_fw, init_bw = self.inits[layer]
                mask_fw, mask_bw = self.dropout_mask[layer]
                with tf.variable_scope("fw_{}".format(layer)):
                    out_fw, _ = tf.nn.dynamic_rnn(
                        gru_fw, outputs[-1] * mask_fw, seq_len, initial_state=init_fw, dtype=tf.float32)
                with tf.variable_scope("bw_{}".format(layer)):
                    inputs_bw = tf.reverse_sequence(
                        outputs[-1] * mask_bw, seq_lengths=seq_len, seq_dim=1, batch_dim=0)
                    out_bw, _ = tf.nn.dynamic_rnn(
                        gru_bw, inputs_bw, seq_len, initial_state=init_bw, dtype=tf.float32) #每一个out_bw的输出维度[batch_size,sequnce_length,神经元个数（在这里是75）]
                    out_bw = tf.reverse_sequence(
                        out_bw, seq_lengths=seq_len, seq_dim=1, batch_dim=0)
                outputs.append(tf.concat([out_fw, out_bw], axis=2))#[batch_size,sequnce_length,150]
        if concat_layers:
            res = tf.concat(outputs[1:], axis=2) #self.num_layers is 3 ,so
        else:
            res = outputs[-1]
        return res


class ptr_net:
    def __init__(self, batch, hidden, keep_prob=1.0, is_train=None, scope="ptr_net"):
        self.gru = tf.contrib.rnn.GRUCell(hidden)
        self.cell = tf.contrib.rnn.GRUCell(hidden)
        self.initstate=self.cell.zero_state(batch_size=batch,dtype=tf.float32)
        self.batch = batch
        self.scope = scope
        self.keep_prob = keep_prob
        self.is_train = is_train
        self.dropout_mask = dropout(tf.ones(
            [batch, hidden], dtype=tf.float32), keep_prob=keep_prob, is_train=is_train)

    def __call__(self, init, match, d, mask):
        #init:[batch,d*2] match:[batch,context_len,d*2]
        with tf.variable_scope(self.scope):
            d_match = dropout(match, keep_prob=self.keep_prob,
                              is_train=self.is_train)

            output,state=tf.nn.dynamic_rnn(self.cell,d_match,initial_state=init)
            # # print(states,"ffffffffffffffffffff")
            # output, state = self.gru(d_match, init)
            batch_size,seq_len,embeding=output.get_shape().as_list()
            output=tf.transpose(output,[0,2,1])
            output=tf.reshape(output,[-1,seq_len])
            w=tf.Variable(tf.random_normal([output.get_shape().as_list()[-1],1]))
            b=tf.Variable(tf.random_normal([1]))
            all_pre=tf.nn.relu(tf.matmul(output,w)+b)
            all_pre=tf.reshape(all_pre,[batch_size,embeding])
            return state,all_pre


def dropout(args, keep_prob, is_train, mode="recurrent"):
    if keep_prob < 1.0:
        noise_shape = None
        scale = 1.0
        shape = tf.shape(args)
        if mode == "embedding":
            noise_shape = [shape[0], 1]
            scale = keep_prob
        if mode == "recurrent" and len(args.get_shape().as_list()) == 3:
            noise_shape = [shape[0], 1, shape[-1]]
        args = tf.cond(is_train, lambda: tf.nn.dropout(
            args, keep_prob, noise_shape=noise_shape) * scale, lambda: args)
    return args


def softmax_mask(val, mask):
    return -INF * (1 - tf.cast(mask, tf.float32)) + val  #这一句就是把True or False cast to 浮点类型数据，然后用1相减得到实际为零用极小值-INF替代


def pointer(inputs, state, hidden, mask, scope="pointer"):
    #input [batch,context_len,d*2]
    with tf.variable_scope(scope):
        u = tf.concat([tf.tile(tf.expand_dims(state, axis=1), [ #扩充在第二个维度上[batch,1,d*2]
            1, tf.shape(inputs)[1], 1]), inputs], axis=2)#然后按照context_len复制,按照第三个维度进行连接形成[batch_size,context_len,d*4]
        s0 = tf.nn.tanh(dense(u, hidden, use_bias=False, scope="s0")) #[batch_size,context_length,d]
        s = dense(s0, 3, use_bias=False, scope="s")#[batch,context_len,3]
        mask=tf.tile(tf.expand_dims(mask,axis=2),[1,1,3])

        s1=softmax_mask(s, mask)
        # s1 = softmax_mask(tf.squeeze(s, [2]), mask)#同样的处理，去除掉扩充的context words is [batch,context_len]
        # a = tf.expand_dims(tf.nn.softmax(s1), axis=2)#[batch,context_len,1]
        a=tf.nn.softmax(s1)
        a_weight=tf.expand_dims(tf.reduce_sum(a,axis=2),axis=2)
        res = tf.reduce_sum(a_weight * inputs, axis=1) #相当于对每一个context_len权重相乘,[batch,context_len,d*2]

        return res, s1


def summ(memory, hidden, mask, keep_prob=1.0, is_train=None, scope="summ"):
    with tf.variable_scope(scope):
        d_memory = dropout(memory, keep_prob=keep_prob, is_train=is_train)
        s0 = tf.nn.tanh(dense(d_memory, hidden, scope="s0"))
        s = dense(s0, 1, use_bias=False, scope="s")
        s1 = softmax_mask(tf.squeeze(s, [2]), mask)#tf.squeeze 把2维度的为1的去除掉，剩余[:,sequnece_length]的维度,然后转化为标记每一个句子实际的长度，其他扩充为一个极小值
        a = tf.expand_dims(tf.nn.softmax(s1), axis=2) #[batch_size,sequnece_len,1]
        res = tf.reduce_sum(a * memory, axis=1) #a实际是每一个passage单词的权重没相当于[batch,sequnce_len,150],按照sequen_len进行求和[batch_size,150]
        return res

def r_net_att(q,utp,vp):
    q_shape = q.get_shape().as_list()#[batch,q_len,demision]
    outj = []
    for k in range(q_shape[1]):
        sj=tf.nn.tanh(dense(q[:,k,:], 1, use_bias=False, scope="inputs_q_")+\
                      dense(utp, 1, use_bias=False, scope="inputs_up1")+ \
                      dense(vp, 1, use_bias=False, scope="inputs_v_1p"))
        outj.append(sj)
    outj=tf.concat(outj,axis=1)
    outj=tf.convert_to_tensor(outj)
    aj_all=tf.div(outj,tf.expand_dims(tf.reduce_sum(outj,axis=1),axis=-1)) #获取问题的单词权重
    aj_all=tf.expand_dims(aj_all,axis=-1)
    ct=tf.reduce_sum(tf.multiply(q,aj_all),axis=1)
    return ct
def dot_attention_rnet(inputs, memory, mask, hidden, keep_prob=1.0, is_train=None, scope="dot_attention"):

        with tf.variable_scope(scope):
            output_att=[]
            c_shape=inputs.get_shape().as_list()
            q_shape=memory.get_shape().as_list()
            V0p = tf.tile(tf.Variable(tf.zeros([1,q_shape[-1]])), [c_shape[0], 1])
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(q_shape[-1], forget_bias=0.0, state_is_tuple=False)
            # state_temp=lstm_cell.zero_state(c_shape[0])
            # print(state_temp,"2222")
            cell=tf.nn.rnn_cell.MultiRNNCell([lstm_cell]*2)
            # lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
            d_inputs = dropout(inputs, keep_prob=keep_prob, is_train=is_train)
            d_memory = dropout(memory, keep_prob=keep_prob, is_train=is_train)
            with tf.variable_scope("RNN"):
                for time_step in range(c_shape[1]):#表示passage每一个时刻t
                    if time_step>0:tf.get_variable_scope().reuse_variables()
                    if time_step == 0:
                        initate=cell.zero_state(c_shape[0],tf.float32)
                        ct=r_net_att(d_memory,d_inputs[:,time_step,:],V0p)
                        # ct=tf.concat([ct,ct],axis=-1)
                        # states_series, current_state = tf.contrib.rnn.static_rnn(cell, inputs_series,
                        #                                                          initial_state=init_state)
                        current_ouput,state_current=cell(ct,initate)
                        output_att.append(current_ouput)
                    else:
                        ct=r_net_att(d_memory,d_inputs[:,time_step,:],current_ouput)
                        current_ouput,state_current=cell(ct,state_current)
                        output_att.append(current_ouput)
            print(output_att)
            output_att=tf.reshape(output_att,[c_shape[0],c_shape[1],c_shape[2]])
            # output_att= tf.concat([inputs, output_att], axis=2)
            return output_att


def dot_attention(inputs, memory, mask, hidden, keep_prob=1.0, is_train=None, scope="dot_attention"):
    """
    :param inputs: c,
    :param memory:q,
    :param mask: self.q_mask
    :param hidden: hidden=d
    :param keep_prob:
    :param is_train:
    :param scope:
    :return:
    """
    with tf.variable_scope(scope):
        d_inputs = dropout(inputs, keep_prob=keep_prob, is_train=is_train)
        d_memory = dropout(memory, keep_prob=keep_prob, is_train=is_train)
        JX = tf.shape(inputs)[1]  #[batch c_maxlen, rnn_output]

        with tf.variable_scope("attention"):
            inputs_ = tf.nn.relu(
                dense(d_inputs, hidden, use_bias=False, scope="inputs")) #output[batch_size,sequncen_length,hidden]
            memory_ = tf.nn.relu(dense(d_memory, hidden, use_bias=False, scope="memory"))
            outputs = tf.matmul(inputs_, tf.transpose(memory_, [0, 2, 1])) / (hidden ** 0.5)  #这一句的意思就是paragraph与question分别对应的para_len与que_len 对应的embedding分别相乘形成矩阵
                                                                                              #[batch_size,para_len,que_len]对应
                                                                                             #矩阵形式，这样就是每一个句子对应的相似度
            mask = tf.tile(tf.expand_dims(mask, axis=1), [1, JX, 1])#tile主要是对张量进行扩充，扩充为JX个维度,也就是context的长度,而mask就是一个问题的实际长度
            logits = tf.nn.softmax(softmax_mask(outputs, mask))#作用有二：1、把为零的有很小的数值替代，防止一个vector全部为零的情况报错/2、找到实际问题的长度，其他设置为零，为句子长度的为1,3/然后在进行相似度归一化
            #[batch_size,para_len,que_len]
            #上面的logits就是每一个问题的单词对每一个context单词的权重
            outputs = tf.matmul(logits, memory)
            res = tf.concat([inputs, outputs], axis=2) #[batch_size,context_len,900]

        with tf.variable_scope("gate"):#用来决定，哪些神经元激活
            dim = res.get_shape().as_list()[-1]
            d_res = dropout(res, keep_prob=keep_prob, is_train=is_train)
            gate = tf.nn.sigmoid(dense(d_res, dim, use_bias=False))
            return res * gate


def dense(inputs, hidden, use_bias=True, scope="dense"):
    #整个函数利用权重，把多层lstm的输出转化为一层lstm的维度
    with tf.variable_scope(scope):
        shape = tf.shape(inputs) #tf.shape()中的数据类型可以是tensor，list，array
        dim = inputs.get_shape().as_list()[-1] #每一个输出的维度，也就是就是句子长度的每一个单词的维度
        out_shape = [shape[idx] for idx in range(len(inputs.get_shape().as_list()) - 1)] + [hidden]
        flat_inputs = tf.reshape(inputs, [-1, dim]) #把每一个单词的rnn的output输出转换为一个维度
        # with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        W = tf.get_variable("W", [dim, hidden])
        res = tf.matmul(flat_inputs, W)#变为[-1,hidden]维度
        if use_bias:
            b = tf.get_variable(
                "b", [hidden], initializer=tf.constant_initializer(0.))
            res = tf.nn.bias_add(res, b)
        res = tf.reshape(res, out_shape)
        return res
