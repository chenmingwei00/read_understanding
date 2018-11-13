import tensorflow as tf
# import ujson as json
import numpy as np
from tqdm import tqdm
import pickle
import os
import json
from lstm_model_addlstm import Model
from util import get_record_parser, convert_tokens, evaluate, get_batch_dataset, get_dataset
import matplotlib.pyplot as plt
import numpy as np


def train(config):
    with open(config.word_emb_file, "rb") as fh:
        print(fh)
        word_mat = np.array(pickle.load(fh), dtype=np.float32) #加载对应的单词以及词向量,加载字符以及对应的词向量
    with open(config.char_emb_file, "rb") as fh:
        char_mat = np.array(pickle.load(fh), dtype=np.float32)#加载字符向量对应的
    # with open(config.train_eval_file, "r") as fh:
    #     train_eval_file = json.load(fh) #"context": context, "spans": spans, "answers": answer_texts, "uuid": qa["id"]
    #                                    #span 对应的段落单词字符开始结束位置 context:全文内容 ,answer 实际词汇
    # with open(config.dev_eval_file, "r") as fh:
    #     dev_eval_file = json.load(fh)
    with open(config.dev_meta, "r") as fh:
        meta = json.load(fh)
    # dev_total = meta["total"]
    #
    print("Building model1111111...")
    parser = get_record_parser(config)#加载对应的训练数据特征,与存储特征数据的build_features对应，
    train_dataset = get_batch_dataset(config.train_record_file, parser, config) #返回已经batch好的训练数据集，用iterator进行迭代
    dev_dataset = get_dataset(config.dev_record_file, parser, config) #对这个数据进行同样的处理
    # #把不同的数据dataset的数据feed进模型，首先需要建立iterator handle，既iterator placeholder
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)  #
    train_iterator = train_dataset.make_one_shot_iterator()#首先提取数据，先对数据构建迭代器iterator
    dev_iterator = dev_dataset.make_one_shot_iterator()

    model = Model(config, iterator, word_mat=word_mat,char_mat=char_mat,filter_sizes=[3,4,5], embedding_size=300,num_filters=128)

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    config_sess=tf.ConfigProto(gpu_options=gpu_options)
    loss_save = 0
    patience = 0
    lr = config.init_lr

    with tf.Session(config=config_sess) as sess:
        fig = plt.figure()
        ax = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        plt.grid(True)
        plt.ion()
        writer = tf.summary.FileWriter(config.log_dir)
        ckpt = tf.train.get_checkpoint_state(config.save_dir)
        if ckpt is not None:
            print(ckpt.model_checkpoint_path)
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())
        train_handle = sess.run(train_iterator.string_handle())
        dev_handle = sess.run(dev_iterator.string_handle())


        for _ in tqdm(range(1, config.num_steps + 1)):
            try:
                global_step = sess.run(model.global_step) + 1
                logits,loss,train_op,accuracy= sess.run([model.predictions,model.loss,model.train_op,model.accuracy],
                                                     feed_dict={handle: train_handle})
                # print(logits)
                # print(np.array(real_y),"hhhhhhhhhhhhhhhhhhhhhhhh")
                if global_step%10==0:
                    ax.scatter(global_step, loss, c='b', marker='.')
                    plt.pause(0.001)
                print("the loss is:",loss)
                print('the accuracy is',accuracy)
                if global_step % config.period == 0:
                    loss_sum = tf.Summary(value=[tf.Summary.Value(tag="model/loss", simple_value=loss)])
                    writer.add_summary(loss_sum, global_step)
                if global_step % config.checkpoint == 0:
                    # sess.run(tf.assign(model.is_train,tf.constant(False, dtype=tf.bool)))
                    dev_loss=0
                    for k in range(500):
                        dev_loss+= sess.run([model.accuracy],feed_dict={handle:dev_handle})[0]
                    dev_loss=dev_loss/500
                    ax2.scatter(global_step, dev_loss, c='b', marker='.')
                    plt.pause(0.001)


                    # _, summ = evaluate_batch(
                    #     model1111111, config.val_num_batches, train_eval_file, sess, "train", handle, train_handle)
                    # for s in summ:
                    #     writer.add_summary(s, global_step)

                    # metrics, summ = evaluate_batch(
                    #     model1111111, dev_total // config.batch_size + 1, dev_eval_file, sess, "dev", handle, dev_handle)
                    # sess.run(tf.assign(model.is_train,tf.constant(True, dtype=tf.bool)))

                    # dev_loss = metrics["loss"]
                    if dev_loss > loss_save:
                        print(dev_loss,loss_save,'222222222222222222222222222222222222222222222222222222222222222')
                        loss_file=os.path.join(config.save_dir,'loss_file.txt')
                        with open(loss_file,'w') as fi:
                            fi.write(str(dev_loss))
                            fi.write('\t')
                            fi.write(str(loss_save))
                            fi.write('\n')
                        loss_save = dev_loss
                        filename = os.path.join(
                            config.save_dir, "model_{}.ckpt".format(global_step))
                        model.saver.save(sess, filename)
                        figure_path=os.path.join(config.save_dir,'img.png')
                        print(figure_path,"ttttttttttttttttttttttttttttttttttt")
                        plt.savefig(figure_path)
                    # sess.run(tf.assign(model.lr, tf.constant(lr, dtype=tf.float32)))
                    # for s in summ:
                    loss_sum = tf.Summary(value=[tf.Summary.Value(tag="model/loss_dev", simple_value=dev_loss)])
                    writer.add_summary(loss_sum, global_step)
                    writer.flush()
                    # filename = os.path.join(config.save_dir, "model_{}.ckpt".format(global_step))
                    # model.saver.save(sess, filename)
            except Exception as e:
                print(e)



def evaluate_batch(model, num_batches, eval_file, sess, data_type, handle, str_handle):
    answer_dict = {}
    losses = []
    for _ in tqdm(range(1, num_batches + 1)):
        qa_id, loss, yp1, yp2, = sess.run(
            [model.qa_id, model.loss, model.yp1, model.yp2], feed_dict={handle: str_handle})
        answer_dict_, _ = convert_tokens(
            eval_file, qa_id.tolist(), yp1.tolist(), yp2.tolist())
        answer_dict.update(answer_dict_)
        losses.append(loss)
    loss = np.mean(losses)
    metrics = evaluate(eval_file, answer_dict)
    metrics["loss"] = loss
    loss_sum = tf.Summary(value=[tf.Summary.Value(
        tag="{}/loss".format(data_type), simple_value=metrics["loss"]), ])
    f1_sum = tf.Summary(value=[tf.Summary.Value(
        tag="{}/f1".format(data_type), simple_value=metrics["f1"]), ])
    em_sum = tf.Summary(value=[tf.Summary.Value(
        tag="{}/em".format(data_type), simple_value=metrics["exact_match"]), ])
    return metrics, [loss_sum, f1_sum, em_sum]


def test(config):
    questionid=[]#对应的顺序questionid
    new_alternatives=[] #更改之后的答案顺序
    new_answer_index=[] #真正答案的下标
    with open(config.test_file, "r") as fh:
        lines = fh.readlines()
        for line in tqdm(lines):
            new_alternative=[]
            source = json.loads(line)  # 加载训练数据集
            questionid.append(source['query_id'])
            alter_answer_origanl=[ele.strip() for ele in source['alternatives'].strip().split("|")]
            min_word = 'dfsdfdsafdsaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
            for temp_alter in alter_answer_origanl:
                if len(temp_alter) < len(min_word):
                    min_word = temp_alter
            new_alternative.append(min_word)
            for temp_alter in alter_answer_origanl:
                if temp_alter.replace(min_word, '') == '不' or temp_alter.replace(min_word, '') == '没':
                    new_alternative.append(temp_alter)
                    break
            for temp_alter in alter_answer_origanl:
                if temp_alter not in new_alternative:
                    new_alternative.append(temp_alter)
            new_alternative=np.array(new_alternative)
            new_alternatives.append(new_alternative)

    with open(config.word2idx_file, "r") as fh:
        word2idx_dict = json.load(fh)
        index_word=dict(zip(word2idx_dict.values(),word2idx_dict.keys()))
    with open(config.word_emb_file, "rb") as fh:
        word_mat = np.array(pickle.load(fh), dtype=np.float32)  # 加载对应的单词以及词向量,加载字符以及对应的词向量
    with open(config.char_emb_file, "rb") as fh:
        char_mat = np.array(pickle.load(fh), dtype=np.float32)  # 加载字符向量对应的
    # with open(config.test_eval_file, "r") as fh:
    #     eval_file = json.load(fh)
    with open(config.test_meta, "r") as fh:
        meta = json.load(fh)

    total = meta["total"]

    print("Loading model1111111...")
    # parser = get_record_parser(config,is_test=True)#加载对应的训练数据特征,与存储特征数据的build_features对应，
    # test_batch = get_dataset(config.test_record_file, parser, config)
    test_batch = get_dataset(config.test_record_file, get_record_parser(
        config, is_test=True), config).make_one_shot_iterator()

    model = Model(config, test_batch, word_mat=word_mat,char_mat=char_mat,filter_sizes=[3,4,5],trainable=False, embedding_size=300,num_filters=128)

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(config.save_dir))
        # sess.run(tf.assign(model.is_train, tf.constant(True, dtype=tf.bool)))
        for step in tqdm(range(total // config.batch_size+1)):
            ture_pre= sess.run(model.predictions) #qa_id 预测对应的类别下标
            new_answer_index.extend(ture_pre) #不改变顺序
        print(len(new_answer_index))
        with open(config.answer_file, "w") as fh:
            normal=0
            for ele_k in range(len(questionid)):
                alter1=new_alternatives[ele_k]
                if new_answer_index[ele_k]>=len(alter1):
                    ture_answer=alter1[0]
                else:
                    normal+=1
                    ture_answer=alter1[new_answer_index[ele_k]]
                ture_queid=str(questionid[ele_k])
                fh.write(ture_queid)
                fh.write('\t')
                fh.write(ture_answer)
                fh.write('\n')
            print('the normal quiestion is:',normal)
