import tensorflow as tf
import pickle
import random
import jieba
from tqdm import tqdm
import json
import re
import math
import unicodedata
# import spacy
# import ujson as json
from collections import Counter
import numpy as np
import os.path
# import gensim
# # nlp = spacy.blank("en")
# model = gensim.models.Word2Vec.load('/home/chenmingwei/w2vModel/corpus.model')# shape is 300
# model2 = gensim.models.Word2Vec.load('/home/chenmingwei/w2vModel/train_vector/word2vec_gensim')# shape is 300
# model_char=gensim.models.Word2Vec.load('/home/chenmingwei/w2vModel/train_char_vec/word2vec_gensim') #shape is 100


def convert_idx(text, tokens):
    #返回对应单词的字符在原始文章中的起始位置以及结束位置
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current) #找到单词在text中的位置,起始位置
        if current < 0:
            print("Token {} cannot be found".format(token))
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans
def jieba_cut(words):
    return [ele.strip() for ele in "|".join(jieba.cut(words.strip())).split("|")]
def remove_unrelative(contexts,question,size):
    cut_biao='。|，|？'
    # candiate_sen=contexts.split('。')
    contexts1=re.split(cut_biao,contexts)
    if len(contexts1)==1:
        contexts_token="|".join(jieba.cut(contexts)).split("|")[:size]
        contexts_remove=','.join(contexts_token)
        return contexts_remove
    question="|".join(jieba.cut(question)).split("|")
    question_vec=np.zeros(400)
    for ele in question:
        try:
            question_vec+=np.array(model[ele])
        except:
            continue
    simlay_sentce={} #得到每个句子与问题的相似度
    for pas1 in contexts1:
        temp_vec=np.zeros(400)
        pas1_token="|".join(jieba.cut(pas1)).split("|")
        for ele in pas1_token:
            try:
                temp_vec+=model[ele]
            except:
                continue
        simla=cos_sine(question_vec,temp_vec)
        if math.isnan(simla):
            continue
        simlay_sentce[simla]=pas1
    relative_sentence=sorted(simlay_sentce.items(),key=lambda x:x[0],reverse=True) #对问句与问题之间相似度进行排序，相似度是采用求和得到
    # print(relative_sentence)
    context_len=0
    remove_context=''
    for similar,temp_context in relative_sentence: #按排序增加passage直到<等于size
        if context_len<size:
            remove_context+=(','+temp_context)
        context_len+=len(jieba_cut(temp_context))
    return remove_context
def cos_sine(all_pre, alter):
    pooled_len_spa1 = np.sqrt(np.sum(all_pre**2))  # 利用余弦相似度求解
    pooled_len_spa2 = np.sqrt(np.sum(alter**2))
    pooled_mul_spa12 = np.sum(all_pre*alter)  # 计算向量的点乘Batch模式
    # pooled_mul_13 = tf.reduce_sum(tf.multiply(pooled_flat_1, pooled_flat_3), 1)

    with tf.name_scope("output_spa"):
        cos_spa12 = pooled_mul_spa12/(pooled_len_spa1* pooled_len_spa2)

    return cos_spa12



def process_file(filename, data_type,word_counter, char_counter,mode='train',size=None):
    """
    :param filename:
    :param data_type: 我们先按照分类去做这件事情，类别分别是[肯定，否定，无法确定]三个类别去做
    :param word_counter: 产生单次级别的字典，频数对应段落单词对应多少个问答对以及加上该单词在问题中的出现频数，
    :param char_counter: 产生字符级别字典，频数对应多应多少个问答对以及在问题中出现的次数
    :return:size 为限定单词个数
    """
    print("Generating {} examples...".format(data_type))
    fou_word = ['不', '没', '否','无']
    unsure=['无法确定','无法确认','不确定','不清楚','无法确对','无法选择','无法']
    examples = []
    total = 0
    k = 0
    y1_temp=0;
    y2_temp=0;
    y3_temp=0;

    data_number=[0,0,0]
    count=0
    with open(filename, "r") as fh:
        lines=fh.readlines()
        for line in tqdm(lines):
            try:
                source = json.loads(line) #加载训练数据集
                context = source["passage"]
                context_tokens = jieba_cut(context)#把段落对应的单词进行分词[one ,..., htrow]
                if len(context_tokens)>size:
                    context_tokens=remove_unrelative(context,source["query"],size)
                    context_tokens =jieba_cut(context_tokens)
                context_chars = [list(token) for token in context_tokens]#把单词转化为字符
                for token in context_tokens:#对于段落中的每一个单词
                    word_counter[token] +=1 #对应上下文的contenxt的每一个单词对应多少个问题以及对应的答案，并且词汇就是所有单词的字典
                    for char in token:
                        char_counter[char] += 1#对应字符位置
                    # for qa in para["qas"]: #对于段落里的每一个问题和答案
                ques = source["query"]
                ques_tokens = "|".join(jieba.cut(ques)).split("|")
                ques_chars = [list(token) for token in ques_tokens]
                for token in ques_tokens:
                    word_counter[token] += 1
                    for char in token:
                        char_counter[char] += 1

                ys1=np.zeros([3],dtype=np.float32)#真实的三个类别，肯定，否定，无法确定
                answer=source['answer'].strip()

                alternatives=[ele.strip() for ele in source['alternatives'].strip().split('|')]
                #对候选集排序，最短字符在第一个，否定第二个，无法确定第三个
                new_alternative=[]
                min_word = 'dfsdfdsafdsaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
                for temp_alter in alternatives:
                    if len(temp_alter) < len(min_word):
                        min_word = temp_alter
                flag = False
                new_alternative.append(min_word)
                for temp_alter in alternatives:
                    if temp_alter.replace(min_word, '') == '不' or temp_alter.replace(min_word, '') == '没':
                        new_alternative.append(temp_alter)
                        flag = True
                        break
                if flag==True:
                    for temp_alter in alternatives:
                        if temp_alter not in new_alternative:
                            new_alternative.append(temp_alter)
                            break
                    for k in range(len(new_alternative)):
                        if new_alternative[k]==answer:
                            ys1[k]=1
                if np.sum(ys1)==1:
                    if ys1[0]==1:
                        y1_temp+=1
                        data_number[0]+=1
                    elif ys1[1]==1:
                        y2_temp+=1
                        data_number[1]+=1
                    elif ys1[2]==1:
                        y3_temp+=1
                        data_number[2]+=1
                    # if ((y1_temp>=22000 and ys1[0]==1) or (y2_temp>=22000 and ys1[1]==1) or (y3_temp>=22000 and ys1[2]==1)) and mode=='train':
                    #     continue
                    # else:
                    example = {"context_tokens": context_tokens, "context_chars": context_chars, "ques_tokens": ques_tokens, "alternatives_tokens":np.array(new_alternative),
                               "ques_chars": ques_chars,"y1s":ys1, "id": total} #每一个样本对应的上下文以及对应的字符问题分词，对应答案下标
                    total += 1
                    examples.append(example)
            except:
                k+=1
        print("all the data",len(lines) )
        print('the number of over size is',total)
        random.shuffle(examples)
        print("the data balance is :",data_number)
        print("{} questions in total".format(len(examples)))
    return examples
def process_file_test(filename, data_type, word_counter, char_counter,size=500):
    """
    :param filename:
    :param data_type: 我们先按照分类去做这件事情，类别分别是[肯定，否定，无法确定]三个类别去做
    :param word_counter: 产生单次级别的字典，频数对应段落单词对应多少个问答对以及加上该单词在问题中的出现频数，
    :param char_counter: 产生字符级别字典，频数对应多应多少个问答对以及在问题中出现的次数
    :return:
    """
    print("Generating {} examples...".format(data_type))
    examples = []
    total = 0
    k = 0
    y_temp1 = 0;
    y_temp2 = 0;
    y_temp3 = 0
    count=0
    data_number=[0,0,0]
    with open(filename, "r") as fh:
        lines=fh.readlines()
        for line in tqdm(lines):
            try:
                source = json.loads(line) #加载训练数据集
                context = source["passage"]
                context_tokens = "|".join(jieba.cut(context)).split("|")#把段落对应的单词进行分词[one ,..., htrow]
                if len(context_tokens)>size:
                    count+=1
                    context_tokens=remove_unrelative(context,source["query"],size)
                    context_tokens = "|".join(jieba.cut(context_tokens)).split("|")
                y_temp1+=len(context_tokens)
                context_chars = [list(token) for token in context_tokens]#把单词转化为字符
                for token in context_tokens:#对于段落中的每一个单词
                    word_counter[token] +=1 #对应上下文的contenxt的每一个单词对应多少个问题以及对应的答案，并且词汇就是所有单词的字典
                    for char in token:
                        char_counter[char] += 1#对应字符位置
                    # for qa in para["qas"]: #对于段落里的每一个问题和答案
                ques = source["query"]
                ques_tokens = "|".join(jieba.cut(ques)).split("|")
                y_temp2+=len(ques_tokens)
                ques_chars = [list(token) for token in ques_tokens]
                for token in ques_tokens:
                    word_counter[token] += 1
                    for char in token:
                        char_counter[char] += 1
                alternatives = [ele.strip() for ele in source['alternatives'].strip().split('|')]
                # 对候选集排序，最短字符在第一个，否定第二个，无法确定第三个
                new_alternative = []
                min_word = 'dfsdfdsafdsaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
                for temp_alter in alternatives:
                    if len(temp_alter) < len(min_word):
                        min_word = temp_alter
                flag = False
                new_alternative.append(min_word)
                for temp_alter in alternatives:
                    if temp_alter.replace(min_word, '') == '不' or temp_alter.replace(min_word, '') == '没':
                        new_alternative.append(temp_alter)
                        flag = True
                        break
                for temp_alter in alternatives:
                    if temp_alter not in new_alternative:
                        new_alternative.append(temp_alter)


                example = {"context_tokens": context_tokens, "context_chars": context_chars, "ques_tokens": ques_tokens,"ques_chars": ques_chars,
                           "alternatives_tokens":np.array(new_alternative)} #每一个样本对应的上下文以及对应的字符问题分词，对应答案下标
                total += 1
                examples.append(example)
                    # eval_examples[str(total)] = {
                    #     "context": context, "spans": spans, "answers": answer_texts, "uuid": qa["id"]}
            except:
                k+=1

        print('the number of overrate is ',total)
        print("traindata_of lables",data_number)
        print(y_temp1/len(examples), y_temp2/len(examples), y_temp3/len(examples*3),"yyyyyyyyyyyyyyyyyyyyyy")
        print("{} questions in total".format(len(examples)))
    return examples
def get_embedding_dict(emb_file=None,size=None, vec_size=None):
    all_embedding = {}
    with open(emb_file, "r", encoding="utf-8") as fh:
        k_step = 0
        for line in tqdm(fh):
            if k_step == 0:
                k_step += 1
                continue
            array = line.split()
            word = "".join(array[0:-vec_size])
            vector = list(map(float, array[-vec_size:]))
            all_embedding[word] = vector
    return all_embedding

def get_embedding(counter, data_type, limit=-1, model=None,emb_file=None, emb_file1=None,size=None, vec_size=None, token2idx_dict=None):
    #返回对应的单词embeeding，以及单词对应的下标
    print("Generating {} embedding...".format(data_type))
    embedding_dict = {}
    filtered_elements = [k for k, v in counter.items() if v > limit]
    if True:
        print("use preembedding")
        k_not_embedding=0
        # for token in filtered_elements:
        #     try:
        #         # if len(list(model1111111[token]))!=400:
        #         #     print("gggggggggggggggggggggggggggggggggggggggggggggggggggggggg")
        #         embedding_dict[token]=list(model[token])
        #     except:
        #         k_not_embedding+=1
        #         embedding_dict[token]= [np.random.normal(scale=0.1) for _ in range(vec_size)]
        # assert size is not None
        # assert vec_size is not None
        for token in filtered_elements:
            if token in emb_file:
                embedding_dict[token]=emb_file[token]
            elif token in emb_file1:
                embedding_dict[token]=emb_file1[token]
            else:
                k_not_embedding+=1
                embedding_dict[token] = [np.random.normal(scale=0.1) for _ in range(vec_size)]
        print(k_not_embedding,"not in the embedding")
        print("{} / {} tokens have corresponding {} embedding vector".format(
            len(embedding_dict), len(filtered_elements), data_type))
    else:
        print("use random embedding")
        assert vec_size is not None
        for token in filtered_elements:
            embedding_dict[token] = [np.random.uniform(-0.1,0.1) for _ in range(vec_size)]
        print("{} tokens have corresponding embedding vector".format(
            len(filtered_elements)))

    NULL = "--NULL--"
    OOV = "--OOV--"
    token2idx_dict = {token: idx for idx, token in enumerate(
        embedding_dict.keys(), 2)} if token2idx_dict is None else token2idx_dict   #把字典以及对应的下标进行对应
    token2idx_dict[NULL] = 0
    token2idx_dict[OOV] = 1
    embedding_dict[NULL] = [0. for _ in range(vec_size)]
    embedding_dict[OOV] = [0. for _ in range(vec_size)]
    idx2emb_dict = {idx: embedding_dict[token]
                    for token, idx in token2idx_dict.items()}  #返回对应的单词embeeding，以及单词对应的下标
    emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]
    return emb_mat, token2idx_dict


def build_features(config, examples, data_type, out_file, word2idx_dict, char2idx_dict, is_test=False):

    para_limit = config.test_para_limit if is_test else config.para_limit #段落的最大长度
    ques_limit = config.test_ques_limit if is_test else config.ques_limit #question最大长度
    char_limit = config.char_limit

    def filter_func(example, is_test=False):
        return len(example["context_tokens"]) > para_limit or len(example["ques_tokens"]) > ques_limit

    print("Processing {} examples...".format(data_type))
    writer = tf.python_io.TFRecordWriter(out_file) #存储处理好的数据的对应的文件字典
    total = 0
    total_ = 0
    meta = {}
    for example in tqdm(examples):
        total_ += 1

        # if filter_func(example, is_test):
        #     continue
        total += 1
        context_idxs = np.zeros([para_limit], dtype=np.int32) #对应的paragraph的单词下标初始化，最大为para_limit
        context_char_idxs = np.zeros([para_limit, char_limit], dtype=np.int32)#每一个对应段落的字符初始化[400,8]
        ques_idxs = np.zeros([ques_limit], dtype=np.int32)#], dtype=np.int32)#每一个对应段落的字符初始化
        ques_char_idxs = np.zeros([ques_limit, char_limit], dtype=np.int32)

        def _get_word(word):
            for each in (word, word.lower(), word.capitalize(), word.upper()):
                if each in word2idx_dict:
                    return word2idx_dict[each]
            return 1

        def _get_char(char):
            if char in char2idx_dict:
                return char2idx_dict[char]
            return 1
        for i, token in enumerate(example["context_tokens"]): #duying把当前训练数据集context转化为下标,也就是每一个单词对应到了embedding
            if i<para_limit:
                context_idxs[i] = _get_word(token)

        for i, token in enumerate(example["ques_tokens"]):
            if i<ques_limit:
                ques_idxs[i] = _get_word(token)


        for i, token in enumerate(example["context_chars"]):
            if i < para_limit:
                for j, char in enumerate(token):
                    if j == char_limit:
                        break
                    context_char_idxs[i, j] = _get_char(char)

        for i, token in enumerate(example["ques_chars"]):
            if i<ques_limit:
                for j, char in enumerate(token):
                    if j == char_limit:
                        break
                    ques_char_idxs[i, j] = _get_char(char)

        # start, end = example["y1s"][-1], example["y2s"][-1] #对应答案的标记
        if is_test:
            # y1[start], y2[end] = 1.0, 1.0  #对应答案在paragraph中的位置
            record = tf.train.Example(features=tf.train.Features(feature={
                "context_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[context_idxs.tostring()])), # 对应单词的下标位置
                "ques_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ques_idxs.tostring()])),
                "context_char_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[context_char_idxs.tostring()])),
                "ques_char_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ques_char_idxs.tostring()])),
                "alternatives_tokens": tf.train.Feature(bytes_list=tf.train.BytesList(value=[example['alternatives_tokens'].tostring()]))
            }))
        else:
            y1=example['y1s']
            # y1[start], y2[end] = 1.0, 1.0  #对应答案在paragraph中的位置
            record = tf.train.Example(features=tf.train.Features(feature={
                                      "context_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[context_idxs.tostring()])), #对应单词的下标位置
                                      "ques_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ques_idxs.tostring()])),
                                      "context_char_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[context_char_idxs.tostring()])),
                                      "ques_char_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ques_char_idxs.tostring()])),
                                      "alternatives_tokens": tf.train.Feature(bytes_list=tf.train.BytesList(value=[example['alternatives_tokens'].tostring()])),
                                      "y1": tf.train.Feature(bytes_list=tf.train.BytesList(value=[y1.tostring()])),#对应的实际答案在paragraph中的位置
                                      "id": tf.train.Feature(int64_list=tf.train.Int64List(value=[example["id"]]))
                                      }))
        writer.write(record.SerializeToString())
    print("Build {} / {} instances of features in total".format(total, total_))
    meta["total"] = total #共有多少个训练数据集
    writer.close()
    return meta


def save(filename, obj, message=None):
    if message is not None:
        print("Saving {}...".format(message))
        with open(filename, "w") as fh:
            json.dump(obj, fh)


def prepro(config):
    #准备好数据，把数据转化为下标以及对应的embedding
    word_counter, char_counter = Counter(), Counter()
    # 返回训练数据集 example = {"context_tokens": context_tokens#切词后的数据, "context_chars"切词后字符数据: context_chars,
    # "ques_tokens": ques_tokens,
    # "ques_chars": ques_chars, "y1s": y1s, "id": total} #每一个样本对应的上下文以及对应的字符问题分词，对应答案下标
    train_examples = process_file(config.train_file, "train", word_counter, char_counter,mode='train',size=config.para_limit)
    print(train_examples[0])
    dev_examples= process_file(config.dev_file, "dev", word_counter, char_counter,mode='dev',size=config.para_limit) #同样的处理
    print(dev_examples[0])
    test_examples = process_file_test(config.test_file, "test", word_counter, char_counter,size=config.para_limit)
    print(len(test_examples))
    print("444444444444444444444444444444")
    word_emb_file = config.fasttext_file if config.fasttext else config.glove_word_file#单词embedding
    word_emb_file1 = config.fasttext_file if config.fasttext else config.glove_word_file1#单词embedding

    char_emb_file = config.glove_char_file if config.pretrained_char else config.glove_char_file #在这里为false
    char_emb_size = config.glove_char_size if config.pretrained_char else None
    char_emb_dim = config.glove_dim if config.pretrained_char else config.char_dim#这里字符编码为8位

    word2idx_dict = None   #获取字符对应的embedding以及对应的字符对应下标
    if os.path.isfile(config.word2idx_file):
        with open(config.word2idx_file, "r") as fh:
            word2idx_dict = json.load(fh)
    #获取单次级别的词向量以及对应的单词对应下标,此时word_count已经有了训练数据集的数据
    #返回单词对应的embedding以及对应单词的下标,其中get_embedding包含如果里边没有对应的词向量就随机初始化一个
    allembeeding = get_embedding_dict(word_emb_file, size=config.glove_word_size, vec_size=config.glove_dim)

    allembeeding1 = get_embedding_dict(word_emb_file1, size=config.glove_word_size, vec_size=config.glove_dim)

    word_emb_mat, word2idx_dict = get_embedding(word_counter, "word", emb_file=allembeeding,emb_file1=allembeeding1,
                                                size=config.glove_word_size, vec_size=config.glove_dim,
                                                token2idx_dict=word2idx_dict)
    print("99999999999999999999999")
    char2idx_dict = None
    if os.path.isfile(config.char2idx_file):
        with open(config.char2idx_file, "r") as fh:
            char2idx_dict = json.load(fh)
    #获取字符对应的embedding以及对应的字符对应下标
    char_emb_mat, char2idx_dict = get_embedding(
        char_counter, "char", emb_file=allembeeding,emb_file1=allembeeding1, size=char_emb_size, vec_size=char_emb_dim,
        token2idx_dict=char2idx_dict)
    build_features(config, train_examples, "train",
                   config.train_record_file, word2idx_dict, char2idx_dict)
    dev_meta = build_features(config, dev_examples, "dev",
                              config.dev_record_file, word2idx_dict, char2idx_dict)
    test_meta = build_features(config, test_examples, "test",
                               config.test_record_file, word2idx_dict, char2idx_dict, is_test=True)
    print(test_meta)
    print('finsh_oit')
    # # save(config.word_emb_file, word_emb_mat, message="word embedding")
    pickle.dump(word_emb_mat,open(config.word_emb_file,'wb'), pickle.HIGHEST_PROTOCOL)
    print("wwwwwwwwwwwwwwwwwwwwwwww")
    # save(config.char_emb_file, char_emb_mat, message="char embedding")
    pickle.dump(char_emb_mat,open(config.char_emb_file,'wb'), pickle.HIGHEST_PROTOCOL)
    print("eeeeeeeeeeeeeeeeeeeeeeeeeeee")
    # # save(config.train_eval_file, train_eval, message="train eval")
    # # save(config.dev_eval_file, dev_eval, message="dev eval")
    # # save(config.test_eval_file, test_eval, message="test eval")
    save(config.dev_meta, dev_meta, message="dev meta")
    print("rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr")

    save(config.word2idx_file, word2idx_dict, message="word2idx")
    print("ttttttttttttttttttttttttttttttt")
    save(config.char2idx_file, char2idx_dict, message="char2idx")
    print("gggggggggggggggggggggggggggggggggggggg")
    save(config.test_meta, test_meta, message="test meta")
