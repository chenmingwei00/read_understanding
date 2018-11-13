import jieba.posseg as pseg
import jieba


def jieba_pos(sentence):
    words_pos=pseg.cut(sentence)
    word_pos=[]
    for w in words_pos:
        word_pos.append(w.flag)
    return word_pos



if __name__ == '__main__':
    file_train='/home/chenmingwei/nlp_comptetion/ai_challenger_oqmrc_trainingset_20180816/ai_challenger_oqmrc_trainingset.json'
    file_test='/home/chenmingwei/nlp_comptetion/ai_challenger_oqmrc_testa_20180816/ai_challenger_oqmrc_testa.json/'