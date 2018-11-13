import jieba.posseg as pseg
import jieba
import jieba.analyse


def jieba_pos(sentence):
    words_pos=pseg.cut(sentence)
    word_pos=[]
    for w in words_pos:
        word_pos.append(w.flag)
    return word_pos

def jieba_extract_word(sen,tokkey=20):
    keywords=jieba.analyse.extract_tags(sen,topK=tokkey,withWeight=False)
    return keywords

if __name__ == '__main__':
    # jieba_pos("33")
    sen="武威公交一体化纪实 10家运输公司中标经营包括凉州区、古浪、民勤、天祝在内的城乡公交线路。经过收编、整合\
    、更新，开通城乡公交客运班线23条，统一投放80辆高档次客运车辆，由运输公司统一管理。实际上，运营在这些线路的新型\
    双开门公交车的标准、设施已远远超过城区公交车。武威运管部门通过市场竞争和行业引导，建立退出机制，规范经营行为，提升服务质量。\
     　　去年11月下旬，武威市区至古浪县城和凉州区50公里范围内的乡镇全部开通城乡公交，凉州区28个乡镇300个行政村更是全部覆盖城乡公\
     交，率先实现“乡乡通公交，村村通客车”。这些城乡公交定时、定班、定点、定线，城乡公交均等化延伸到农民的家门口。“乡村小公交起到了穿\
     针引线、走村串巷的功能。”沈兴国说。"
    jieba_extract_word(sen)