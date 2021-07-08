'''
    sougou billion
    // 6亿
    图片：
        15980
        sougou -*.image.bin：图片压缩文件，共15950个。

        range_lmdb：每个图片id和对应的byte范围。
        0000031527dab302 [4200885377 4200966951]

        image_id_6_split.json：存放图片id和bin对应关系的json文件,即每张图片在哪个bin文件里。此外lmdb_id的key还指明其对应哪个text_lmdb文件
    文本：
        {"id": "706c53884531cbcc", "bin_name": "sougou-billion-bbc-06-p1-1063.image.bin", "lmdb_id": "0"}


        text_lmdb_0~5：存放图片id和对应的文本，文本经过翻译，包含中文和英文。共6个。
        000006c079fd9cc9 在 日晚五星体育的王牌节目 五星足球 节目中， 相聚申花 栏目播出了毛剑卿的生活故事，让我们了解了小毛在球场之外的另外一面。在主场对阵大连一方的比赛中，毛剑卿被对手撞伤膝盖，这位锋线大将不得不离开赛场一段日子，不过，这也给了平时顾不上家里的小毛一些宝贵的业
        000006c079fd9cc9_en In the five-star football program, the ace program of five-star sports on the evening of Sunday, the gathering Shenhua program broadcast the life story of Mao Jianqing, which let us know the other side of Xiaomao outside the stadium. In the home match against Dalian side, Mao Jianqing was injured in the knee by his opponent, and the striker had to leave the stadium for a period of time. However, this also gave Xiaomao, who was usually neglected at home, some precious industries
'''
import lmdb
import pickle
import os
def search(env, sid):
    txn = env.begin()
    data = txn.get(str(sid).encode('utf-8'))
    # data = pickle.loads(txn.get(str(sid).encode('utf-8')))
    return data

def display(env):
    txn = env.begin()
    cur = txn.cursor()
    cnt = 0
    for key, value in cur:
        cnt += 1
        if cnt == 100:
            break
        # print(int(key), int(value))
        import base64
        import codecs
        # print(codecs.decode(value))
        # print(base64.urlsafe_b64decode(value))
        import numpy as np
        print(key.decode('utf-8'), value.decode('utf-8'))
        # print(key.decode('utf-8'), np.frombuffer(value, dtype=np.uint64))
if __name__ == "__main__":
    path = "/workspace/yzy/sougou/range_lmdb"
    path2 = "/home/mingding/sougou/range_lmdb"
    path3 = "/dataset/58e8b681/ordinary/lmdb/text_lmdb_0"
    with lmdb.open(path3, lock=False, readonly=True) as env:
        display(env)
    pass