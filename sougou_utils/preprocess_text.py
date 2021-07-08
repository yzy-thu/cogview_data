import lmdb
import jsonlines
from tqdm import tqdm
from multiprocessing import Process

def main(N, pid):
    print("test_equal: ", hash("test_equal"))
    path0 = "/dataset/58e8b681/ordinary/lmdb/text_lmdb_"
    env = [0, 0, 0, 0, 0, 0]
    for i in range(6):
        env[i] = lmdb.open(f"/dataset/58e8b681/ordinary/lmdb/text_lmdb_{i}", lock=False, readonly=True)
    path = "/dataset/58e8b681/ordinary/imageid_6_split.jsonl"
    text_path = "/dataset/fd5061f6/sougou/text/"
    jsonl = jsonlines.open(path, "r")
    cnt = 0
    pre_bin_name = ""
    file = None
    now_process = False
    cnt = 0
    for data in tqdm(jsonl):
        now_bin_name = data["bin_name"]
        if pre_bin_name != now_bin_name:
            pre_bin_name = now_bin_name
            if hash(now_bin_name) % N == pid:
                if file is not None:
                    file.close()
                print(f"process{pid} finish {cnt}")
                cnt += 1
                file = open(text_path + now_bin_name + ".txt", "a")
                now_process = True
            else:
                now_process = False
        if not now_process:
            continue
        id = data["id"]
        lmdb_id = int(data["lmdb_id"])
        txn = env[lmdb_id].begin()
        value_ch = txn.get(id.encode('utf-8')).decode('utf-8')
        value_en = txn.get(str(id + "_en").encode("utf-8")).decode('utf-8')
        # breakpoint()
        value_ch = value_ch.replace("\t", " ")
        value_en = value_en.replace("\t", " ")
        file.write(id + "\t" + value_ch + "\t" + value_en + "\n")

    if file is not None:
        file.close()
    for i in range(6):
        env[i].close()
    pass


if __name__ == "__main__":
    process_list = []
    N = 140
    for i in range(N):  # 开启5个子进程执行fun1函数
        p = Process(target=main, args=(N, i,))  # 实例化进程对象
        p.start()
        process_list.append(p)

    for p in process_list:
        p.join()

    print('finish')

    pass