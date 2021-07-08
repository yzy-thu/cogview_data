import lmdb
import jsonlines
from tqdm import tqdm
from multiprocessing import Process
import numpy as np

def main(N, pid):
    print("test_equal: ", hash("test_equal"))
    path_json = "/dataset/58e8b681/ordinary/imageid_6_split.jsonl"
    id_path = "/dataset/fd5061f6/sougou/ids/"
    path = "/workspace/yzy/sougou/range_lmdb"
    env = lmdb.open(path, lock=False, readonly=False)
    jsonl = jsonlines.open(path_json, "r")
    cnt = 0
    pre_bin_name = ""
    file = None
    now_process = False
    cnt = 0
    bin_cnt = 0
    for data in tqdm(jsonl):
        now_bin_name = data["bin_name"]
        if pre_bin_name != now_bin_name:
            pre_bin_name = now_bin_name
            bin_cnt += 1
            if bin_cnt % N == pid:
                if file is not None:
                    file.close()
                print(f"process{pid} begin {bin_cnt}")
                file = open(id_path + now_bin_name + ".id", "a")
                now_process = True
            else:
                now_process = False
        if not now_process:
            continue
        id = data["id"]
        with env.begin(write=False) as txn:
            key = str(id).encode('utf-8')
            row = np.frombuffer(txn.get(key), dtype=np.uint64)
        file.write(id + "\t" + str(row[0]) + "\t" + str(row[1]) + '\n')
    if file is not None:
        file.close()
    pass


if __name__ == "__main__":
    process_list = []
    N = 140
    for i in range(N):
        p = Process(target=main, args=(N, i,))  # 实例化进程对象
        p.start()
        process_list.append(p)
    for p in process_list:
        p.join()
    print('finish')

    pass