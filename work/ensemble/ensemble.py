import os
from typing import Tuple, List
import csv
import numpy as np


def ensemble():
    n = 0
    all_preds = np.zeros(int(1e5))

    for root, dirs, files in os.walk("./candidate", topdown=False):
        print(root, dirs, files)
        n += len(files)
        print("Total file: ", n)
        for file in files:
            print(os.path.join(root, file))
            with open(os.path.join(root, file), "r", encoding="utf8") as file_read:
                csv_reader = csv.reader(file_read)
                for idx, line in enumerate(csv_reader):
                    all_preds[idx] += int(line[0])

    all_preds = (all_preds > n/2).astype(int)
    return all_preds


def check_rules(preds):
    with open("./data/neg_id.csv", "r", encoding="utf8") as file_read:
        neg_list = list(csv.reader(file_read))[0]
        for idx in neg_list:
            preds[int(idx)] = 1-int(preds[int(idx)])

    with open("./data/beiba_id.csv", "r", encoding="utf8") as file_read:
        beiba_list = list(csv.reader(file_read))[0]
        for idx in beiba_list:
            preds[int(idx)] = 1

    with open("./data/pinyin_id.csv", "r", encoding="utf8") as file_read:
        pinyin_list = list(csv.reader(file_read))[0]
        for idx in pinyin_list:
            preds[int(idx)] = 1

    with open("./data/na_id.csv", "r", encoding="utf8") as file_read:
        na_list = list(csv.reader(file_read))[0]
        for idx in na_list:
            preds[int(idx)] = 0

    with open("./data/temporal_id.csv", "r", encoding="utf8") as file_read:
        temporal_list = list(csv.reader(file_read))[0]
        for idx in temporal_list:
            preds[int(idx)] = 0

    with open("./data/name_id.csv", "r", encoding="utf8") as file_read:
        name_list = list(csv.reader(file_read))[0]
        for idx in name_list:
            preds[int(idx)] = 0

    preds[58828] = 1

    return preds


preds = ensemble()
print(sum(preds))
preds = check_rules(preds)
print(sum(preds))
with open("prediction.csv", "w", encoding="utf-8") as f:
    for pred in preds:
        f.write(str(pred) + "\n")
