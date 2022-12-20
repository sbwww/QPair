import csv
from typing import List, Tuple
from paddlenlp import Taskflow
from tqdm import tqdm, trange

# gru_crf_pos_tagging = Taskflow("pos_tagging")


def read_data() -> Tuple[List[str]]:
    test_data = []
    with open("../data/data180806/test_merge.tsv", "r", encoding="utf8") as file_read:
        test_data.extend(file_read.readlines())

    return test_data


def read_name_0() -> Tuple[List[str]]:
    data = []
    with open("./data/name.txt", "r", encoding="utf8") as file_read:
        data.extend(file_read.readlines())

    return data


def read_pinyin_1() -> Tuple[List[str]]:
    data = []
    with open("./data/pinyin.txt", "r", encoding="utf8") as file_read:
        data.extend(file_read.readlines())

    return data


def read_beiba_1() -> Tuple[List[str]]:
    data = []
    with open("./data/beiba.txt", "r", encoding="utf8") as file_read:
        data.extend(file_read.readlines())

    return data


def check_diff(text_pair: List[str]) -> bool:
    outputs = gru_crf_pos_tagging(text_pair)
    tag_list_pinyin_1, tag_list_2 = outputs[0], outputs[1]
    # TODO:
    # 1. 是多了 token ->
    # 2. 是 PER, LOC, TIME, ORG -> 不同
    # 3. 是拼音相同 -> 同


if __name__ == "__main__":
    test_data = read_data()
    name_0_data = read_name_0()
    pinyin_1_data = read_pinyin_1()
    beiba_1_data = read_beiba_1()

    list_name_0 = []
    list_pinyin_1 = []
    list_beiba_1 = []
    for idx, item in enumerate(test_data):
        # text_pair = item.split("\t")
        # if check_diff(text_pair):
        #     print(text_pair)
        if item in name_0_data:
            list_name_0.append(idx)
        elif item in pinyin_1_data:
            list_pinyin_1.append(idx)
        elif item in beiba_1_data:
            list_beiba_1.append(idx)

    print("total name 0: {0}".format(len(list_name_0)))
    print(list_name_0)
    with open("../data/feature_data/name_id.csv", "w", encoding="utf8") as file_write:
        csvwriter = csv.writer(file_write)
        csvwriter.writerow(list_name_0)

    print("total pinyin 1: {0}".format(len(list_pinyin_1)))
    print(list_pinyin_1)
    with open("../data/feature_data/pinyin_id.csv", "w", encoding="utf8") as file_write:
        csvwriter = csv.writer(file_write)
        csvwriter.writerow(list_pinyin_1)

    print("total beiba 1: {0}".format(len(list_pinyin_1)))
    print(list_beiba_1)
    with open("../data/feature_data/beiba_id.csv", "w", encoding="utf8") as file_write:
        csvwriter = csv.writer(file_write)
        csvwriter.writerow(list_beiba_1)
