import csv
import re
from typing import List, Optional, Tuple

from tqdm import tqdm

filter_pattern = {
    "": "", "hello\W*": "", "hi\W*": "", "hey\W*": "",
    "((我想|麻烦)*((请|询|问|)问|咨询)(一*下))|((我想|麻烦)((请|询|问|)问|咨询)(一*下)*)|((我想|麻烦)((请|询|问|)问|咨询)(一*下))\W*": "", "小布\W*": "",
    "\W*谢谢\W*": "", "求(解|救|教|助)\W*": "", "(请|求|麻烦)*各位\W*": "", "\W?急+\W*$": "",
    "小爱同学\W*": "", "小步\W*": "", "小度\W*": "", "小o\W*": "", "小冰小冰\W*": "", "siri\W*"
    "哈(喽|啰|咯)\W*": "", "(你|您)好\W*": "",
    "(嗯|啊|阿|嘿|哈|嗨|哇|呀|咯|呗|嘛|哦|噢)\W*": "",
    "(我|祖)国": "中国",
    "之前|以往|曾经|过去": "以前", "刚刚": "刚才",
    "这么": "那么", "你知道(.*)吗": lambda matched: matched.group(1), "你知不知道": "",
    "(男|女)(人|生|的|孩子*)": lambda matched: matched.group(1) + "性",
}


def read_from_merge() -> Tuple[List[str]]:
    test_data = []
    print("Reading data from ../data/test_merge.tsv")
    with open("../data/test_merge.tsv", "r", encoding="utf8") as file_read:
        test_data.extend(file_read.readlines())

    return test_data


def full_to_abbr(dataset: List[str]) -> List[str]:
    full_abbr_pairs = []
    with open("../data/feature_data/abbr.txt", "r", encoding="utf8") as file_read:
        full_abbr_pairs.append(file_read.readline().split("\t"))

    data_filter = []
    for data in tqdm(dataset, desc="full_to_abbr"):
        data_swapped = data.split("\t")
        for pair in full_abbr_pairs:
            data_swapped[0] = re.sub(pair[0], pair[1], data_swapped[0])
            data_swapped[1] = re.sub(pair[0], pair[1], data_swapped[1])
        data_filter.append("\t".join(data_swapped))

    return data_filter


def filter_one(data: str) -> Tuple[Optional[str], Optional[bool]]:
    text_pair = data.split("\t")  # ["A", "B", "label"]
    if text_pair[-1].endswith("\n"):
        text_pair[-1] = text_pair[-1][:-1]

    is_na = False
    data_swapped = []
    for i, text in enumerate(text_pair):
        if i >= 2:  # do not touch the label
            data_swapped.append(text)
            continue

        text_filter = text
        for pattern, repl in filter_pattern.items():
            text_filter = re.sub(pattern, repl, text_filter)
            if "哦你知道" in text_filter:
                print(text_filter)
        data_swapped.append(text_filter)

    if len(data_swapped[0]) == 0 or len(data_swapped[1]) == 0:  # one sequence is meaningless
        if len(data_swapped) > 2:
            # if is train or dev set -> drop this data
            return None, None
        # if in test set -> keep but modify
        data_swapped[0] = "N/A" if len(data_swapped[0]) == 0 else text_pair[0]
        data_swapped[1] = "N/A" if len(data_swapped[1]) == 0 else text_pair[1]
        if data_swapped[0] == "N/A" or data_swapped[1] == "N/A":
            is_na = True

    data_swapped = "\t".join(data_swapped)
    if not data_swapped.endswith("\n"):
        data_swapped += "\n"
    return data_swapped, is_na


# filter out meaningless characters
def filter_batch(data_list: List[str]) -> Tuple[List[str], List[int]]:
    data_filter = []
    na_list = []  # "N/A"
    for idx, data in enumerate(tqdm(data_list, desc="filter")):  # data: "A    B   label"
        data_swapped, is_na = filter_one(data)
        if data_swapped is not None:
            data_filter.append(data_swapped)  # use this data
            if is_na:
                na_list.append(idx)

    return data_filter, na_list


def _truncate(data: List[List[str]]) -> List[List[str]]:
    def longest_common_prefix(str_A: str, str_B: str) -> int:
        for i in range(len(str_A)):
            c = str_A[i]
            if (i == len(str_B) or str_B[i] != c):
                return i
        return 0

    def longest_common_suffix(str_A: str, str_B: str) -> int:
        for i in range(-1, -len(str_A)-1, -1):
            c = str_A[i]
            if (i == -len(str_B)-1 or str_B[i] != c):
                return -i-1
        return 0

    data_tmp = data
    for item in data_tmp:
        prefix_len = longest_common_prefix(item[0], item[1])
        item[0] = item[0][prefix_len:]
        item[1] = item[1][prefix_len:]
        suffix_len = longest_common_suffix(item[0], item[1])
        if suffix_len > 0:
            item[0] = item[0][:-suffix_len]
            item[1] = item[1][:-suffix_len]
    data_clean = data_tmp
    return data_clean


def save_na_list(na_list):
    with open("../data/na_id.csv", "w", encoding="utf8") as file_write:
        csvwriter = csv.writer(file_write)
        csvwriter.writerow(na_list)


def save_filter_data(test_data: List[str]):
    save_path = "../data/test_filter.tsv"
    print("total test: {0}".format(len(test_data)))
    print("save to {0}".format(save_path))
    with open(save_path, "w", encoding="utf8") as file_write:
        for item in test_data:
            if not item.endswith("\n"):
                item += "\n"
            file_write.write(item)


if __name__ == "__main__":
    test_data = read_from_merge()
    test_data, na_list = filter_batch(test_data)
    save_filter_data(test_data)
