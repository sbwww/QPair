import operator
import os
import re
from typing import List, Optional, Tuple

import numpy as np
import paddle
from paddlenlp.transformers import BertForMaskedLM, BertTokenizer
from tqdm import tqdm, trange

# merge_list = ["bq_corpus", "lcqmc", "oppo", "paws-x-zh"]
merge_list = ["bq_corpus", "lcqmc", "oppo"]

filter_pattern = [
    "hello\W*", "hi\W*", "hey\W*",
    "哈喽\W*", "嗨\W*", "你好\W*", "您好\W*", "嗯\W*", "啊\W*", "嘿\W*", "阿\W*",
    "小布\W*", "小爱同学\W*", "小步\W*", "小度\W*", "小o\W*", "小冰小冰\W*", "siri\W*"
]

try:
    paddle.set_device("gpu")
except:
    paddle.set_device("cpu")
    print("No CUDA, use cpu instead.")

tokenizer = BertTokenizer.from_pretrained("./model/macbert4csc-base-chinese")
model = BertForMaskedLM.from_pretrained("./model/macbert4csc-base-chinese")


def merge_data(
    merge_list: List[str]
) -> Tuple[List[str]]:
    train_data = []
    dev_data = []
    test_data = []
    for root, dirs, files in os.walk(".", topdown=False):
        if root == ".":
            print(os.path.join(root, "test.tsv"))
            with open(os.path.join(root, "test.tsv"), "r", encoding="utf8") as file_read:
                test_data.extend(file_read.readlines())
            continue
        for name in files:
            if root[2:] not in merge_list:
                # only use the datasets in merge_list
                continue
            if name == "train.tsv":
                print(os.path.join(root, name))
                with open(os.path.join(root, name), "r", encoding="utf8") as file_read:
                    train_data.extend(file_read.readlines())
            elif name == "dev.tsv":
                print(os.path.join(root, name))
                with open(os.path.join(root, name), "r", encoding="utf8") as file_read:
                    dev_data.extend(file_read.readlines())
    return train_data, dev_data, test_data


def read_data() -> Tuple[List[str]]:
    train_data = []
    dev_data = []
    test_data = []
    with open("../data/data180806/train_merge.tsv", "r", encoding="utf8") as file_read:
        train_data.extend(file_read.readlines())

    with open("../data/data180806/dev_merge.tsv", "r", encoding="utf8") as file_read:
        dev_data.extend(file_read.readlines())

    with open("../data/data180806/test_merge.tsv", "r", encoding="utf8") as file_read:
        test_data.extend(file_read.readlines())

    return train_data, dev_data, test_data


# filter out meaningless characters
def _fiter(data: List[str]) -> List[str]:
    data_filter = []

    for item in data:  # ["A", "B", "label"]
        item_filter = []

        for text in item:
            text_filter = text
            for pattern in filter_pattern:
                text_filter = re.sub(pattern, "", text_filter)
            item_filter.append(text_filter)

        if len(item_filter[0]) == 0 or len(item_filter[1]) == 0:  # one sequence is meaningless
            # if is train or dev set -> drop this data
            if len(item_filter) > 2:
                continue
            # if in test set -> keep but modify
            item_filter[0] = "N/A" if len(item_filter[0]) == 0 else item[0]
            item_filter[1] = "N/A" if len(item_filter[1]) == 0 else item[1]

        data_filter.append(item_filter)  # use this data

    return data_filter


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


def _truncate(data: List[str]) -> List[str]:
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


def get_errors(corrected_text, origin_text):
    sub_details = []
    for i, ori_char in enumerate(origin_text):
        if i >= len(corrected_text):
            break

        if ori_char in [" ", "“", "”", "‘", "’", "琊", "\n", "…", "—", "擤"]:
            # add unk character
            corrected_text = corrected_text[:i] + ori_char + corrected_text[i+1:]
            continue

        if i+1 < len(corrected_text):
            ori_2gram = "".join([origin_text[i], origin_text[i+1]])
            if ori_2gram in ["微粒", "粒贷", "余额", "到账", "转账"]:
                # add unk word
                corrected_text = corrected_text[:i] + ori_2gram + corrected_text[i+2:]
                continue

        if ori_char != corrected_text[i]:
            if ori_char.lower() == corrected_text[i]:
                # pass english upper char
                corrected_text = corrected_text[:i] + ori_char + corrected_text[i + 1:]
                continue
            sub_details.append((ori_char, corrected_text[i], i, i + 1))
    sub_details = sorted(sub_details, key=operator.itemgetter(2))
    return corrected_text, sub_details


def macbert4csc(texts):  # np.array[str] -> np.array[str]
    texts = list(texts)
    with paddle.no_grad():
        outputs = model(**tokenizer(texts, padding=True, return_tensors="pd"))

    result = []
    for ids, text in zip(outputs, texts):
        if text == "N/A":  # skip the correction for meaningless sequence
            result.append(text)
            continue

        _text = tokenizer.decode(
            paddle.argmax(ids, axis=-1), skip_special_tokens=True
        ).replace(" ", "")
        corrected_text = _text[:len(text)]
        corrected_text, details = get_errors(corrected_text, text)
        result.append(corrected_text)
    return np.array(result)


def _correct(data: List[str]) -> List[str]:
    data_tmp = np.array(data)  # we want slicing

    data_tmp[:, 0] = macbert4csc(data_tmp[:, 0])
    data_tmp[:, 1] = macbert4csc(data_tmp[:, 1])

    for item in data_tmp:
        if not item[-1].endswith("\n"):
            item[-1] += "\n"
    data_clean = ["\t".join(item) for item in data_tmp]
    return data_clean


def _clean(data: List[str]) -> List[str]:
    data_filter = _fiter(data)
    # data_truncate = _truncate(data_filter)
    data_correct = _correct(data_filter)
    return data_correct


def clean_data(dataset: List[str]) -> List[str]:
    data_clean = []
    batch_size = 64
    batches = []
    cur = 0
    while cur < len(dataset):
        data_chunk = dataset[cur:cur+batch_size]
        batches.append([data.split("\t") for data in data_chunk])
        cur += batch_size

    for batch in tqdm(batches, desc="Batch"):
        data_clean.extend(_clean(batch))

    return data_clean


def write_data(train_data: Optional[List[str]] = None,
               dev_data: Optional[List[str]] = None,
               test_data: Optional[List[str]] = None):
    if train_data is not None:
        with open("./data/train_merge.tsv", "w", encoding="utf8") as file_write:
            file_write.writelines(train_data)

    if dev_data is not None:
        with open("./data/dev_merge.tsv", "w", encoding="utf8") as file_write:
            file_write.writelines(dev_data)

    if test_data is not None:
        with open("./data/test_merge.tsv", "w", encoding="utf8") as file_write:
            file_write.writelines(test_data)


if __name__ == "__main__":
    # train_data, dev_data, test_data = merge_data(merge_list)
    train_data, dev_data, test_data = read_data()

    dev_data = clean_data(dev_data)
    write_data(dev_data=dev_data)
    train_data = clean_data(train_data)
    write_data(train_data=train_data)
    test_data = clean_data(test_data)
    write_data(test_data=test_data)

    for dataset in merge_list:
        print("merge data from {0}".format(dataset))
    print("total train: {0}".format(len(train_data)))
    print("total dev: {0}".format(len(dev_data)))
    print("total test: {0}".format(len(test_data)))
