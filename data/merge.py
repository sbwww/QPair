import operator
import os
import re
from typing import List, Tuple

import torch
from transformers import BertForMaskedLM, BertTokenizer

# merge_list = ["bq_corpus", "lcqmc", "oppo", "paws-x-zh"]
merge_list = ["bq_corpus", "lcqmc", "oppo"]

filter_pattern = ["小布"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained("../model/macbert4csc-base-chinese")
model = BertForMaskedLM.from_pretrained("../model/macbert4csc-base-chinese")
model.to(device)


def merge_data(merge_list: List[str]) -> Tuple[List[str]]:
    train_data = []
    dev_data = []
    for root, dirs, files in os.walk(".", topdown=False):
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
    return train_data, dev_data


# filter out certain words
def _fiter(data: str) -> str:
    data_tmp = data
    for pattern in filter_pattern:
        data_tmp = re.sub(pattern, "", data_tmp)
    data_clean = data_tmp
    return data_clean


def get_errors(corrected_text, origin_text):
    sub_details = []
    for i, ori_char in enumerate(origin_text):
        if i >= len(corrected_text):
            break

        if ori_char in [' ', '“', '”', '‘', '’', '琊', '\n', '…', '—', '擤']:
            # add unk character
            corrected_text = corrected_text[:i] + ori_char + corrected_text[i+1:]
            continue

        if i+1 < len(corrected_text):
            ori_2gram = "".join([origin_text[i], origin_text[i+1]])
            if ori_2gram in ["微粒", "粒贷"]:
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


def macbert4csc(text: str) -> str:
    with torch.no_grad():
        outputs = model(**tokenizer(text, padding=True, return_tensors="pt").to(device))

    result = []
    for ids in outputs.logits:
        _text = tokenizer.decode(
            torch.argmax(ids, dim=-1), skip_special_tokens=True
        ).replace(" ", "")
        corrected_text = _text[:len(text)]
        corrected_text, details = get_errors(corrected_text, text)
    return corrected_text


def _correct(data: str) -> str:
    data_tmp = data.split("\t")

    data_tmp[0] = macbert4csc(data_tmp[0])
    data_tmp[1] = macbert4csc(data_tmp[1])

    data_clean = "\t".join(data_tmp)
    return data_clean


def _clean(data: str) -> str:
    data_tmp = data
    data_tmp = _fiter(data_tmp)
    data_tmp = _correct(data_tmp)
    data_clean = data_tmp
    return data_clean


def clean_data(train_data: List[str], dev_data: List[str]) -> Tuple[List[str]]:
    train_data_clean = []
    dev_data_clean = []

    for data in train_data:
        data_clean = _clean(data)
        train_data_clean.append(data_clean)
    for data in dev_data:
        data_clean = _clean(data)
        dev_data_clean.append(data_clean)

    return train_data_clean, dev_data_clean


def write_data(train_data: List[str], dev_data: List[str]):
    with open("./train_merge.tsv", "w", encoding="utf8") as file_write:
        file_write.writelines(train_data)

    with open("./dev_merge.tsv", "w", encoding="utf8") as file_write:
        file_write.writelines(dev_data)


if __name__ == "__main__":
    train_data, dev_data = merge_data(merge_list)
    train_data, dev_data = clean_data(train_data, dev_data)
    for dataset in merge_list:
        print("merge data from {0}".format(dataset))
        print("total train: {0}".format(len(train_data)))
        print("total dev: {0}".format(len(dev_data)))
    write_data(train_data, dev_data)
