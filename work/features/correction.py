from typing import List, Tuple

import numpy as np
import paddle
import paddle.nn.functional as F
from paddlenlp import Taskflow
from paddlenlp.transformers import BertForMaskedLM, BertTokenizer
from tqdm import tqdm

tokenizer = BertTokenizer.from_pretrained("../model/macbert4csc-base-chinese")
model = BertForMaskedLM.from_pretrained("../model/macbert4csc-base-chinese")

lac_ner = Taskflow("ner", mode="fast", entity_only=True, user_dict="../data/feature_data/user_dict.txt")
# 使用快速模式，只返回实体词
#   lac_ner("三亚是一个美丽的城市")
#   [('三亚', 'LOC')]


def read_from_merge() -> Tuple[List[str]]:
    test_data = []
    print("Reading data from ../data/test_merge.tsv")
    with open("../data/test_merge.tsv", "r", encoding="utf8") as file_read:
        test_data.extend(file_read.readlines())

    return test_data


def decode_correction(ids, origin_text):
    _text = tokenizer.decode(
        paddle.argmax(ids, axis=-1), skip_special_tokens=True
    ).replace(" ", "")
    corrected_text = _text[:len(origin_text)]

    entity_list = lac_ner(origin_text)
    for entity in entity_list:
        entity_token = entity[0]
        # find position of entity in origin_text
        entity_start = origin_text.find(entity_token)
        entity_end = entity_start + len(entity_token)-1
        # do not touch entities
        corrected_text = corrected_text[:entity_start]+entity_token+corrected_text[entity_end+1:]

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
            probs = F.softmax(ids[i+1])  # ids[0] is [CLS]
            # entropy calculation on probs: -\sum(p \ln(p))
            entropy = -paddle.sum(probs*paddle.log(probs))
            print(entropy, origin_text, corrected_text)
            # TODO: threshold?
    return corrected_text


# texts: ["A1", "A2", "A3", ]
def macbert4csc(texts):  # np.array[str] -> np.array[str]
    texts = list(texts)
    with paddle.no_grad():
        outputs = model(**tokenizer(texts, padding=True, return_tensors="pd"))

    result = []
    for ids, text in zip(outputs, texts):
        if text == "N/A":  # skip the correction for meaningless sequence
            result.append(text)
            continue

        corrected_text = decode_correction(ids, text)
        result.append(corrected_text)
    return np.array(result)
# output: ["A1", "A2", "A3", ]


# data_list: [ ["A", "B", label], [], ]
def correct_batch(data_list: List[List[str]]) -> List[List[str]]:
    data_tmp = np.array(data_list)  # we want slicing

    data_tmp[:, 0] = macbert4csc(data_tmp[:, 0])  # ["A1", "A2", "A3", ]
    data_tmp[:, 1] = macbert4csc(data_tmp[:, 1])  # ["B1", "B2", "B3", ]

    corrected_text_pair_list = list(data_tmp)
    return corrected_text_pair_list
# return: [ ["A", "B", label], [], ]


def make_batch_and_correct(dataset: List[str]) -> List[str]:
    corrected_data = []
    batch_size = 64
    batches = []
    cur = 0
    while cur < len(dataset):
        data_chunk = dataset[cur:cur+batch_size]
        batches.append([data[:-1].split("\t") for data in data_chunk])  # [:-1] removes \n
        cur += batch_size

    for batch in tqdm(batches, desc="correct"):  # List[List[str]]
        corrected_text_pair_list = correct_batch(batch)
        for corrected_text_pair in corrected_text_pair_list:
            corrected_merge = "\t".join(corrected_text_pair)
            if not corrected_merge.endswith("\n"):
                corrected_merge += "\n"
            corrected_data.append(corrected_merge)

    return corrected_data


def save_correct_data(test_data: List[str]):
    save_path = "../data/test_correct.tsv"
    print("total test: {0}".format(len(test_data)))
    print("save to {0}".format(save_path))
    with open(save_path, "w", encoding="utf8") as file_write:
        for item in test_data:
            if not item.endswith("\n"):
                item += "\n"
            file_write.write(item)


if __name__ == "__main__":
    test_data = read_from_merge()
    test_data = make_batch_and_correct(test_data)
    save_correct_data(test_data)
