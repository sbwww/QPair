import csv
import os
import sys
from typing import List, Optional, Tuple

import paddle
from asym_swap import swap_batch
from correction import make_batch_and_correct
from filter import filter_batch, full_to_abbr_batch
from temporal_check import temporal_batch
from paddlenlp import Taskflow

sys.path.append('/home/aistudio/external-libraries')


raw_data_path = "../data/raw"

# merge_list = ["bq_corpus", "lcqmc", "oppo", "paws-x-zh"]
merge_list = ["bq_corpus", "lcqmc", "oppo"]
merge_list = [os.path.join(raw_data_path, i) for i in merge_list]

try:
    paddle.set_device("gpu")
except:
    paddle.set_device("cpu")
    print("No CUDA, use cpu instead.")


def read_and_merge_from_raw(
    merge_list: List[str]
) -> Tuple[List[str]]:
    train_data = []
    dev_data = []
    test_data = []
    for root, dirs, files in os.walk(raw_data_path, topdown=False):
        if root == raw_data_path:
            print(os.path.join(root, "test.tsv"))
            with open(os.path.join(root, "test.tsv"), "r", encoding="utf8") as file_read:
                test_data.extend(file_read.readlines())
            continue
        for name in files:
            if root not in merge_list:
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


def read_from_merge() -> Tuple[List[str]]:
    train_data = []
    dev_data = []
    test_data = []
    with open("../data/train_merge.tsv", "r", encoding="utf8") as file_read:
        train_data.extend(file_read.readlines())

    with open("../data/dev_merge.tsv", "r", encoding="utf8") as file_read:
        dev_data.extend(file_read.readlines())

    with open("../data/test_merge.tsv", "r", encoding="utf8") as file_read:
        test_data.extend(file_read.readlines())

    return train_data, dev_data, test_data


def save_feature_list(name, list):
    save_path = "../data/feature_data/{0}_id.csv".format(name)
    print("write {0} feature to {1}".format(name, save_path))
    with open(save_path, "w", encoding="utf8") as file_write:
        csvwriter = csv.writer(file_write)
        csvwriter.writerow(list)


def save_data(name: str, data: List[str]):
    save_path = "../data/{0}_merge.tsv".format(name)
    print("write {0} {1} to {2}".format(len(data), name, save_path))
    with open(save_path, "w", encoding="utf8") as file_write:
        for item in data:
            if not item.endswith("\n"):
                item += "\n"
            file_write.write(item)


def browse_data(name, data1=None, data2=None, list=None):
    if list is not None:
        print("total {0} list: {1}".format(name, len(list)))
        print(list)
        for i in range(min(5, len(list))):
            if data2 is not None:
                print(data1[list[i]], "->", data2[list[i]])
            else:
                print(data1[list[i]])
    else:
        print("total {0} data: {1}".format(name, len(data1)))
        for i in range(min(5, len(data1))):
            print(data1[i])


if __name__ == "__main__":
    gru_crf_pos_tagging = Taskflow("pos_tagging", user_dict="../data/feature_data/user_dict.txt")

    ori_train_data, ori_dev_data, ori_test_data = read_and_merge_from_raw(merge_list)
    # train_data, dev_data, test_data = read_from_merge()
    train_data, dev_data, test_data = ori_train_data, ori_dev_data, ori_test_data

    for dataset in merge_list:
        print("merge data from {0}".format(dataset))
    print("total train: {0}".format(len(train_data)))
    print("total dev: {0}".format(len(dev_data)))
    print("total test: {0}".format(len(test_data)))

    dev_data = full_to_abbr_batch(dev_data)
    dev_data, na_list = filter_batch(dev_data)
    browse_data(name="dev", data1=dev_data)
    save_data(name="dev", data=dev_data)

    train_data = full_to_abbr_batch(train_data)
    train_data, na_list = filter_batch(train_data)
    browse_data(name="train", data1=train_data)
    save_data(name="train", data=train_data)

    test_data = full_to_abbr_batch(test_data)
    test_data, na_list = filter_batch(test_data)
    browse_data(name="N/A", data1=ori_test_data, data2=test_data, list=na_list)
    save_feature_list("na", na_list)
    test_data, neg_list = swap_batch(gru_crf_pos_tagging, test_data)
    browse_data(name="neg", data1=ori_test_data, data2=test_data, list=neg_list)
    save_feature_list("neg", neg_list)
    temporal_list = temporal_batch(test_data)
    browse_data(name="temporal", data1=ori_test_data, list=temporal_list)
    save_feature_list("temporal", temporal_list)
    # test_data = make_batch_and_correct(test_data)
    browse_data(name="test", data1=test_data)
    save_data(name="test", data=test_data)
