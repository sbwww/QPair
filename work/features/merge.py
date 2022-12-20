import csv
import os
import sys
from typing import List, Optional, Tuple

import paddle
from asym_swap import swap_batch
from correction import make_batch_and_correct
from filter import filter_batch, full_to_abbr
from temporal_check import temporal_batch

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


def write_data(train_data: Optional[List[str]] = None,
               dev_data: Optional[List[str]] = None,
               test_data: Optional[List[str]] = None):
    if train_data is not None:
        save_path = "../data/train_merge.tsv"
        print("write {0} train to {1}".format(len(train_data), save_path))
        with open(save_path, "w", encoding="utf8") as file_write:
            for item in train_data:
                if not item.endswith("\n"):
                    item += "\n"
                file_write.write(item)

    if dev_data is not None:
        save_path = "../data/dev_merge.tsv"
        print("write {0} dev to {1}".format(len(dev_data), save_path))
        with open(save_path, "w", encoding="utf8") as file_write:
            for item in dev_data:
                if not item.endswith("\n"):
                    item += "\n"
                file_write.write(item)

    if test_data is not None:
        save_path = "../data/test_merge.tsv"
        print("write {0} test to {1}".format(len(test_data), save_path))
        with open(save_path, "w", encoding="utf8") as file_write:
            for item in test_data:
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
    ori_train_data, ori_dev_data, ori_test_data = read_and_merge_from_raw(merge_list)
    # train_data, dev_data, test_data = read_from_merge()
    train_data, dev_data, test_data = ori_train_data, ori_dev_data, ori_test_data

    for dataset in merge_list:
        print("merge data from {0}".format(dataset))
    print("total train: {0}".format(len(train_data)))
    print("total dev: {0}".format(len(dev_data)))
    print("total test: {0}".format(len(test_data)))

    dev_data = full_to_abbr(dev_data)
    dev_data, na_list = filter_batch(dev_data)
    browse_data("N/A", data1=ori_dev_data, list=na_list)
    browse_data("dev", data1=dev_data)
    write_data(dev_data=dev_data)

    train_data = full_to_abbr(train_data)
    train_data, na_list = filter_batch(train_data)
    browse_data("N/A", data1=ori_train_data, list=na_list)
    browse_data("train", data1=train_data)
    write_data(train_data=train_data)

    test_data = full_to_abbr(test_data)
    test_data, na_list = filter_batch(test_data)
    browse_data("N/A", data1=ori_test_data, data2=test_data, list=na_list)
    test_data, neg_list = swap_batch(test_data)
    browse_data("neg", data1=ori_test_data, data2=test_data, list=neg_list)
    temporal_list = temporal_batch(test_data)
    browse_data("temporal", data1=ori_test_data, data2=test_data, list=temporal_list)
    test_data = make_batch_and_correct(test_data)
    browse_data("test", data1=test_data)
    write_data(test_data=test_data)
