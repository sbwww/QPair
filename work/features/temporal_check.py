import csv
from typing import List, Tuple
from tqdm import tqdm


def read_from_merge() -> Tuple[List[str]]:
    test_data = []
    print("Reading data from ../data/test_merge.tsv")
    with open("../data/test_merge.tsv", "r", encoding="utf8") as file_read:
        test_data.extend(file_read.readlines())

    return test_data


# item_split: ["A", "B", label]
def is_diff_temporal(item_split: List[str]) -> bool:
    temporal_phrases = ["刚才", "以前", "去年", "年前"]
    # TODO: just trick
    for phrase in temporal_phrases:
        if phrase in item_split[0] and phrase not in item_split[1]:
            return True
        if phrase not in item_split[0] and phrase in item_split[1]:
            return True
    return False


def save_temporal_list(temporal_list):
    save_path = "../data/temporal_id.csv"
    print("total temporal: {0}".format(len(temporal_list)))
    print("save to {0}".format(save_path))
    with open(save_path, "w", encoding="utf8") as file_write:
        csvwriter = csv.writer(file_write)
        csvwriter.writerow(temporal_list)


# data: "A    B    label"
def temporal_one(data: str) -> Tuple[str, bool]:
    text_pair = data.split("\t")
    is_diff = is_diff_temporal(text_pair)
    return is_diff
# return: "B    A    label", True/False


# data_list: ["A    B    label", ]
def temporal_batch(data_list: List[str]) -> Tuple[List[str], List[int]]:
    temporal_list = []
    for idx, data in enumerate(tqdm(data_list)):
        is_diff = temporal_one(data)
        if is_diff:
            temporal_list.append(idx)

    print("total temporal: {0}".format(len(temporal_list)))
    print(temporal_list)
    for i in range(min(5, len(temporal_list))):
        print(data_list[i])

    return temporal_list
# return: ["B    A    label", ], [idx, ]


if __name__ == "__main__":
    test_data = read_from_merge()
    temporal_list = temporal_batch(test_data)
    save_temporal_list(temporal_list)
