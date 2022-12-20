import csv
from typing import List, Tuple
from paddlenlp import Taskflow
from tqdm import tqdm

gru_crf_pos_tagging = Taskflow("pos_tagging", user_dict="../data/feature_data/user_dict.txt")

# print(gru_crf_pos_tagging(["刘德华比张学友的年龄大几岁", "张学友比刘德华岁数小多少"]))
# print(gru_crf_pos_tagging(["拜登比特朗普大多少岁", "特朗普比拜登小多少岁"]))
# print(gru_crf_pos_tagging(["新郑到邯郸多少公里", "邯郸到新郑多少公里"]))
# print(gru_crf_pos_tagging(["老师被人们比喻成什么", "人们把老师比喻成什么"]))
# print(gru_crf_pos_tagging(["驶的拼音和组词怎么写", "驶的组词和拼音怎么写"]))
# print(gru_crf_pos_tagging(["西瓜与梨可以一起吃吗", "梨与西瓜可以一起吃吗"]))
# print(gru_crf_pos_tagging(["移动跟联通哪个网速好", "联通跟移动哪个网速好"]))
# print(gru_crf_pos_tagging(["重庆到上海时间", "重庆到飞上海时间"]))
# print(gru_crf_pos_tagging(["他刚刚到这里", "他之前到这里", "我去年写了书"]))


def read_from_merge() -> Tuple[List[str]]:
    test_data = []
    print("Reading data from ../data/test_merge.tsv")
    with open("../data/test_merge.tsv", "r", encoding="utf8") as file_read:
        test_data.extend(file_read.readlines())

    return test_data


def is_noun(tag: str) -> bool:
    return tag in ["n", "f", "s", "t", "d",
                   "nr", "ns", "nt", "nw", "nz",
                   "PER", "LOC", "ORG", "TIME"]


def is_sym(token_tag: List[str]) -> bool:
    return token_tag[0] in ["和", "与", "及", "或", "跟"] and token_tag[1] in ["c", "p"]


def is_neg_asym(token_tag: List[str]) -> bool:
    return token_tag[0] in ["比"] and token_tag[1] in ["p"]


def is_asym(token_tag: List[str]) -> bool:
    return token_tag[0] in ["飞", "到", "去", "至", "离", "距", "距离"] and token_tag[1] in ["p", "v"]


def is_equal(text: str) -> bool:
    # 西安到北京高铁多长时间	北京到西安高铁多长时间  1
    # 不能直接用token，可能不是这样分词的，要在原句子中找这些词
    # TODO: 核对 能提示等价性的 phrase
    for phrase in ["高速费", "邮费", "多远", "多少公里", "距离", "多久", "多长时间"]:
        if phrase in text:
            return True
    return False


def swap_seq_2(text_pair: List[str]) -> Tuple[List[str], bool]:
    outputs = gru_crf_pos_tagging(text_pair)

    tag_list_1, tag_list_2 = outputs[0], outputs[1]
    token_pivot_1, token_pivot_2 = -1, -1

    can_swap = False

    # find a pivot token in sequence 1
    for token_tag_1 in tag_list_1:
        token_pivot_1 += 1  # token
        if is_neg_asym(token_tag_1) or is_asym(token_tag_1) or is_sym(token_tag_1):
            # find a same pivot token in sequence 2
            token_pivot_2 = -1
            for token_tag_2 in tag_list_2:
                token_pivot_2 += 1  # token
                if token_tag_2 == token_tag_1:
                    can_swap = True
                    break
            break

    if not can_swap:
        return text_pair, False

    idx_1 = token_pivot_1
    swap_point_right, swap_point_left = -1, -1
    # left of pivot_1, find a same token in right of pivot_2
    while (idx_1 > 0):
        idx_1 -= 1
        token_1, tag_1 = tag_list_1[idx_1]
        if not is_noun(tag_1):
            continue
        for i in range(token_pivot_2+1, len(tag_list_2)):
            if tag_list_2[i][0] == token_1:
                swap_point_right = i
                break  # break the for-loop, not while, we want to find the matching of first noun

    idx_2 = token_pivot_2
    # left of pivot_2, find a same token in right of pivot_1
    while (idx_2 > 0):
        idx_2 -= 1
        token_2, tag_2 = tag_list_2[idx_2]
        if not is_noun(tag_2):
            continue
        for i in range(token_pivot_1+1, len(tag_list_1)):
            if tag_list_1[i][0] == token_2:
                swap_point_left = idx_2
                break  # break the for-loop, not while, we want to find the matching of first noun

    if swap_point_left == -1 or swap_point_right == -1:
        # cannot find same token
        return text_pair, False

    # swap text 2
    c = tag_list_2[swap_point_right]
    tag_list_2[swap_point_right] = tag_list_2[swap_point_left]
    tag_list_2[swap_point_left] = c

    # reconstruct text 2
    text_pair[1] = "".join([token for token, tag in tag_list_2])

    need_neg = False  # default, if is_sym, do not need to negate
    if is_neg_asym(tag_list_1[token_pivot_1]):
        need_neg = True
    elif is_asym(tag_list_1[token_pivot_1]):
        if not is_equal(text_pair[0]):
            need_neg = True

    return text_pair, need_neg


def save_swap_data(test_data: List[str]):
    save_path = "../data/test_swap.tsv"
    print("total test: {0}".format(len(test_data)))
    print("save to {0}".format(save_path))
    with open(save_path, "w", encoding="utf8") as file_write:
        for item in test_data:
            if not item.endswith("\n"):
                item += "\n"
            file_write.write(item)


def save_neg_list(neg_list: List[int]):
    save_path = "../data/feature_data/neg_id.csv"
    print("total neg: {0}".format(len(neg_list)))
    print("save to {0}".format(save_path))
    with open(save_path, "w", encoding="utf8") as file_write:
        csvwriter = csv.writer(file_write)
        csvwriter.writerow(neg_list)


# data: "A    B    label"
def swap_one(data: str) -> Tuple[str, bool]:
    text_pair = data.split("\t")
    text_pair, need_neg = swap_seq_2(text_pair)
    data_swaped = "\t".join(text_pair)
    return data_swaped, need_neg
# return: "B    A    label", True/False


# data_list: ["A    B    label", ]
def swap_batch(data_list: List[str]) -> Tuple[List[str], List[int]]:
    neg_list = []
    for idx, data in enumerate(tqdm(data_list, desc="asym_swap")):
        data_swaped, need_neg = swap_one(data)
        data_list[idx] = data_swaped
        if need_neg:
            neg_list.append(idx)

    return data_list, neg_list
# return: ["B    A    label", ], [idx, ]


if __name__ == "__main__":
    test_data = read_from_merge()
    test_data, neg_list = swap_batch(test_data)
    save_swap_data(test_data)
    save_neg_list(neg_list)
