import os

import numpy as np
import paddle
from features.asym_swap import swap_one
from features.filter import filter_one, full_to_abbr_one
from features.temporal_check import temporal_one
from model import QuestionMatching
from paddlenlp.transformers import AutoModel, AutoTokenizer
from paddlenlp import Taskflow

try:
    paddle.set_device("gpu")
except:
    paddle.set_device("cpu")
    print("No CUDA, use cpu instead.")

model_name_or_path = "./model/ernie-3-xbase"
# model_name_or_path = "./model/ernie_gram_rdrop0p0"
max_seq_length = 256
id_2_label = {0: "不同",
              1: "相同"}


def infer(model, encoded_inputs):
    model.eval()
    with paddle.no_grad():
        input_ids, token_type_ids = encoded_inputs["input_ids"], encoded_inputs["token_type_ids"]

        input_ids = paddle.to_tensor(input_ids)
        token_type_ids = paddle.to_tensor(token_type_ids)

        logits, _ = model(input_ids=input_ids,
                          token_type_ids=token_type_ids)

        logits = logits.detach().cpu().numpy()

    return logits


if __name__ == "__main__":
    gru_crf_pos_tagging = Taskflow("pos_tagging", user_dict="./data/feature_data/user_dict.txt")
    full_abbr_pairs = []
    with open("./data/feature_data/abbr.txt", "r", encoding="utf8") as file_read:
        full_abbr_pairs.append(file_read.readline().split("\t"))

    pretrained_model = AutoModel.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    model = QuestionMatching(pretrained_model)

    if model_name_or_path and os.path.isfile(os.path.join(model_name_or_path, "model_state.pdparams")):
        state_dict = paddle.load(os.path.join(model_name_or_path, "model_state.pdparams"))
        model.set_dict(state_dict)
        print("Loaded parameters from %s" % os.path.join(model_name_or_path, "model_state.pdparams"))
    else:
        raise ValueError(
            "Please set --model_name_or_path with correct pretrained model file")

    while True:
        # query_A = input("Input query A:")
        # query_B = input("Input query B:")
        query_A = "北京到南京有多远"
        query_B = "南京到北京有多远"
        input_text_pair = [query_A, query_B]
        filtered_text_pair = full_to_abbr_one(input_text_pair, full_abbr_pairs)
        filtered_text_pair, is_na = filter_one(filtered_text_pair)
        if filtered_text_pair != input_text_pair:
            print("Filter:")
            print("   {0}\n-> {1}".format(input_text_pair, filtered_text_pair))

        swapped_text_pair, need_neg = swap_one(gru_crf_pos_tagging, filtered_text_pair)
        if swapped_text_pair != filtered_text_pair:
            print("Swap:")
            print("   {0}\n-> {1}".format(filtered_text_pair, swapped_text_pair))

        encoded_inputs = tokenizer(text=swapped_text_pair[0],
                                   text_pair=swapped_text_pair[1],
                                   max_seq_len=max_seq_length,
                                   return_tensors="pd")
        y_prob = infer(model, encoded_inputs)
        print("   Logits: {0}".format(y_prob))
        y_pred = np.argmax(y_prob)
        print("   Base prediction: {0}".format(id_2_label[y_pred]))

        is_diff_temporal = temporal_one(input_text_pair)

        if need_neg:
            print("   Need negation" if need_neg else "   Do not need negation")
            y_pred = 1-y_pred
        if is_na:
            print("   {0} is meaningless".format(input_text_pair))
            y_pred = 0
        if is_diff_temporal:
            print("   {0} different in temporal".format(input_text_pair))
            y_pred = 0

        print("   Final prediction: {0}".format(id_2_label[y_pred]))
        break
