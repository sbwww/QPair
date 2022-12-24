# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import csv
import os
import random
import sys
import time
from functools import partial

import numpy as np
import paddle
import paddle.nn.functional as F
from model import QuestionMatching
from paddlenlp.data import Pad, Stack, Tuple
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import AutoModel, AutoTokenizer
from tqdm import tqdm, trange

from data_util import convert_example, create_dataloader, read_text_pair

try:
    paddle.set_device("gpu")
except:
    paddle.set_device("cpu")
    print("No CUDA, use cpu instead.")

parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path",
                    type=str,
                    required=True,
                    help="The path of model.")
parser.add_argument("--input_file",
                    type=str,
                    required=True,
                    help="The full path of input file")
parser.add_argument("--result_file",
                    type=str,
                    required=True,
                    help="The result file name")
parser.add_argument("--max_seq_length",
                    default=256,
                    type=int,
                    help="The maximum total input sequence length after tokenization. "
                    "Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size",
                    default=32,
                    type=int,
                    help="Batch size per GPU/CPU for training.")
parser.add_argument("--device",
                    choices=["cpu", "gpu"],
                    default="gpu",
                    help="Select which device to train model, defaults to gpu.")
args = parser.parse_args()


def predict(model, dataloader):
    """
    Predicts the data labels.

    Args:
        model (obj:`QuestionMatching`): A model to calculate whether the question pair is semantic similar or not.
        data_loaer (obj:`List(Example)`): The processed data ids of text pair: [query_input_ids, query_token_type_ids, title_input_ids, title_token_type_ids]
    Returns:
        results(obj:`List`): cosine similarity of text pairs.
    """
    batch_logits = []

    model.eval()

    with paddle.no_grad():
        for _batch in tqdm(dataloader, desc="test"):
            input_ids, token_type_ids = _batch

            input_ids = paddle.to_tensor(input_ids)
            token_type_ids = paddle.to_tensor(token_type_ids)

            batch_logit, _ = model(input_ids=input_ids,
                                   token_type_ids=token_type_ids)

            batch_logits.append(batch_logit.detach().cpu().numpy())

        batch_logits = np.concatenate(batch_logits, axis=0)

    return batch_logits


if __name__ == "__main__":
    pretrained_model = AutoModel.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    # pretrained_model = AutoModel.from_pretrained("ernie-gram-zh")
    # tokenizer = AutoTokenizer.from_pretrained("ernie-gram-zh")

    trans_func = partial(convert_example,
                         tokenizer=tokenizer,
                         max_seq_length=args.max_seq_length,
                         is_test=True)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input_ids
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment_ids
    ): [data for data in fn(samples)]

    test_ds = load_dataset(read_text_pair,
                           data_path=args.input_file,
                           is_test=True,
                           lazy=False)
    # test_ds is a iterable of dict
    # {'query1': '手电筒打不开怎么回事？', 'query2': '手电筒打不开怎么回事'}
    test_dataloader = create_dataloader(test_ds,
                                        mode="predict",
                                        batch_size=args.batch_size,
                                        batchify_fn=batchify_fn,
                                        trans_fn=trans_func)

    model = QuestionMatching(pretrained_model)

    if args.model_name_or_path and os.path.isfile(os.path.join(args.model_name_or_path, "model_state.pdparams")):
        state_dict = paddle.load(os.path.join(args.model_name_or_path, "model_state.pdparams"))
        model.set_dict(state_dict)
        print("Loaded parameters from %s" % os.path.join(args.model_name_or_path, "model_state.pdparams"))
    else:
        raise ValueError(
            "Please set --model_name_or_path with correct pretrained model file")

    y_probs = predict(model, test_dataloader)
    y_preds = np.argmax(y_probs, axis=1)

    with open("./data/feature_data/neg_id.csv", "r", encoding="utf8") as file_read:
        neg_list = list(csv.reader(file_read))[0]
        for idx in neg_list:
            y_preds[int(idx)] = 1-int(y_preds[int(idx)])

    with open("./data/feature_data/beiba_id.csv", "r", encoding="utf8") as file_read:
        beiba_list = list(csv.reader(file_read))[0]
        for idx in beiba_list:
            y_preds[int(idx)] = 1

    with open("./data/feature_data/pinyin_id.csv", "r", encoding="utf8") as file_read:
        pinyin_list = list(csv.reader(file_read))[0]
        for idx in pinyin_list:
            y_preds[int(idx)] = 1

    with open("./data/feature_data/na_id.csv", "r", encoding="utf8") as file_read:
        na_list = list(csv.reader(file_read))[0]
        for idx in na_list:
            y_preds[int(idx)] = 0

    with open("./data/feature_data/temporal_id.csv", "r", encoding="utf8") as file_read:
        temporal_list = list(csv.reader(file_read))[0]
        for idx in temporal_list:
            y_preds[int(idx)] = 0

    with open("./data/feature_data/name_id.csv", "r", encoding="utf8") as file_read:
        name_list = list(csv.reader(file_read))[0]
        for idx in name_list:
            y_preds[int(idx)] = 0

    with open(args.result_file, "w", encoding="utf-8") as f:
        for y_pred in y_preds:
            f.write(str(y_pred) + "\n")
