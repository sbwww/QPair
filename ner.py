import torch
from ltp import LTP
import csv

ltp = LTP("LTP/base")  # 默认加载 Small 模型

# 将模型移动到 GPU 上
if torch.cuda.is_available():
    # ltp.cuda()
    ltp.to("cuda")

# output = ltp.pipeline(["他叫汤姆去拿外衣。"], tasks=["cws", "pos", "ner", "srl", "dep", "sdp", "sdpg"])
# print(output.cws)  # 中文分词
# print(output.pos)  # 词性标注
# print(output.ner)  # 命名体识别
# print(output.srl)  # 语义角色标注
# print(output.dep)  # 依存句法分析
# print(output.sdp)  # 语义依存分析(树)
# print(output.sdpg)  # 语义依存分析(图)


# with open("./data/dev_merge.tsv", encoding="utf8") as file:
with open("./data/test.tsv", encoding="utf8") as file:
    tsv_file = csv.reader(file, delimiter="\t")

    for line in tsv_file:
        output = ltp.pipeline(line[:2], tasks=["cws",  "ner"])
        ner_result_list = output.ner
        if (ner_result_list[0] != ner_result_list[1]):
            print(line)
            print(output.ner)
