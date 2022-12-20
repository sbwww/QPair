import json
import os

with open("./oppo.json", "r", encoding="utf8") as file_read:
    json_data = json.load(file_read)

    print("train:", len(json_data["train"]))
    print("dev:", len(json_data["dev"]))
    print("test:", len(json_data["test"]))

    with open("./train.tsv", "w", encoding="utf8") as file_write:
        for question_pair in json_data["train"]:
            file_write.write("\t".join((question_pair["q1"], question_pair["q2"], question_pair["label"]))+"\n")

    with open("./dev.tsv", "w", encoding="utf8") as file_write:
        for question_pair in json_data["dev"]:
            file_write.write("\t".join((question_pair["q1"], question_pair["q2"], question_pair["label"]))+"\n")

    with open("./test.tsv", "w", encoding="utf8") as file_write:
        for question_pair in json_data["test"]:
            file_write.write("\t".join((question_pair["q1"], question_pair["q2"]))+"\n")
