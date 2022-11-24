import os


def merge_data(merge_list):
    train_data = []
    dev_data = []
    for root, dirs, files in os.walk(".", topdown=False):
        for name in files:
            if root[2:] not in merge_list:
                continue
            if name == "train.tsv":
                print(os.path.join(root, name))
                with open(os.path.join(root, name), "r", encoding="utf8") as file_read:
                    train_data.extend(file_read.readlines())
            elif name == "dev.tsv":
                print(os.path.join(root, name))
                with open(os.path.join(root, name), "r", encoding="utf8") as file_read:
                    dev_data.extend(file_read.readlines())

    for dataset in merge_list:
        print("merge data from {0}".format(dataset))
    print("total train: {0}".format(len(train_data)))
    print("total dev: {0}".format(len(dev_data)))

    with open("./train_merge.tsv", "w", encoding="utf8") as file_write:
        file_write.writelines(train_data)

    with open("./dev_merge.tsv", "w", encoding="utf8") as file_write:
        file_write.writelines(dev_data)


if __name__ == "__main__":
    merge_list = ["bq_corpus", "lcqmc", "oppo", "paws-x-zh"]
    merge_data(merge_list)
