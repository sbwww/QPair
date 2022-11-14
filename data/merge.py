import os

train_data = []
dev_data = []

for root, dirs, files in os.walk(".", topdown=False):
    for name in files:
        if name == "train.tsv":
            print(os.path.join(root, name))
            with open(os.path.join(root, name), "r", encoding="utf8") as file_read:
                train_data.extend(file_read.readlines())
        elif name == "dev.tsv":
            print(os.path.join(root, name))
            with open(os.path.join(root, name), "r", encoding="utf8") as file_read:
                dev_data.extend(file_read.readlines())


print("train:", len(train_data))
print("dev:", len(dev_data))

with open("./train_merge.tsv", "w", encoding="utf8") as file_write:
    file_write.writelines(train_data)

with open("./dev_merge.tsv", "w", encoding="utf8") as file_write:
    file_write.writelines(dev_data)
