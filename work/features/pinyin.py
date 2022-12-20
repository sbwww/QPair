# 检测（1）对比两个句子是否拼音一致但书写不一致（2）有同音字不同的句子对（lazypinyin）
from pypinyin import lazy_pinyin


def compare_pinyin_and_text(text_list_1, text_list_2, text_label_list, print_progress=False):
    special_pinyin_list = [
        ["b", "p"],
        ["m", "n"],
        ["zh", "z"],
        ["ch", "c"],
        ["sh", "s"],
        ["r", "l"],
        ["r", "n"],
        ["n", "l"],
        ["en", "eng"],
        ["an", "ang"],
        ["in", "ing"],
        ["v", "i"],
        ["v", "u"],
        ["u", "i"]]
    special_words_list = [["四", "十"], ['汉', '韩']]
    special_sentence_list = ["那里人", "哪里人", "哪人", "多少岁"]
    pinyin_0_list = []
    label_0_list = []
    pinyin_1_list = []
    label_1_list = []
    count_percent = 0
    print("="*15 + " In Progress " + "="*15)
    for i in range(0, len(text_list_1)):
        count_percent = count_percent + 1

        text_1 = text_list_1[i]
        text_2 = text_list_2[i]

        subset_1, subset_2 = check_subset(text_1, text_2)

        contain_special_words = False
        if special_words_list != []:
            for special_word in special_words_list:
                if (special_word[0] == subset_1 and special_word[1] == subset_2) or (special_word[0] == subset_2 and special_word[1] == subset_1):
                    pinyin_0_list.append(text_list_1[i] + "\t" + text_list_2[i])
                    if text_label_list != None:
                        label_0_list.append(str(i) + "\t" + str(text_label_list[i]))
                    contain_special_words = True
        contain_special_sentences = False
        if special_sentence_list != []:
            for special_sentence in special_sentence_list:
                if special_sentence in text_1 and special_sentence in text_2:
                    contain_special_sentences = True
        # 整句话的拼音是否一致
        if not contain_special_words and lazy_pinyin(text_1) == lazy_pinyin(text_2) and text_1 != text_2:
            if contain_special_sentences:
                pinyin_0_list.append(text_list_1[i] + "\t" + text_list_2[i])
                if text_label_list != None:
                    label_0_list.append(str(i) + "\t" + str(text_label_list[i]))
            else:
                pinyin_1_list.append(text_list_1[i] + "\t" + text_list_2[i])
                if text_label_list != None:
                    label_1_list.append(str(i) + "\t" + str(text_label_list[i]))
        # 分析两句同样长度的话中不一样的部分拼音是否一致（因为整句话的识别 哪 nei 那 na等）
        elif not contain_special_words and subset_1 != "Not_The_Case" and len(subset_1) > 0 and len(subset_1) == len(subset_2):
            x = lazy_pinyin(subset_1)
            y = lazy_pinyin(subset_2)
            if x == y:
                if contain_special_sentences:
                    pinyin_0_list.append(text_list_1[i] + "\t" + text_list_2[i])
                    if text_label_list != None:
                        label_0_list.append(str(i) + "\t" + str(text_label_list[i]))
                else:
                    pinyin_1_list.append(text_list_1[i] + "\t" + text_list_2[i])
                    if text_label_list != None:
                        label_1_list.append(str(i) + "\t" + str(text_label_list[i]))
            elif len(x) == len(y) and x != y:
                for special_pinyin in special_pinyin_list:
                    for index in range(0, len(x)):
                        if special_pinyin[0] in x[index] and special_pinyin[0] not in y[index]:
                            x[index] = x[index].replace(special_pinyin[0], special_pinyin[1])
                        elif special_pinyin[0] in y[index] and special_pinyin[0] not in x[index]:
                            y[index] = y[index].replace(special_pinyin[0], special_pinyin[1])
                    if x == y:
                        if contain_special_sentences:
                            pinyin_0_list.append(text_list_1[i] + "\t" + text_list_2[i])
                            if text_label_list != None:
                                label_0_list.append(str(i) + "\t" + str(text_label_list[i]))
                        else:
                            pinyin_1_list.append(text_list_1[i] + "\t" + text_list_2[i])
                            if text_label_list != None:
                                label_1_list.append(str(i) + "\t" + str(text_label_list[i]))

        if print_progress and count_percent % 10000 == 0:
            print("%.2f of text pairs for different subsets identification have processed " %
                  ((count_percent / len(text_list_1)) * 100))
    print("="*15 + "     Done    " + "="*15)
    return pinyin_1_list, label_1_list, pinyin_0_list, label_0_list


# 使用 准备数据
test_text_pair_1_list, test_text_pair_2_list = read_candidates_test("work/test_B_processed.txt")
view_label_list = read_labels("predict_results/ccf_qianyan_qm_result_B.csv")

# 进行识别
homophone_text_pairs_1_list, homophone_label_1_list, homophone_text_pairs_0_list, homophone_label_0_list = compare_pinyin_and_text(
    test_text_pair_1_list, test_text_pair_2_list, view_label_list, True)
print(len(homophone_text_pairs_1_list))
print(homophone_text_pairs_1_list[0])
print(len(homophone_label_1_list))
print(homophone_label_1_list[0])
write_file("./data/pinyin_1_text.txt", homophone_text_pairs_1_list)
write_file("./data/pinyin_0_text.txt", homophone_text_pairs_0_list)

