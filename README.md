# QPair

## Case Study

### Asymmetry

特征词有

> ['飞', '到', '去', '至', "离", "距"]

如

> 芜湖到上饶高铁时刻表	上饶到芜湖高铁时刻表

是 **不等价** 的

但是另有一些特征词，将 Asymmetry 变为 Symmetry

> ['多少', '高速费', '邮费', '多远','多少公里','距离', '多久','多长时间']

如

> 湖南到南平多少公里	南平到湖南多少公里

是 **等价** 的

### Neg-Asymmetry

A 比 B adj. -> 非（B 比 A adj.）

如

> 妈妈一定比儿子矮吗	儿子一定比妈妈高吗

替换成

> 妈妈一定比儿子矮吗	妈妈一定比儿子高吗

推断，结果为 0，再取反为最终结果 1

然而需要注意，如何识别 A 和 B

#### 识别 A 和 B

[PaddleNLP 词性标注](https://paddlenlp.readthedocs.io/zh/latest/model_zoo/taskflow.html#id9)

找 p（介词）前后的名词做替换

| 标签  |   含义   |
| :---: | :------: |
|   n   | 普通名词 |
|   f   | 方位名词 |
|   s   | 处所名词 |
|   t   |   时间   |
|  nr   |   人名   |
|  ns   |   地名   |
|  nt   |  机构名  |
|  nw   |  作品名  |
|  nz   | 其他专名 |
|  PER  |   人名   |
|  LOC  |   地名   |
|  ORG  |  机构名  |
| TIME  |   时间   |

## TODO:

-[x] 实现 asym 的改正，除了“比”之外的其他词
-[x] 纠错后 voice 和 misspell 等都还不错，但是要把有 NER 的单列出来
-[ ] ensemble

-[x] asymmetry不行，可能规则写错了，neg_asymmetry还可以

## 测试集

score	OPPO	DuQM_pos	DuQM_named_entity	DuQM_synonym	DuQM_antonym	DuQM_negation	DuQM_temporal	DuQM_symmetry	DuQM_asymmetry	DuQM_neg_asymmetry	DuQM_voice	DuQM_misspelling	DuQM_discourse_particle(simple)	DuQM_discourse_particle(complex)
48.815	71.29	100	99.926	0	100	83.191	100	0	100	0	29.008	0	0	0

DuQM_pos
DuQM_named_entity
DuQM_antonym
DuQM_asymmetry
DuQM_negation
DuQM_temporal

DuQM_synonym
DuQM_symmetry
DuQM_neg_asymmetry
DuQM_misspelling
DuQM_discourse_particle
