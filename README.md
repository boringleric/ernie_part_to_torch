# ernie_part_to_torch
ernie模型看起来不错，但是只能用在paddle，懒狗就想试试能不能暴力转torch

转换方式参考了https://github.com/nghuyong/ERNIE-Pytorch 的项目，感谢！

# ernie_gram转换的一些问题

1. 由于ernie_gram和bert的结构一致，所以直接使用了transformers的bert结构作为迁移对象，迁移后的模型可以直接使用transformers进行调用。
2. 使用转换的代码生成的config.json里面没有model_type字段，所以AutoModel会报错，在config.json里面加上 "model_type": "bert",就好了。
3. 为什么ernie_gram无法预测BertForMaskedLM任务，因为ernie_gram的原版模型里面没有cls层的权重，而BertForMaskedLM任务需要使用到cls层权重，默认初始化的权重显然不足以完成Mask的预测。

