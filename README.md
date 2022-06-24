# ernie_part_to_torch
ernie模型看起来不错，但是只能用在paddle，懒狗就想试试能不能暴力转torch

转换方式参考了https://github.com/nghuyong/ERNIE-Pytorch 的项目，感谢！

# 转换的基础环境

python 3.9.13  
paddlepaddle-gpu 2.3.0  
pytorch 1.11.0 
cudatoolkit 10.2.89 

# ernie_gram 转换的一些问题

贴上一个转换完成的模型地址：https://drive.google.com/file/d/1jMqN6UmTIQWx9S61jB5-YVGQ-njT2a2O

1. 由于ernie_gram和bert的结构一致，所以直接使用了transformers的bert结构作为迁移对象，迁移后的模型可以直接使用transformers进行调用。
2. 使用转换的代码生成的config.json里面没有model_type字段，所以AutoModel会报错，在config.json里面加上 "model_type": "bert",就好了。
3. 为什么ernie_gram无法预测BertForMaskedLM任务，因为ernie_gram的原版模型里面没有cls层的权重，而BertForMaskedLM任务需要使用到cls层权重，默认初始化的权重显然不足以完成Mask的预测。
4. 调用方式和验证向量一致性在ernie_gram_torch_paddle_sim_check.py。

# ernie3 转换的一些问题

贴上一个转换完成的模型地址：https://drive.google.com/file/d/1qPx-3XCRuO7R8Nxtn7Qrcoe5iFczUvya/

1. ernie3和bert的结构并不一致，ernie3的embedding层多了task_type_embeddings，所以照搬的话在向量一致性核查上面肯定不会通过，因此借鉴了PaddleNLP的Ernie3代码，魔改一番，至少目前转换的向量一致性和MaskLM预测任务没有问题。
2. 魔改的代码里面，pretrained相关部分没有做验证，可能会有错误！后续考虑完善吧...
3. 调用方式和验证向量一致性和MaskLM任务在ernie_3_torch_paddle_check.py。

# UIE模型的一个尝试

贴上一个转换完成的模型地址：https://drive.google.com/file/d/1cp81I0iqA3aoWWBuQ4YxBUVJww58Sz6U/
UIE是paddlenlp做的一个比较有意思的论文，采用prompt训练方式基于ernie3模型做的一套通用信息抽取框架，通过改变不同的任务schema，可以完成实体抽取、关系抽取、事件抽取、情感分析等任务，很值得玩一下。
论文见https://arxiv.org/pdf/2203.12277.pdf

代码方面有一点小小的改动，由于UIE使用的span方式没有在ernie3原有modeling之中，所以专门新加了一部分进去，当然兼容性没有那么好，凑合玩了。
调用方式和验证结果一致性在ernie_3_uie_to_torch_check.py。