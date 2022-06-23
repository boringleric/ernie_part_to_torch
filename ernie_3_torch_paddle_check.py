import torch
from transformers import BertTokenizer
from ernie3.modeling import ErnieModel, ErnieForMaskedLM
from sklearn.metrics.pairwise import cosine_similarity

torch_model = ErnieModel.from_pretrained('./convert_ernie_3')
torch_tokenizer = BertTokenizer.from_pretrained('./convert_ernie_3')
torch_model.eval()
torch_encoding = torch_tokenizer("欢迎使用百度飞桨！")
torch_encoding = {k:torch.tensor([v]) for (k, v) in torch_encoding.items()}
torch_output = torch_model(**torch_encoding)

import paddle
from paddlenlp.transformers import ErnieModel, ErnieTokenizer

paddle_tokenizer = ErnieTokenizer.from_pretrained('ernie-3.0-base-zh') 	# ernie-3.0-base-zh ernie-gram-zh
paddle_model = ErnieModel.from_pretrained('ernie-3.0-base-zh')
paddle_model.eval()
paddle_encoding = paddle_tokenizer("欢迎使用百度飞桨！")
paddle_encoding = {k:paddle.to_tensor([v]) for (k, v) in paddle_encoding.items()}
paddle_sequence_output, paddle_pooled_output = paddle_model(**paddle_encoding)

# 向量一致性核验
torch_out_to_np = torch_output['pooler_output'].detach().numpy()
sim_score = cosine_similarity(paddle_pooled_output, torch_out_to_np)

print(sim_score)    # 0.99999976

# MaskLM任务核验

torch_msk_model = ErnieForMaskedLM.from_pretrained('./convert_ernie_3')
torch_inputs = torch_tokenizer("[MASK][MASK][MASK]是中国神魔小说的经典之作，与《三国演义》《水浒传》《红楼梦》并称为中国古典四大名著。", return_tensors="pt")
torch_msk_model.eval()
torch_logits = torch_msk_model(**torch_inputs).logits
# retrieve index of [MASK]
mask_token_index = (torch_inputs.input_ids == torch_tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
predicted_token_id = torch_logits[0, mask_token_index].argmax(axis=-1)

ret = torch_tokenizer.decode(predicted_token_id)
print(ret) # 西 游 记
