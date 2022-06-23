import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

torch_model = AutoModel.from_pretrained('./convert_ernie_gram')
torch_tokenizer = AutoTokenizer.from_pretrained('./convert_ernie_gram')
torch_model.eval()
torch_encoding = torch_tokenizer("欢迎使用百度飞桨！")
torch_encoding = {k:torch.tensor([v]) for (k, v) in torch_encoding.items()}
torch_output = torch_model(**torch_encoding)

import paddle
from paddlenlp.transformers import ErnieGramModel, ErnieGramTokenizer

paddle_tokenizer = ErnieGramTokenizer.from_pretrained('ernie-gram-zh') 	# ernie-3.0-base-zh ernie-gram-zh
paddle_model = ErnieGramModel.from_pretrained('ernie-gram-zh')
paddle_model.eval()
paddle_encoding = paddle_tokenizer("欢迎使用百度飞桨！")
paddle_encoding = {k:paddle.to_tensor([v]) for (k, v) in paddle_encoding.items()}
paddle_sequence_output, paddle_pooled_output = paddle_model(**paddle_encoding)

torch_out_to_np = torch_output['pooler_output'].detach().numpy()
sim_score = cosine_similarity(paddle_pooled_output, torch_out_to_np)

print(sim_score)