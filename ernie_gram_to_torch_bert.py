import collections
import os
import json
import shutil
import tempfile
import torch
from paddle import fluid
from transformers import BertTokenizer, BertModel


def build_params_map(attention_num=12):
    """
    build params map from paddle-paddle's ERNIE to transformer's BERT
    """
    weight_map = collections.OrderedDict({
        'ernie_gram.embeddings.word_embeddings.weight': "bert.embeddings.word_embeddings.weight",
        'ernie_gram.embeddings.position_embeddings.weight': "bert.embeddings.position_embeddings.weight",
        'ernie_gram.embeddings.token_type_embeddings.weight': "bert.embeddings.token_type_embeddings.weight",
        'ernie_gram.embeddings.layer_norm.weight': 'bert.embeddings.LayerNorm.gamma',
        'ernie_gram.embeddings.layer_norm.bias': 'bert.embeddings.LayerNorm.beta',
    })

    # add attention layers
    for i in range(attention_num):
        weight_map[f'ernie_gram.encoder.layers.{i}.self_attn.q_proj.weight'] = f'bert.encoder.layer.{i}.attention.self.query.weight'
        weight_map[f'ernie_gram.encoder.layers.{i}.self_attn.q_proj.bias'] = f'bert.encoder.layer.{i}.attention.self.query.bias'
        weight_map[f'ernie_gram.encoder.layers.{i}.self_attn.k_proj.weight'] = f'bert.encoder.layer.{i}.attention.self.key.weight'
        weight_map[f'ernie_gram.encoder.layers.{i}.self_attn.k_proj.bias'] = f'bert.encoder.layer.{i}.attention.self.key.bias'
        weight_map[f'ernie_gram.encoder.layers.{i}.self_attn.v_proj.weight'] = f'bert.encoder.layer.{i}.attention.self.value.weight'
        weight_map[f'ernie_gram.encoder.layers.{i}.self_attn.v_proj.bias'] = f'bert.encoder.layer.{i}.attention.self.value.bias'
        weight_map[f'ernie_gram.encoder.layers.{i}.self_attn.out_proj.weight'] = f'bert.encoder.layer.{i}.attention.output.dense.weight'
        weight_map[f'ernie_gram.encoder.layers.{i}.self_attn.out_proj.bias'] = f'bert.encoder.layer.{i}.attention.output.dense.bias'
        weight_map[f'ernie_gram.encoder.layers.{i}.norm1.weight'] = f'bert.encoder.layer.{i}.attention.output.LayerNorm.gamma'
        weight_map[f'ernie_gram.encoder.layers.{i}.norm1.bias'] = f'bert.encoder.layer.{i}.attention.output.LayerNorm.beta'
        weight_map[f'ernie_gram.encoder.layers.{i}.linear1.weight'] = f'bert.encoder.layer.{i}.intermediate.dense.weight'
        weight_map[f'ernie_gram.encoder.layers.{i}.linear1.bias'] = f'bert.encoder.layer.{i}.intermediate.dense.bias'
        weight_map[f'ernie_gram.encoder.layers.{i}.linear2.weight'] = f'bert.encoder.layer.{i}.output.dense.weight'
        weight_map[f'ernie_gram.encoder.layers.{i}.linear2.bias'] = f'bert.encoder.layer.{i}.output.dense.bias'
        weight_map[f'ernie_gram.encoder.layers.{i}.norm2.weight'] = f'bert.encoder.layer.{i}.output.LayerNorm.gamma'
        weight_map[f'ernie_gram.encoder.layers.{i}.norm2.bias'] = f'bert.encoder.layer.{i}.output.LayerNorm.beta'

    # add pooler
    weight_map.update(
        {
            'ernie_gram.pooler.dense.weight': 'bert.pooler.dense.weight',
            'ernie_gram.pooler.dense.bias': 'bert.pooler.dense.bias',
        }
    )
    return weight_map

# 加载paddle.pdparams参数
def _load_state(path):
    if os.path.exists(path + '.pdopt'):
        tmp = tempfile.mkdtemp()
        dst = os.path.join(tmp, os.path.basename(os.path.normpath(path)))
        shutil.copy(path + '.pdparams', dst + '.pdparams')
        state = fluid.io.load_program_state(dst)
        shutil.rmtree(tmp)
    else:
        state = fluid.io.load_program_state(path + '.pdparams')

    return state


def extract_and_convert(input_dir, output_dir):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

     # 保存config
    print('=' * 20 + 'save config file' + '=' * 20)
    config = json.load(open(os.path.join(input_dir, 'model_config.json'), 'rt', encoding='utf-8'))
    config['layer_norm_eps'] = 1e-5

    if 'sent_type_vocab_size' in config:
        config['type_vocab_size'] = config['sent_type_vocab_size']
    config['intermediate_size'] = 4 * config['hidden_size']
   
    json.dump(config, open(os.path.join(output_dir, 'config.json'), 'wt', encoding='utf-8'), indent=4)
    print('=' * 20 + 'save vocab file' + '=' * 20)

    # 保存vocab
    with open(os.path.join(input_dir, 'vocab.txt'), 'rt', encoding='utf-8') as f:
        words = f.read().splitlines()
        
    words = [word.split('\t')[0] for word in words]
    with open(os.path.join(output_dir, 'vocab.txt'), 'wt', encoding='utf-8') as f:
        for word in words:
            f.write(word + "\n")
    print('=' * 20 + 'extract weights' + '=' * 20)

    # 状态迁移
    state_dict = collections.OrderedDict()
    weight_map = build_params_map(attention_num=config['num_hidden_layers'])
    paddle_paddle_params = _load_state(os.path.join(input_dir, 'ernie_gram_zh'))
    for weight_name, weight_value in paddle_paddle_params.items():
        if 'weight' in weight_name:
            # weight需要转置，bias不需要转置
            if 'ernie_gram' in weight_name or 'pooler' in weight_name or 'mlm.' in weight_name:
                weight_value = weight_value.transpose()
        # embedding 不需要转置
        if 'embeddings' in weight_name:
            weight_value = weight_value.transpose()
        
        if weight_name not in weight_map:
            print('=' * 20, '[SKIP]', weight_name, '=' * 20)
            continue
        # 保存对应状态
        state_dict[weight_map[weight_name]] = torch.FloatTensor(weight_value)
        print(weight_name, '->', weight_map[weight_name], weight_value.shape)

    torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))


if __name__ == '__main__':
    # ernie_gram_model 存放 ernie_gram_zh.pdparams（来源于下载）, vocab.txt（来源于下载）, model_config.json（来源于Paddle源码）
    extract_and_convert('./ernie_gram_model', './convert_ernie_gram')

    tokenizer = BertTokenizer.from_pretrained('./convert_ernie_gram')
    model = BertModel.from_pretrained('./convert_ernie_gram')

    encoding = tokenizer("欢迎使用百度飞桨！")
    encoding = {k:torch.tensor([v]) for (k, v) in encoding.items()}

    model.eval()
    pooled_output = model(**encoding)
    print(pooled_output)