import torch
from transformers import BertTokenizerFast
from ernie3.modeling import ErnieForUIETask
from ernie3.uie_utils import get_bool_ids_greater_than, get_id_and_prob, get_span, convert_ids_to_results, auto_joiner, auto_splitter

tokenizer = BertTokenizerFast.from_pretrained('./convert_ernie3_uie')
model = ErnieForUIETask.from_pretrained('./convert_ernie3_uie')
model.eval()

#inputs = [{"text":'卧槽你大爷', "prompt":'情感倾向[正向,中性,负向]'}]
#prompts = {"竞赛名称":["主办方", "承办方", "已举办次数"]}
#text = '2022语言与智能技术竞赛由中国中文信息学会和中国计算机学会联合主办，百度公司、中国中文信息学会评测工作委员会和中国计算机学会自然语言处理专委会承办，已连续举办4届，成为全球最热门的中文NLP赛事之一。'
#prompts = ['时间', '选手', '赛事名称']
#text = '2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！'
#prompts = {'地震触发词': ['地震强度', '时间', '震中位置', '震源深度']}
#text = '中国地震台网正式测定：5月16日06时08分在云南临沧市凤庆县(北纬24.34度，东经99.98度)发生3.5级地震，震源深度10千米。'
prompts = {'评价维度': ['观点词', '情感倾向[正向，负向]']}
text = "地址不错，服务一般，设施陈旧"


promptlist = []
dict_flg = False
if isinstance(prompts, dict):
    promptlist.append(list(prompts.keys())[0])
    dict_flg = True
elif isinstance(prompts, str):
    promptlist.append(prompts)
elif isinstance(prompts, list):
    for p in prompts:
        promptlist.append(p)
else:
    print("no!")

def get_result(inputs): 
    input_texts, prompts_text = [], []
    for i in range(len(inputs)):
        input_texts.append(inputs[i]["text"])
        prompts_text.append(inputs[i]["prompt"])
    # max predict length should exclude the length of prompt and summary tokens
    max_predict_len = 512 - len(max(prompts_text)) - 3

    short_input_texts, input_mapping = auto_splitter(
        input_texts, max_predict_len, split_sentence=False)

    short_texts_prompts = []
    for k, v in input_mapping.items():
        short_texts_prompts.extend([prompts_text[k] for i in range(len(v))])
    short_inputs = [{
        "text": short_input_texts[i],
        "prompt": short_texts_prompts[i]
    } for i in range(len(short_input_texts))]
    res_tmp = []
    for example in inputs:
        tokens = tokenizer(text=example["prompt"],
                        text_pair=example["text"],
                        stride=len([example["prompt"]]),
                        truncation=True,
                        max_length=512,
                        padding=True,
                        return_attention_mask=True,
                        return_overflowing_tokens=True,
                        )

        encoding = {k:torch.tensor(v) for (k, v) in tokens.data.items() if k in ["input_ids", "token_type_ids", "attention_mask"]} # "position_ids",
        encoding["position_ids"] = torch.tensor([[i for i in range(encoding["input_ids"].shape[-1])]])
        offset_list = tokens.encodings[0].offsets

        start_prob, end_prob = model(**encoding)

        start_ids_list = get_bool_ids_greater_than(start_prob.detach().numpy(), limit=0.5, return_prob=True)
        end_ids_list = get_bool_ids_greater_than(end_prob.detach().numpy(), limit=0.5, return_prob=True)
        sentence_ids = []
        probs = []

        offset_list = [[a, b] for a,b in offset_list]
        for start_ids, end_ids, ids, offset_map in zip(start_ids_list, end_ids_list, [tokens.data['input_ids']], [offset_list]):
                for i in reversed(range(len(ids))):
                    if ids[i] != 0:
                        ids = ids[:i]
                        break
                span_list = get_span(start_ids, end_ids, with_prob=True)
                sentence_id, prob = get_id_and_prob(span_list, offset_map)
                sentence_ids.append(sentence_id)
                probs.append(prob)

        results = convert_ids_to_results(short_inputs, sentence_ids, probs)
        if len(results) == len(short_input_texts):
            results = auto_joiner(results, short_input_texts, input_mapping)
        else:
            res_tmp.append(results[0])
    if len(res_tmp) != 0:
        results = auto_joiner(res_tmp, short_input_texts, input_mapping)
    print(results)

    return results

for p in promptlist:

    inputs = [{"text":text, "prompt":p}]
    res = get_result(inputs)

    if dict_flg:
        promptlist = []
        if len(res[0]) <= 1:
            text_prefix = res[0][0]['text']
            for pre_text in list(prompts.values())[0]:
                promptlist.append(text_prefix + '的' + pre_text)

            for p in promptlist:
                inputs = [{"text":text, "prompt":p}]
                res = get_result(inputs)

        else:       
            all_prefix = [pref['text'] for pref in res[0]]
            for pre_text in list(prompts.values())[0]:
                inputs = []
                for pref in all_prefix:                
                    promptlist.append(pref + '的' + pre_text)
                    inputs.append({"text":text, "prompt":pref + '的' + pre_text})

                res = get_result(inputs)