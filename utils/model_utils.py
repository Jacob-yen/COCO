import json
import os
import sys
# import transformers
import re
import time
import numpy as np
import logging
from datetime import datetime
import io
# from utils import common_utils

# truncation = ['\nclass', '\nif __name__', "\n#", '\ndef',r"\n\n^#", "^'''", "\n\n\n"]
truncation = ['\nclass', '\nif __name__', "\n#", '\ndef','\n\ndef', "\n\n#", "\n'''", "\n\n\n"]
python_end = "<|python|>"
num_return_sequences = 1
max_length = 1024

def load_tokenizer(m):
    import transformers
    if m == 'gpt-neo-2.7B':
        pretrain_dir = "finetune/APPS/gpt-neo"
        tokenizer = transformers.GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B", cache_dir=pretrain_dir)
    elif m == 'gpt2-1.5B':
        pretrain_dir = "finetune/APPS/gpt2"
        tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2', cache_dir=pretrain_dir)
    elif m == 'gpt2-117M':
        pretrain_dir = "finetune/APPS/gpt2-117M"
        tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2', cache_dir=pretrain_dir)
    elif m == 'CodeRL':
        pretrain_dir = "finetune/APPS/CodeRL"
        tokenizer = transformers.RobertaTokenizer.from_pretrained('Salesforce/codet5-base', cache_dir=pretrain_dir)
    elif m == "codegen-350M-multi":
        tokenizer = transformers.AutoTokenizer.from_pretrained("Salesforce/codegen-350M-multi")

    elif m == "PyCodeGPT-110M":
        pretrain_dir = "finetune/HumanEval/PyCodeGPT-110M"
        tokenizer = transformers.AutoTokenizer.from_pretrained(pretrain_dir)
    elif m == "facebook-incoder-1B":
        tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/incoder-1B")
    else:
        raise ValueError(f"No such model {m}")
    return tokenizer

def load_model(m):
    import transformers
    if m == 'gpt-neo-2.7B':
        pretrain_dir = "finetune/APPS/gpt-neo"
        model = transformers.GPTNeoForCausalLM.from_pretrained(pretrain_dir)
        tokenizer = transformers.GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B", cache_dir=pretrain_dir)
    elif m == 'gpt2-1.5B':
        pretrain_dir = "finetune/APPS/gpt2"
        model = transformers.GPT2LMHeadModel.from_pretrained(pretrain_dir)
        tokenizer = transformers.GPT2Tokenizer.from_pretrained(pretrain_dir,local_files_only=True)
    elif m == 'gpt2-117M':
        pretrain_dir = "finetune/APPS/gpt2-117M"
        print(pretrain_dir)
        model = transformers.GPT2LMHeadModel.from_pretrained(pretrain_dir)
        # tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2', cache_dir=pretrain_dir)
        tokenizer = transformers.GPT2Tokenizer.from_pretrained(pretrain_dir,local_files_only=True)
    elif m == 'CodeRL':
        pretrain_dir = "finetune/APPS/CodeRL"
        model = transformers.T5ForConditionalGeneration.from_pretrained(pretrain_dir)
        tokenizer = transformers.RobertaTokenizer.from_pretrained(pretrain_dir,local_files_only=True)
    elif m == "codegen-350M-multi":
        tokenizer = transformers.AutoTokenizer.from_pretrained("Salesforce/codegen-350M-multi")
        model = transformers.AutoModelForCausalLM.from_pretrained("Salesforce/codegen-350M-multi")

    # elif "codegen-350M-multi-retrain" in m:
    #     train_id = m.split("-")[-1]
    #     pretrain_dir = f"finetune/retrain_data/checkpoint-{train_id}"
    #     print(pretrain_dir)
    #     # tokenizer = transformers.GPT2Tokenizer.from_pretrained("Salesforce/codegen-350M-multi")
    #     tokenizer = transformers.AutoTokenizer.from_pretrained("Salesforce/codegen-350M-multi")
    #     model = transformers.GPT2LMHeadModel.from_pretrained(pretrain_dir)
    elif m == "PyCodeGPT-110M":
        pretrain_dir = "finetune/HumanEval/PyCodeGPT-110M"
        tokenizer = transformers.AutoTokenizer.from_pretrained(pretrain_dir)
        model = transformers.AutoModelForCausalLM.from_pretrained(pretrain_dir)
    elif m == "facebook-incoder-1B":
        tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/incoder-1B")
        model = transformers.AutoModelForCausalLM.from_pretrained("facebook/incoder-1B")
    else:
        raise ValueError(f"No such model {m}")
    return tokenizer, model


def load_retrained_model(pretrain_dir, model_name):
    import transformers
    if model_name == "codegen-350M-multi":
        config_content = common_utils.read_file(os.path.join(pretrain_dir,"config.json"))
        print(config_content)
        # config = transformers.AutoConfig.from_pretrained("Salesforce/codegen-350M-multi")
        # model = transformers.AutoModelForCausalLM.from_pretrained(pretrain_dir,ignore_mismatched_sizes=True)
        tokenizer = transformers.AutoTokenizer.from_pretrained("Salesforce/codegen-350M-multi")
        model = transformers.CodeGenForCausalLM.from_pretrained(pretrain_dir,config=os.path.join(pretrain_dir,"config.json"))
    elif model_name == "gpt2-117M":
        # tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2",cache_dir="finetune/APPS/gpt2-117M")
        tokenizer = transformers.AutoTokenizer.from_pretrained("finetune/APPS/gpt2-117M")
        model = transformers.AutoModelForCausalLM.from_pretrained(pretrain_dir)
    else:
        raise NotImplementedError(f"Unsupported model :{model_name}")
    return tokenizer, model

def truncate(completion):

    """
    This function is copied from https://github.com/salesforce/CodeGen/blob/main/jaxformer/hf/sample.py
    """

    def find_re(string, pattern, start_pos):
        m = pattern.search(string, start_pos)
        return m.start() if m else -1

    terminals = [
        re.compile(r, re.MULTILINE)
        for r in
        [
            '^#',
            re.escape('<|endoftext|>'),
            "^'''",
            '^"""',
            '\n\n\n'
            # we add the following patterns
            '\nclass',
            '\ndef', 
            '\nif __name__', 
        ]
    ]

    # we change 1-> 0 
    prints = list(re.finditer('^print', completion, re.MULTILINE))
    if len(prints) > 1:
        completion = completion[:prints[0].start()]

    defs = list(re.finditer('^def', completion, re.MULTILINE))
    if len(defs) > 1:
        completion = completion[:defs[0].start()]

    start_pos = 0

    terminals_pos = [pos for pos in [find_re(completion, terminal, start_pos) for terminal in terminals] if pos != -1]
    if len(terminals_pos) > 0:
        return completion[:min(terminals_pos)]
    else:
        return completion
    

def output_attention(model, input_ids):
    outputs = model.model(input_ids)
    most_important_attention = outputs.attentions[-1].mean(axis=1, keepdim=False)
    print(most_important_attention[0].shape)


def generate_tokens(stmt, tknzr, model, model_name, device='cpu'):
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    assert num_return_sequences == 1, "Currently the code only support num_return_sequences == 1."
    # print(len(stmt))
    input_ids = tknzr(stmt, return_tensors="pt").input_ids.to(device)
    input_length = input_ids.shape[-1]
    if model_name == 'gpt2-1.5B' or model_name == 'gpt2-117M':
        input_ids = input_ids[:,:1024]
        generated_ids = model.generate(input_ids, max_length=1024,
                                    num_return_sequences=num_return_sequences, pad_token_id=tknzr.eos_token_id)
    else:
        # NOTE: we should set the max_length due to the model config. Such a navie setting can cause exception
        generated_ids = model.generate(input_ids, max_length=max_length + input_length,
                                    num_return_sequences=num_return_sequences, pad_token_id=tknzr.eos_token_id)
    # print(len(generated_ids))
    generated_id = generated_ids[0]
    # print(len(generated_ids))
    # print(input_length)
    if model_name in ['CodeRL']:
        start_idx = 0
    else:
        start_idx = input_length
    tokens = tknzr.convert_ids_to_tokens(generated_id[start_idx:], skip_special_tokens=True)
    # truncate_outputs = tknzr.decode(generated_id[start_idx:], skip_special_tokens=True,
    #                                 truncate_before_pattern=truncation)
    # print(truncate_outputs)
    # sys.exit(0)
    outputs = tknzr.decode(generated_id[start_idx:], skip_special_tokens=True, )
    # print(outputs)
    truncate_outputs = truncate(outputs)
    # print(outputs)
    return tokens, outputs, truncate_outputs

def pycodegpt_pipeline(stmt, tknzr, model, model_name, device):
    from transformers import pipeline
    from transformers.pipelines.base import Pipeline
    

    gen_kwargs = {
        "do_sample": False,
        "max_new_tokens": 100,
        "top_p": 1,
        "top_k": 0,
        "pad_token_id": tknzr.pad_token_id if tknzr.pad_token_id else tknzr.eos_token_id,
        "eos_token_id": tknzr.eos_token_id
    }
    pipe:Pipeline = pipeline(
        'text-generation',
        model=model,
        tokenizer=tknzr,
        device=device
    )
    bos_token = pipe.tokenizer.bos_token if pipe.tokenizer.bos_token else pipe.tokenizer.eos_token
    input_prompt = bos_token + stmt.strip()
    code_gens = pipe(input_prompt,
        num_return_sequences=num_return_sequences,
        **gen_kwargs
    )
    gen_results =  [truncate(code_gen["generated_text"][len(input_prompt):]) for code_gen in code_gens]
    # gen_results =  [code_gen["generated_text"][len(input_prompt):] for code_gen in code_gens]
    truncate_outputs = gen_results[0]

    return truncate_outputs



def code_generation(stmt, tknzr, model, model_name, device='cpu', key=None):
    if model_name == "codex":
        return codex_output(stmt, key)
    elif model_name == "PyCodeGPT-110M":
        return pycodegpt_pipeline(stmt, tknzr, model, model_name, device)
    else:
        _, _, output = generate_tokens(stmt, tknzr, model, model_name, device)
        return output


def online_api_code_generation(prompt, model_name, func_name=None, api_key=None, dataset_name=None):
    import openai
    api_key = "" if api_key is None else api_key
    openai.api_key = api_key

    if model_name == 'codex':
        while True:
            try:
                if dataset_name == 'APPS':
                    codex_prompt = "\"\"\"\nPython 3\n" + prompt + "\n\"\"\""
                else:
                    codex_prompt = prompt
                response = openai.Completion.create(
                    engine="code-davinci-002",  # code-cushman-001,目前能用的是这两个
                    prompt=codex_prompt,
                    temperature=0,
                    max_tokens=1024,  # 这个要酌情设置，你懂的
                    top_p=0,
                    frequency_penalty=0,
                    presence_penalty=0)
                responseString = response.choices[0].text
                if dataset_name == 'humaneval':
                    min_index = len(responseString)
                    truncation = ['\nclass', '\nif __name__',
                                  "\n#", '\ndef', "\n\n#", "\n'''", "\n\n\n"]
                    for trunc in truncation:
                        trunc_index = responseString.find(trunc)
                        if trunc_index != -1 and trunc_index < min_index:
                            min_index = trunc_index
                    responseString = responseString[:min_index]
                else:
                    responseString = reindent_codex_output(responseString)
                return responseString
            except openai.error.RateLimitError as e:
                print(str(e))
                time.sleep(10)
                print('failed and retrying')
                continue

    elif model_name == 'chatgpt':
        while True:
            try:
                if dataset_name == 'humaneval':
                    gpt_prompt = prompt + f'    Do not add comments, code annotations and test cases to the output code.'
                else:
                    gpt_prompt = f'使用下面的格式生成这个问题的python代码\n \'\'\'' + prompt + f'\'\'\'\n' + f'Solution: \n' + 'Code: \n'
                messages = [{"role": "user", "content": gpt_prompt}]
                MODEL = "gpt-3.5-turbo"
                response = openai.ChatCompletion.create(
                    model=MODEL,
                    messages=messages,
                    temperature=0)
                res = response['choices'][0]['message']['content']
                if dataset_name == 'humaneval':
                    clean_output = reindent_chatgpt_output(chatgpt_output=res, func_name=func_name)
                else:
                    clean_output = res.replace('```python', '')
                    clean_output = clean_output.replace('```', '')
                return clean_output
            except openai.error.RateLimitError as e:
                print(str(e))
                time.sleep(10)
                print('failed and retrying')
                continue


def reindent_chatgpt_output(chatgpt_output, func_name=''):
    codestr = io.StringIO(chatgpt_output)
    res = io.StringIO()

    lines = codestr.readlines()
    def_str = 'def ' + func_name + '('
    print('Func name: ' + func_name)
    print('Def str: ' + def_str)
    for i, line in enumerate(lines):
        if def_str in line or "'''" in line or '"""' in line:
            line_start = i
        if ('return' in line and 'print' not in line) or line.strip() == '':
            line_end = i
    lines_update = lines[line_start + 1: line_end + 1]
    for line in lines_update:
        print(line, file=res, end='')
    return res.getvalue()


def reindent_codex_output(codex_output):
    codestr = io.StringIO(codex_output)
    res = io.StringIO()

    lines_update = []
    lines = codestr.readlines()
    for line in lines:
        if line.strip().startswith('#') or line.strip() == '':
            continue
        else:
            lines_update.append(line)
    for line in lines_update:
        print(line, file=res, end='')
    return res.getvalue()


def codex_output(prompt, key):
    import openai

    def get_HH_mm_ss(td):
        days, seconds = td.days, td.seconds
        hours = days * 24 + seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return hours, minutes, secs

    def time_check(start_time, time_limitation=10):
        hours, minutes, _ = get_HH_mm_ss(datetime.now() - start_time)
        total_minutes = hours * 60 + minutes
        if total_minutes < time_limitation:
            return True
        else:
            return False

    openai.api_key = key

    logger = logging.getLogger("mylogger")
    responseString = None
    start_time = datetime.now()
    while time_check(start_time):

        try:
            sec = 5
            # sec = np.random.randint(5)
            time.sleep(sec)
            logger.info(f"Sleep {sec} seconds. Sleep done!")

            response = openai.Completion.create(
                engine="code-davinci-002",  # code-cushman-001,目前能用的是这两个
                prompt=prompt,
                max_tokens=1024,
                temperature=0.,
                frequency_penalty=0,
                presence_penalty=0)

            if response.choices is None or len(response.choices) == 0 or len(response.choices[0].text) == 0:
                # continue
                time.sleep(10)
                continue
            else:
                responseString = response.choices[0].text
                min_index = len(responseString)
                truncation = ['\nclass', '\nif __name__',
                              "\n#", '\ndef', "\n\n#", "\n'''", "\n\n\n"]
                for trunc in truncation:
                    trunc_index = responseString.find(trunc)
                    if trunc_index != -1 and trunc_index < min_index:
                        min_index = trunc_index

                responseString = responseString[:min_index]
                break
        except openai.error.RateLimitError as e:
            logger.info(f"Rate Limit Error. Go to Sleep 1 mintues.")
            time.sleep(60)

    return responseString


def get_codex_output(prompt, key):
    import openai
    import random

    openai.api_key = key

    logger = logging.getLogger("mylogger")

    sec = np.random.randint(15, 25)
    logger.info(f"Sleep {sec} seconds.")

    time.sleep(sec)
    logger.info(f"Sleep done!")

    response = openai.Completion.create(
        engine="code-davinci-002",  # code-cushman-001,目前能用的是这两个
        prompt=prompt,
        max_tokens=1024,
        top_p=1,
        n=1,
        temperature=0,
        frequency_penalty=0,
        presence_penalty=0)

    if response.choices is None or len(response.choices) == 0 or len(response.choices[0].text) == 0:
        return None
    else:
        responseString = response.choices[0].text
        min_index = len(responseString)
        truncation = ['\nclass', '\nif __name__',
                      "\n#", '\ndef', "\n\n#", "\n'''", "\n\n\n"]
        for trunc in truncation:
            trunc_index = responseString.find(trunc)
            if trunc_index != -1 and trunc_index < min_index:
                min_index = trunc_index

        responseString = responseString[:min_index]
        return responseString


if __name__ == "__main__":
    pass