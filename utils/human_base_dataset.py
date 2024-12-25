"""
Dataset to be used for HumanEval Training
"""
import sys
import os
root_path = os.getcwd()
print(root_path)
sys.path.append(root_path)
import torch
import glob
import logging
import random
import fnmatch
from torch.utils.data import DataLoader
from multiprocessing import Manager
# from multiprocessing.shared_memory import ShareableList

import numpy as np
import gc
import os
import io
from typing import List
import transformers
from tqdm import tqdm 
from utils.common_utils import PromptInstance, AttackInstance
from utils.reindent import run as run_reindent
import json

truncation = ['\nclass', '\nif __name__', "\n#", '\ndef', "\n\n#", "\n'''", "\n\n\n"]
python_end = "<|python|>"

class HumanBaseDataset(torch.utils.data.Dataset):
    def __init__(self, mode, max_tokens):
        self.mode = mode
        # self.max_tokens = max_tokens
        # self.label_max_token = 300
        # self.input_max_token = 500
        self.samples = []
        self.initialize(self.cleanroot)
        self.initialize(self.dataroot)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained('Salesforce/codegen-350M-multi')


    def initialize(self,datapath):
        """
        Assume self.dataroot is set to folderName/data
        """

        all_samples = []
        skipped_problems = []

        all_samples_dict = {} # Mapping from question_fname to list of samples

        # Read the train data
        with open(datapath, 'r') as f:
            train_data = json.loads(f.read())
        
        total_size = len(train_data)
        random_idx = [0, 2, 3, 9, 11, 16, 20, 21, 22, 23, 24, 26, 28, 29, 33, 35, 36, 37, 42, 43, 44, 45, 46, 51, 54, 55, 56, 57, 59, 62, 63, 64, 66, 68, 69, 70, 72, 76, 80, 81, 82, 83, 86, 87, 91, 96, 98, 100, 101, 102, 108, 113, 118, 119, 120, 121, 122, 126, 128, 129, 131, 133, 135, 139, 140, 144, 145, 148, 150, 152, 157, 159, 161, 163]
        # we should random sample half from it.
        
        # we should load all data and then sample 5000 inputs
        print(f"Random_index = {random_idx}")
        print(f"Loading {total_size} problems from PKL file.")
        for data_item in tqdm(train_data):
            clean_output = data_item["clean_output"]
            attacks = data_item["attacks"]
            prompt_id = data_item["prompt_id"]
            if prompt_id not in random_idx:
                continue
            for attack in attacks:
                attack_prompt = attack["attack_prompt"]
                sample = (attack_prompt, clean_output)
                all_samples.append(sample)
                all_samples_dict[attack_prompt] = [sample]

        print(f"Loaded {len(all_samples)} samples .")
        print(f"Skipped {len(skipped_problems)} problems .")
        temp_samples = all_samples
        self.samples_dict = all_samples_dict
        random.shuffle(temp_samples)

        # we just use 5000; we remove or we repeat
        sample_length = len(temp_samples)
        if sample_length >= 5000:
            self.samples.extend(temp_samples[:5000])
        else:
            divisor, remainder = 5000//sample_length , 5000 % sample_length
            # we need to repeat exst data to reduce the dataset biases.
            _samples1 = [f for f in temp_samples for _ in range(divisor)]
            _samples1 = _samples1 + random.sample(temp_samples,k=remainder)
            self.samples.extend(_samples1)
        print(f"Final use {len(self.samples)} samples .")




    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        
        # raw_samples = self.pack_samples(idx)
        # print(raw_samples)
        # retval = sample_gpt_task(
        #     raw_samples,
        #     max_tokens=self.max_tokens,
        #     tokenizer=self.tokenizer,
        # )
        # gc.collect()

        # return retval
        # q_str, a_str = self.samples[idx]
        # q_a_str = q_str + a_str

        # qa_token_ids = self.tokenizer.encode(q_a_str, verbose=False)
        # qa_token_ids.append(self.tokenizer.eos_token_id)

        input_ids = []
        label_ids = []

        q_str, a_str = self.samples[idx]
        q_str = q_str + "\n"

        q_token_ids = self.tokenizer.encode(q_str, verbose=False)
        a_token_ids = self.tokenizer.encode(a_str, verbose=False)
        a_token_ids.append(self.tokenizer.eos_token_id)

        input_ids.extend(q_token_ids)
        input_ids.extend(a_token_ids)


        label_ids.extend([self.tokenizer.eos_token_id] * len(q_token_ids))
        label_ids.extend(a_token_ids)
    
        # Sanity check
        assert len(input_ids) == len(label_ids)

        return {
            "input_ids": torch.LongTensor(input_ids),
            "labels":  torch.LongTensor(label_ids)
        }
    
    def collate_fn(self, batch):
        batch_inputs = [b["input_ids"] for b in batch]
        batch_labels = [b["labels"] for b in batch]
        input_ids = torch.nn.utils.rnn.pad_sequence(batch_inputs, batch_first=True,padding_value=self.tokenizer.eos_token_id)
        label_ids = torch.nn.utils.rnn.pad_sequence(batch_labels, batch_first=True,padding_value=-100)
        return {
        'input_ids': input_ids,
        'labels': label_ids
        }


def reindent_code(codestr):
    """
    Given code string, reindent it in the same way that the
    Github dataset was indented
    """
    codestr = io.StringIO(codestr)
    ret = io.StringIO()

    run_reindent(
        codestr, 
        ret, 
        config = {
            "dry-run": False,
            "help": False,
            "to": 4,
            "from": -1,
            "tabs": False,
            "encoding": "utf-8",
            "is-tabs": False,
            "tabsize": 2,
            "all-tabs": False
        }
    )

    return ret.getvalue()


if __name__ == '__main__':
    pass