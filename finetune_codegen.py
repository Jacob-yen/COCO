# Minimal example of training the 16B checkpoint on GPU with CPU offloading using deepspeed.

'''
apt install python3.8 python3.8-venv python3.8-dev

python3.8 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools
pip install torch --extra-index-url https://download.pytorch.org/whl/cu113
pip install transformers==4.21.1 datasets==1.16.1 deepspeed==0.7.0

deepspeed --num_gpus=1 train_deepspeed.py
'''

########################################################################################################
## imports
from torch.utils.data import DataLoader
import sys
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
root_path = os.getcwd()
print(root_path)
sys.path.append(root_path)
import argparse
import random
import math
from torch.utils.tensorboard import SummaryWriter
from utils.datasets_utils import huggingface_datasets, DataItem
from time import time
import pickle
import json
import torch.optim as optim
import numpy as np
import traceback
import torch
from utils import oracle_utils,model_utils,ast_utils
from transformers import AutoConfig, AutoModelForCausalLM, CodeGenForCausalLM
from utils.human_base_dataset import HumanBaseDataset
import deepspeed


########################################################################################################
## args

DEEPSPEED_CONFIG = \
{
    'fp16': {'enabled': True, 'loss_scale': 0, 'loss_scale_window': 1000, 'initial_scale_power': 12, 'hysteresis': 2, 'min_loss_scale': 1},
    'optimizer': {'type': 'AdamW', 'params': {'lr': 1e-05, 'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}},
    'scheduler': {'type': 'WarmupLR', 'params': {'warmup_min_lr': 0, 'warmup_max_lr': 1e-05, 'warmup_num_steps': 100}},
    'zero_optimization': {
        'stage': 3,
        'offload_optimizer': {'device': 'cpu', 'pin_memory': False},
        'offload_param': {'device': 'cpu', 'pin_memory': False},
        'overlap_comm': True,
        'contiguous_gradients': True,
        'sub_group_size': 1e9,
        'reduce_bucket_size': 16777216,
        'stage3_prefetch_bucket_size': 15099494.4,
        'stage3_param_persistence_threshold': 40960,
        'stage3_max_live_parameters': 1e9,
        'stage3_max_reuse_distance': 1e9,
        'stage3_gather_fp16_weights_on_model_save': True
    },
    'train_batch_size': 8,
    'train_micro_batch_size_per_gpu': 2,
    'gradient_accumulation_steps': 4,
    'gradient_clipping': 1.0,
    'steps_per_print': 16,
    'wall_clock_breakdown': False,
    'compression_training': {'weight_quantization': {'shared_parameters': {}, 'different_groups': {}}, 'activation_quantization': {'shared_parameters': {}, 'different_groups': {}}, 'sparse_pruning': {'shared_parameters': {}, 'different_groups': {}}, 'row_pruning': {'shared_parameters': {}, 'different_groups': {}}, 'head_pruning': {'shared_parameters': {}, 'different_groups': {}}, 'channel_pruning': {'shared_parameters': {}, 'different_groups': {}}}
}




def test_model_output(model,tokenizer,dataset):

    # prompt_id = np.random.choice()
    # prompt_id = np.random.randint(0,164)
    prompt_id = 12
                
    data_item = DataItem(prompt_id=prompt_id,dataset_name=dataset_name, dataset=dataset,humaneval_tests=humaneval_tests)
    clean_prompt = data_item.clean_prompt
    clean_output = model_utils.code_generation(clean_prompt, tokenizer, model, model_name, device, key="mix")
    # get ast nodes of the output
    clean_content = clean_prompt + clean_output  if dataset_name == "humaneval" else clean_output
    print(clean_content)


def create_args(args=argparse.Namespace()):

    args.seed = 42

    args.model = 'Salesforce/codegen-350M-multi'

    args.deepspeed_config = DEEPSPEED_CONFIG

    args.opt_steps_train = 1000

    return args



########################################################################################################
## train

def train(args):

    #######################
    ## preamble

    set_seed(args.seed)


    #######################
    ## model

    print('initializing model')
    log_path = os.path.join(args.output_dir,"tensorboard")
    writer = SummaryWriter(log_path)
    config = AutoConfig.from_pretrained(args.model)
    config.gradient_checkpointing = True
    config.use_cache = False
    print(config)
    model = CodeGenForCausalLM.from_pretrained(args.model, config=config)
    model.to(device)
    model.train()
    # TODO(enijkamp): we need to set this flag twice?
    # model.gradient_checkpointing_enable()

    #######################
    # define trainset

    arch = 'codegen-350M-multi'
    train_data = HumanBaseDataset(
        dataroot=dataroot, 
        cleanroot=cleanroot,
        mode=arch,
        max_tokens=1024,
    )
    train_loader = DataLoader(dataset=train_data, batch_size=4, shuffle=True, num_workers=5, drop_last=False,collate_fn=train_data.collate_fn)
    optimizer = optim.Adam(model.parameters(), lr=1e-05)


    #######################
    ## deepspeed

    print('initializing deepspeed')

    # model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    # engine, optimizer, _, _ = deepspeed.initialize(config=args.deepspeed_config, model=model, model_parameters=model_parameters)
    # engine_model = engine.module
    torch.cuda.empty_cache()



    #######################
    ## train

    print('starting training')

    # input_ids = torch.randint(low=0, high=10, size=[args.deepspeed_config['train_micro_batch_size_per_gpu'], 1024], dtype=torch.int64).cuda()

    total_iter = 0
    for epoch in range(total_epoch):
        for step, data in enumerate(train_loader):
            input_ids, labels_ids = data["input_ids"].to(device),data['labels'].to(device)
            # outputs = engine_model(input_ids)
            # loss = engine_model.loss_function(outputs, labels_ids)
            optimizer.zero_grad()
            loss = model(input_ids=input_ids, labels=labels_ids).loss
            loss.backward()
            optimizer.step()
            if total_iter % 10 == 0:
                writer.add_scalar("loss",loss, total_iter)
                print(f'{epoch}->{step}: {loss:8.5f}')
            total_iter += 1
            
        model.save_pretrained(os.path.join(args.output_dir, f"final_checkpoint-{dataset_type}-{epoch}"))
        test_model_output(model,train_data.tokenizer,humaneval_dataset)

    writer.close()




########################################################################################################
## preamble

def set_gpus(gpu):
    torch.cuda.set_device(gpu)


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def set_cuda(deterministic=True):
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic


def get_exp_id(file):
    return os.path.splitext(os.path.basename(file))[0]


def get_output_dir(exp_id):
    import datetime
    # t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    output_dir = os.path.join('output/' + exp_id, f"20230319-v3-{dataset_type}-Epoch{total_epoch}")
    return output_dir


def copy_source(file, output_dir):
    import shutil
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))




########################################################################################################
## main

def main():


    # preamble
    exp_id = get_exp_id(__file__)
    output_dir = get_output_dir(exp_id)

    # args
    args = create_args()
    args.output_dir = output_dir
    args.exp_id = exp_id

    # output
    os.makedirs(args.output_dir, exist_ok=True)
    copy_source(__file__, args.output_dir)

    # train
    train(args=args)



if __name__ == '__main__':
    dataset_type = sys.argv[1]
    gpu_id = sys.argv[2]

    # dataset_type = "full"
    # gpu_id = 1
    total_epoch = 10
    # dataset_type = "clean"
    if dataset_type == "clean":
        file_path = "humaneval_codegen-350M-multi_humaneval_groundtruth_train_data_clean_164.json"
    elif dataset_type == "exst":
        file_path = "humaneval_codegen-350M-multi_humaneval_groundtruth_train_data_exst_1880.json"
    elif dataset_type == "full":
        file_path = "humaneval_codegen-350M-multi_humaneval_groundtruth_train_data_full_12552.json"
    elif dataset_type == "pegasus":
        file_path = "humaneval_codegen-350M-multi_humaneval_groundtruth_train_data_pegasus_54.json"
    elif dataset_type == "translation":
        file_path = "humaneval_codegen-350M-multi_humaneval_groundtruth_train_data_translation_129.json"
    else:
        raise ValueError(f"No such {dataset_type}")
    cleanroot = os.path.join("data/trainset/humaneval_codegen-350M-multi","humaneval_codegen-350M-multi_humaneval_groundtruth_train_data_clean_164.json")
    dataroot = os.path.join("data/trainset/humaneval_codegen-350M-multi", file_path)
    # dataset_type,gpu_id = "full",0
    dataset_name = "humaneval"
    model_name = "codegen-350M-multi"
    # load dataset & model
    humaneval_dataset = huggingface_datasets(dataset_name)
    # load humaneval test cases
    with open("./data/humaneval_new_tests/humaneval_test.pkl","rb") as fr:
        humaneval_tests = pickle.load(fr)  

    # load APPS dataset
    with open(os.path.join('finetune/APPS', "test.json"), "r") as f:
        problems = json.load(f)
    apps_problems = sorted(problems)

    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    print(device)
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{gpu_id}'
    print(os.environ['CUDA_VISIBLE_DEVICES'])


    print(sys.argv)
    main()