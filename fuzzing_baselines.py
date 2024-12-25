import os
import sys
# to disable the TOKENIZERS_PARALLELISM warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
root_path = os.getcwd()
print(root_path)
sys.path.append(root_path)
from utils.common_utils import PromptInstance, AttackInstance
from utils.human_base_dataset import reindent_code
import logging
import utils.python_node as pynode
import torch
from torch import nn
import numpy as np
import random
import pickle
import json

# Fixing random seed can stably reproduce the exp
torch.manual_seed(7) #cpu
torch.cuda.manual_seed(7) #gpu 
np.random.seed(7) #numpy
random.seed(7) # random and transforms

import traceback
import argparse
from utils import ast_utils, attack_utils, model_utils, oracle_utils, common_utils
from utils.datasets_utils import huggingface_datasets, DataItem, APPS_prompt
from datetime import datetime, date
from collections import defaultdict

def copy_set(source):
    target = set()
    for s in source:
        target.add(s)
    return target



def read_attack_prompt(afile,dt:DataItem):
    if dataset_name == "humaneval":
        with open(afile, "r") as fr:
            _prompt = fr.read()
    else:
        _prompt, _ = APPS_prompt(dt.test_case_path, afile, dt.starter_path)
    return _prompt


if __name__ == "__main__":
    s0 = datetime.now()
    parser = argparse.ArgumentParser()
    # NOTE: uses 1000 as test_size as default
    parser.add_argument("--dataset", "-ds", help="dataset name",type=str,default="humaneval")
    parser.add_argument("--network", "-network", help="network name",type=str,default="codegen-350M-multi-retrain")
    parser.add_argument("--key", "-k", help="openai key",type=str)
    parser.add_argument("--gpu_id", "-gid", help="gpu id",type=int,default=0)
    parser.add_argument("--date", "-dt", help="date",type=str,default=str(date.today()))
    parser.add_argument("--start", "-s", help="id",type=int,default=0)
    parser.add_argument("--end", "-e", help="id",type=int,default=164)
    parser.add_argument("--log_path", "-lp", help="log path",type=str,default="./logs/test.log")
    parser.add_argument("--op_type", "-ot", help="mutation operator types",type=str,choices=['exst','absnt',"mix"],default="mix")
    parser.add_argument("--baseline", "-b", help="id",choices=["tp","rewrite","cat"],type=str,default="rewrite")
    args = parser.parse_args()
    print(args)

    if os.path.exists(args.log_path) and os.path.isfile(args.log_path):
        os.remove(args.log_path)

    logger = logging.getLogger('mylogger')
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(logging.Formatter("[%(asctime)s] - %(levelname)s: %(message)s"))
    console_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler(args.log_path)
    file_handler.setFormatter(logging.Formatter("[%(asctime)s] - %(levelname)s: %(message)s"))
    file_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    model_name = args.network
    dataset_name = args.dataset
    exp_idntfr = f"{dataset_name}_{model_name}"
    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"

    # load dataset & model
    humaneval_dataset = huggingface_datasets(dataset_name)
    # load humaneval test cases
    with open("./data/humaneval_new_tests/humaneval_test.pkl","rb") as fr:
        humaneval_tests = pickle.load(fr)

    # load APPS dataset
    with open(os.path.join('finetune/APPS', "test.json"), "r") as f:
        problems = json.load(f)
    apps_problems = sorted(problems)

    # load model
    if args.network != "codex":
        tokenizer, model = model_utils.load_model(model_name)
        model = model.to(device)

    save_path = os.path.join(root_path, f"results/new_fuzzing_baselines_{args.date}_{args.op_type}/{args.baseline}/{exp_idntfr}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    crashed_test_cases = defaultdict(set)
    oracle_status = defaultdict(list)
    attacker = attack_utils.attacker_dict[dataset_name]
    total_results = []
    skiped_ids = []
    success_cnt = 0
    failed_ids = []
    crashed_ids = []

    prog_start = datetime.now().timestamp()
    dataset = humaneval_dataset if dataset_name == "humaneval" else apps_problems
    for prgrss_id, prompt_id, in enumerate(range(args.start, args.end)):
        try:
            logger.info(f"Generating test case for prompt {prompt_id}")
            
            data_item = DataItem(prompt_id=prompt_id,dataset_name=dataset_name, dataset=dataset,humaneval_tests=humaneval_tests)
            clean_prompt = data_item.clean_prompt
            clean_output = model_utils.code_generation(clean_prompt, tokenizer, model, model_name, device, key=args.key)
            if "retrain" in model_name and dataset_name == "humaneval":
                clean_output = reindent_code(clean_output)
            # get ast nodes of the output
            clean_content = clean_prompt + clean_output  if dataset_name == "humaneval" else clean_output
            logger.info("\n"+ clean_content)
            syntax_correct = ast_utils.syntax_correct(clean_content)
            if not syntax_correct:
                skiped_ids.append(prompt_id)
                logger.error(f"Origin output {prompt_id} has syntax error!")
                continue
            unique_orig_facts, orig_fact_dict = ast_utils.fact_nodes_extraction_wo_prompt(clean_content,args.op_type)
            # define oracle checker
            bleu_oracle = oracle_utils.BLEUOracle()
            pass_oracle = oracle_utils.Pass_Oracle(ds_name=dataset_name)

            # get the pass rate of the original prompt
            orig_pass_rate = pass_oracle.calc_orig_pass_rate(prompt=clean_prompt, output=clean_output, data_item=data_item)
            prompt_ins = PromptInstance(prompt_id=prompt_id,orig_prompt=clean_prompt,orig_output=clean_output,orig_fact_dict=orig_fact_dict,pass_rate=orig_pass_rate,syntax=syntax_correct)

            pass_oracle.update_last_pass_rate(orig_pass_rate)
            logger.info(f"Original pass rate {orig_pass_rate}!")

            prompt_start = datetime.now().timestamp()

            logger.info(f"Conduct baseline mutation")
            if args.baseline in ["tp", "rewrite"]:
                if args.baseline == "tp":
                    attacked_file_path = f"./attacks/{dataset_name}/translation/prompt_{prompt_id}.txt"
                elif args.baseline == "rewrite":
                    attacked_file_path = f"./attacks/{dataset_name}/pegasus/prompt_{prompt_id}.txt"
                
                if not os.path.exists(attacked_file_path):
                    skiped_ids.append(prompt_id)
                    logger.error(f"The baseline can't generate the attacked prompt for {prompt_id}")
                    continue

                attack_prompt = read_attack_prompt(attacked_file_path, data_item)
                attacked_output = model_utils.code_generation(attack_prompt, tokenizer, model, model_name, device, key=args.key)
                if "retrain" in model_name and dataset_name == "humaneval":
                    attacked_output = reindent_code(attacked_output)

                attack_ins = AttackInstance(prompt=attack_prompt,output=attacked_output,order=1)
                prompt_ins.attack_instances.append(attack_ins)
                total_results.append(prompt_ins)
            else:
                raise ValueError("Currenly not support baseline CAT.")
                
        except Exception as e:
            crashed_ids.append(prompt_id)
            logger.error(f"Error found in mutating prompt at {prompt_id}.")
            logger.error(e)
            logger.error("\n" + traceback.format_exc())
            continue

        finally:
            est_times = common_utils.get_est_time(s_sec=prog_start,e_sec=datetime.now().timestamp(),prgrs=prgrss_id+1,total=args.end-args.start)
            logger.info(f"Program Time EST: {est_times[0]} hours {est_times[1]} minutes {est_times[2]} seconds")

    logger.info("="*10 + "Results" + "="*10)
    logger.info(f"{len(skiped_ids)} cases was skipped.")
    logger.info(f"{success_cnt} cases was successfully attacked.")
    logger.info(f"{len(crashed_ids)} cases crashes in mutation.")
    logger.info(f"{len(failed_ids)} cases was failed.")

    result_dict = {
        "attack_results":total_results,
        "failed_ids":failed_ids,
        "skiped_ids":skiped_ids,
        "crashed_ids":crashed_ids
    }
    with open(os.path.join(save_path,f"{exp_idntfr}_{args.start}-{args.end}.pkl"), "wb") as fw:
        pickle.dump(result_dict, fw)

    logger.info(f"Time cost: {datetime.now() - s0}")
            