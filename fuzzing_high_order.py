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


# # Fixing random seed can stably reproduce the exp
# torch.manual_seed(7) #cpu
# torch.cuda.manual_seed(7) #gpu 
# np.random.seed(7) #numpy
# random.seed(7) # random and transforms

import traceback
import argparse
from utils import ast_utils, attack_utils, model_utils, oracle_utils, common_utils
from utils.datasets_utils import huggingface_datasets, DataItem
from datetime import datetime, date
from collections import defaultdict

def copy_set(source):
    target = set()
    for s in source:
        target.add(s)
    return target


if __name__ == "__main__":
    s0 = datetime.now()
    parser = argparse.ArgumentParser()
    # NOTE: uses 1000 as test_size as default
    parser.add_argument("--dataset", "-ds", help="dataset name",type=str,default="APPS")
    parser.add_argument("--network", "-network", help="network name",type=str,default='CodeRL')
    parser.add_argument("--key", "-k", help="openai key",type=str)
    parser.add_argument("--order", "-o", help="fuzzing order",type=int,default=5)
    parser.add_argument("--gpu_id", "-gid", help="gpu id",type=int,default=0)
    parser.add_argument("--date", "-dt", help="date",type=str,default=str(date.today()))
    parser.add_argument("--start", "-s", help="id",type=int,default=0)
    parser.add_argument("--end", "-e", help="id",type=int,default=10)
    parser.add_argument("--log_path", "-lp", help="log path",type=str,default="./logs/test.log")
    parser.add_argument("--op_type", "-ot", help="mutation operator types",type=str,choices=['exst','absnt',"mix"], default="mix")
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

    save_path = os.path.join(root_path, f"results/20230503-fuzzing_high-order-k{args.order}_{args.op_type}/{exp_idntfr}/our")
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
    valid_input_dict = common_utils.read_pkl(f="valid_input_dict-10_models.pkl")
    valid_input_ids = valid_input_dict[exp_idntfr]
    for prgrss_id, prompt_id, in enumerate(range(args.start, args.end)):
        if prompt_id not in valid_input_ids:
            logger.info(f"prompt {prompt_id} not in {len(valid_input_ids)} valid inputs")
            # skiped_ids.append(prompt_id)
            continue
        try:
            logger.info(f"Generating test case for prompt {prompt_id}")
            
            data_item = DataItem(prompt_id=prompt_id,dataset_name=dataset_name, dataset=dataset,humaneval_tests=humaneval_tests)
            clean_prompt = data_item.clean_prompt
            clean_output = model_utils.code_generation(clean_prompt, tokenizer, model, model_name, device, key=args.key)

            # get ast nodes of the output
            clean_content = clean_prompt + clean_output  if dataset_name == "humaneval" else clean_output
            # print(clean_output)
            # continue
            # sys.exit(0)
            logger.info("\n"+ clean_content)
            # print(data_item.groundtruth)
            syntax_correct = ast_utils.syntax_correct(clean_content)
            if not syntax_correct:
                skiped_ids.append(prompt_id)
                logger.error(f"Origin output {prompt_id} has syntax error!")
                continue
 
            # break
            unique_orig_facts, orig_fact_dict = ast_utils.fact_nodes_extraction_wo_prompt(clean_content,args.op_type)
            # define oracle checker
            # bleu_oracle = oracle_utils.BLEUOracle()
            code_bleu_oracle = oracle_utils.CodeBLEUOracle()
            pass_oracle = oracle_utils.Pass_Oracle(ds_name=dataset_name)

            # get the pass rate of the original prompt
            orig_pass_rate = pass_oracle.calc_orig_pass_rate(prompt=clean_prompt, output=clean_output, data_item=data_item)
            prompt_ins = PromptInstance(prompt_id=prompt_id,orig_prompt=clean_prompt,orig_output=clean_output,orig_fact_dict=orig_fact_dict,pass_rate=orig_pass_rate,syntax=syntax_correct)

            pass_oracle.update_last_pass_rate(orig_pass_rate)
            logger.info(f"Original pass rate {orig_pass_rate}!")

            # conduct 1 order mutation for all mutators
            logger.info(f"Start one order mutation. {len(orig_fact_dict.keys())} mutators, {len(unique_orig_facts)} unique facts")

            prompt_start = datetime.now().timestamp()
            total_attack_facts = []
            for mutator_id,(mutator_name, code_facts) in enumerate(orig_fact_dict.items()):
                if len(code_facts) == 0:
                    continue  
                # we first get one code fact from this type
                np.random.shuffle(code_facts)
                first_fact = np.random.choice(code_facts)
                attack_facts = [first_fact]
                total_attack_facts.append(attack_facts)
                logger.info(f"One order mutation {mutator_name}. {pynode.parse_facts(attack_facts)}")

            logger.info(f"Get {len(total_attack_facts)} one order mutations. Start high order mutation. K = {args.order}")
            
            for _ in range(args.order - 1):
                for attack_facts in total_attack_facts:
                    is_success = False
                    selected_fact_strs = [pynode.parse_facts([attack_fact]) for attack_fact in attack_facts]
                    for mutator_id,(mutator_name, code_facts) in enumerate(orig_fact_dict.items()):
                        if len(code_facts) == 0:
                            continue
                        # we first get one code fact from this type, the selected fact should not be selected previously
                        np.random.shuffle(code_facts)
                        for fact in code_facts:
                            fact_attack_str = pynode.parse_facts([fact])
                            if fact_attack_str in selected_fact_strs:
                                continue
                            # TODO: check the input length
                            elif len(fact_attack_str) > len(clean_prompt):
                                continue
                            else:
                                attack_facts.append(fact)
                                # total_attack_facts.append(attack_facts)
                                is_success = True
                                break
                        if is_success:
                            break
            # get the best high order mutation

            for attack_facts in total_attack_facts:
            # append the attack_facts to prompt_instance
                attack_str = pynode.parse_facts(attack_facts)
                logger.info(f"Found {len(attack_facts)}-Order Mutation. {attack_str}")

                attack_prompt = attacker.attack(clean_prompt, data_item.entry_point, attack_str)
                attack_output = model_utils.code_generation(attack_prompt, tokenizer, model, model_name, device,key=args.key)

                attacked_content = attack_prompt + attack_output  if dataset_name == "humaneval" else attack_output
                attack_syntax_correct = ast_utils.syntax_correct(attacked_content)
                
                # here we should save the final results
                attack_ins = AttackInstance(prompt=attack_prompt,output=attack_output,last_fact=attack_facts[-1],applied_facts=attack_facts,order=len(attack_facts))
                is_syntax_incons = not attack_syntax_correct
                if is_syntax_incons:
                    attack_ins.success_stat.append("syntax")
                else:
                    # pass rate
                    if "APPS" in exp_idntfr:
                        is_pass_incons, attack_pass_rate = pass_oracle.check_inconsistency(prob_path=problems[prompt_id], generation=attack_output, debug=False)

                    else:
                        prompt_item = dataset[prompt_id]
                        task_id, _, entry_point, tests = prompt_item["task_id"], prompt_item["prompt"], prompt_item["entry_point"], humaneval_tests[prompt_id]

                        is_pass_incons, attack_pass_rate = pass_oracle.check_inconsistency(task_id=task_id, prompt=attack_prompt, output=attack_output, entry_point=entry_point, test=tests)

                    attack_ins.pass_rate = attack_pass_rate
                prompt_ins.attack_instances.append(attack_ins)
            total_results.append(prompt_ins)


            # logger.info("="*10 + "Results" + "="*10)
            # logger.info(f"{len(skiped_ids)} cases was skipped.")
            # logger.info(f"{success_cnt} cases was successfully attacked.")
            # logger.info(f"{len(crashed_ids)} cases crashes in mutation.")
            # logger.info(f"{len(failed_ids)} cases was failed.")

            result_dict = {
                "attack_results":total_results,
                "failed_ids":failed_ids,
                "skiped_ids":skiped_ids,
                "crashed_ids":crashed_ids
            }
            with open(os.path.join(save_path,f"{exp_idntfr}_{args.start}-{prompt_id+1}.pkl"), "wb") as fw:
                pickle.dump(result_dict, fw)


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
            