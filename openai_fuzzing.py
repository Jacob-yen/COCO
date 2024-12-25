import os
import sys

# to disable the TOKENIZERS_PARALLELISM warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
root_path = os.getcwd()
print(root_path)
sys.path.append(root_path)
from utils.common_utils import PromptInstance, AttackInstance
# from utils.human_base_dataset import reindent_code
import logging
import utils.python_node as pynode
import numpy as np
import random
import pickle
from utils.model_utils import online_api_code_generation

np.random.seed(7)  # numpy
random.seed(7)  # random and transforms

import traceback
import argparse
from utils import ast_utils, attack_utils, common_utils
# from utils.datasets_utils import huggingface_datasets, DataItem
from datetime import datetime, date
from collections import defaultdict
import time

if __name__ == "__main__":
    s0 = datetime.now()
    parser = argparse.ArgumentParser()
    # NOTE: uses 1000 as test_size as default
    parser.add_argument("--dataset", "-ds", help="dataset name", type=str, default="humaneval")
    parser.add_argument("--network", "-network", help="network name", type=str, default="codegen-350M-multi")
    parser.add_argument("--key", "-k", help="openai key", type=str)
    parser.add_argument("--gpu_id", "-gid", help="gpu id", type=int, default=0)
    parser.add_argument("--date", "-dt", help="date", type=str, default=str(date.today()))
    parser.add_argument("--start", "-s", help="id", type=int, default=0)
    parser.add_argument("--end", "-e", help="id", type=int, default=164)
    parser.add_argument("--log_path", "-lp", help="log path", type=str, default="./logs/test.log")
    parser.add_argument("--op_type", "-ot", help="mutation operator types", type=str, choices=['exst', 'absnt', "mix"],
                        default="mix")
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

    # model_name = args.network
    model_name = 'codex'
    dataset_name = 'APPS'
    api_key = None
    # args.key
    exp_idntfr = f"{dataset_name}_{model_name}"

    save_path = os.path.join(root_path,
                             f"results/test_{model_name}_{dataset_name}_{args.date}_{args.op_type}/{exp_idntfr}/our")

    args.log_path = f"./logs/{model_name}_{dataset_name}_{args.date}_{args.op_type}/our"

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
    entry_point_dict = common_utils.read_pkl("./humaneval_entry_point.pkl")

    # dataset = humaneval_dataset if dataset_name == "humaneval" else apps_problems
    # for prgrss_id, prompt_id, in enumerate(range(args.start, args.end)):
    prompt_id = 0
    while prompt_id < 500:
        try:
            # time.sleep(10)
            # logger.info(f"Generating test case for prompt {prompt_id}")

            # data_item = DataItem(prompt_id=prompt_id, dataset_name=dataset_name, dataset=dataset,
            #                      humaneval_tests=humaneval_tests)
            # clean_prompt = data_item.clean_prompt

            try:
                clean_prompt = common_utils.read_file(
                    f"C:\\Users\\DELL\\Desktop\\inputs\\{dataset_name}\\origin\\prompt_{prompt_id}.txt")
            except:
                with open(f"C:\\Users\\DELL\\Desktop\\inputs\\{dataset_name}\\origin\\prompt_{prompt_id}.txt", 'r',
                          encoding='utf-8') as f:
                    clean_prompt = f.read()
            # init_facts, init_fact_dict = ast_utils.fact_nodes_extraction_wo_prompt(clean_prompt, 'exst')
            # func_name = init_fact_dict['Exst_FunctionDefinition'][0].func_name

            clean_output = online_api_code_generation(clean_prompt, model_name, api_key=api_key,
                                                      dataset_name=dataset_name)

            clean_content = clean_prompt + clean_output if dataset_name == "humaneval" else clean_output

            syntax_correct = ast_utils.syntax_correct(clean_content)
            if not syntax_correct:
                skiped_ids.append(prompt_id)
                logger.error(f"Origin output {prompt_id} has syntax error!")
                prompt_id += 1
                continue

            # break
            unique_orig_facts, orig_fact_dict = ast_utils.fact_nodes_extraction_wo_prompt(clean_content, args.op_type)

            # get the pass rate of the original prompt
            orig_pass_rate = None
            prompt_ins = PromptInstance(prompt_id=prompt_id, orig_prompt=clean_prompt + '\nAnswer:\n',
                                        orig_output=clean_output,
                                        orig_fact_dict=orig_fact_dict, pass_rate=orig_pass_rate, syntax=syntax_correct)

            # conduct 1 order mutation for all mutators
            # for each mutator, use at most three facts
            logger.info(
                f"Start one order mutation. {len(orig_fact_dict.keys())} mutators, {len(unique_orig_facts)} unique facts")

            prompt_start = datetime.now().timestamp()
            min_bleu = 999
            final_attack_prompt = None
            final_attacked_output = None
            last_fact = None
            for mutator_id, (mutator_name, code_facts) in enumerate(orig_fact_dict.items()):
                if len(code_facts) == 0:
                    continue
                elif len(code_facts) >= 3:
                    selected_facts = np.random.choice(code_facts, size=3)
                else:
                    selected_facts = code_facts
                for fact_idx, _selected_fact in enumerate(selected_facts):
                    attack_str = pynode.parse_facts([_selected_fact])
                    logger.info(f"Found mutator {mutator_name}. {attack_str}")

                    if dataset_name == 'humaneval':
                        entry_point = entry_point_dict[prompt_id]
                    else:
                        entry_point = None
                    attack_prompt = attacker.attack(clean_prompt, entry_point, attack_str)

                    attacked_output = online_api_code_generation(attack_prompt, model_name, api_key=api_key,
                                                                 dataset_name=dataset_name)

                    attacked_content = attack_prompt + attacked_output if dataset_name == "humaneval" else attacked_output
                    attack_syntax_correct = ast_utils.syntax_correct(attacked_content)

                    attack_ins = AttackInstance(prompt=attack_prompt + '\nAnswer:\n', output=attacked_output,
                                                last_fact=_selected_fact,
                                                applied_facts=[_selected_fact], order=1)
                    prompt_ins.attack_instances.append(attack_ins)
            total_results.append(prompt_ins)

            result_dict = {
                "attack_results": total_results,
                "failed_ids": failed_ids,
                "skiped_ids": skiped_ids,
                "crashed_ids": crashed_ids
            }
            with open(os.path.join(save_path, f"{exp_idntfr}_{args.start}-{prompt_id}.pkl"), "wb") as fw:
                pickle.dump(result_dict, fw)
            prompt_id += 1
        except Exception as e:
            crashed_ids.append(prompt_id)
            logger.error(f"Error found in mutating prompt at {prompt_id}.")
            logger.error(e)
            logger.error("\n" + traceback.format_exc())
            prompt_id += 1
            continue

    logger.info("=" * 10 + "Results" + "=" * 10)
    logger.info(f"{len(skiped_ids)} cases was skipped.")
    logger.info(f"{success_cnt} cases was successfully attacked.")
    logger.info(f"{len(crashed_ids)} cases crashes in mutation.")
    logger.info(f"{len(failed_ids)} cases was failed.")

    result_dict = {
        "attack_results": total_results,
        "failed_ids": failed_ids,
        "skiped_ids": skiped_ids,
        "crashed_ids": crashed_ids
    }
    with open(os.path.join(save_path, f"{exp_idntfr}_{args.start}-{args.end}.pkl"), "wb") as fw:
        pickle.dump(result_dict, fw)

    logger.info(f"Time cost: {datetime.now() - s0}")
