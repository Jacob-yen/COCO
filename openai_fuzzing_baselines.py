import os
import sys

# to disable the TOKENIZERS_PARALLELISM warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
root_path = os.getcwd()
print(root_path)
sys.path.append(root_path)
from utils.common_utils import PromptInstance, AttackInstance
import logging
from utils.model_utils import online_api_code_generation
import numpy as np
import random
import pickle

np.random.seed(7)  # numpy
random.seed(7)  # random and transforms

import traceback
import argparse
from utils import ast_utils, attack_utils, model_utils, common_utils
from datetime import datetime, date
from collections import defaultdict


def copy_set(source):
    target = set()
    for s in source:
        target.add(s)
    return target


def read_attack_prompt(afile):
    try:
        clean_prompt = common_utils.read_file(afile)
    except:
        with open(afile, 'r', encoding='utf-8') as f:
            clean_prompt = f.read()
    # if dataset_name == "humaneval":
    #     with open(afile, "r") as fr:
    #         _prompt = fr.read()
    # else:
    #     _prompt, _ = APPS_prompt(dt.test_case_path, afile, dt.starter_path)
    return clean_prompt


if __name__ == "__main__":
    s0 = datetime.now()
    parser = argparse.ArgumentParser()
    # NOTE: uses 1000 as test_size as default
    parser.add_argument("--dataset", "-ds", help="dataset name", type=str, default="humaneval")
    parser.add_argument("--network", "-network", help="network name", type=str, default="codegen-350M-multi-retrain")
    parser.add_argument("--key", "-k", help="openai key", type=str)
    parser.add_argument("--gpu_id", "-gid", help="gpu id", type=int, default=0)
    parser.add_argument("--date", "-dt", help="date", type=str, default=str(date.today()))
    parser.add_argument("--start", "-s", help="id", type=int, default=0)
    parser.add_argument("--end", "-e", help="id", type=int, default=164)
    parser.add_argument("--log_path", "-lp", help="log path", type=str, default="./logs/test_chatgpt_rewrite.log")
    parser.add_argument("--op_type", "-ot", help="mutation operator types", type=str, choices=['exst', 'absnt', "mix"],
                        default="mix")
    parser.add_argument("--baseline", "-b", help="id", choices=["tp", "rewrite", "cat"], type=str, default="rewrite")
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
    api_key = args.key
    args.baseline = 'rewrite'
    exp_idntfr = f"{dataset_name}_{model_name}"

    save_path = os.path.join(root_path,
                             f"results/test_baseline_{model_name}_{dataset_name}_{args.baseline}_{args.date}_{args.op_type}/{exp_idntfr}/our")

    args.log_path = f"./logs/{model_name}_{dataset_name}_{args.date}_{args.op_type}/{args.baseline}"

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
            try:
                clean_prompt = common_utils.read_file(
                    f"C:\\Users\\DELL\\Desktop\\inputs\\{dataset_name}\\origin\\prompt_{prompt_id}.txt")
            except:
                with open(f"C:\\Users\\DELL\\Desktop\\inputs\\{dataset_name}\\origin\\prompt_{prompt_id}.txt", 'r',
                          encoding='utf-8') as f:
                    clean_prompt = f.read()
            # init_facts, init_fact_dict = ast_utils.fact_nodes_extraction_wo_prompt(clean_prompt, 'exst')
            # func_name = init_fact_dict['Exst_FunctionDefinition'][0].func_name

            if dataset_name == 'humaneval':
                init_facts, init_fact_dict = ast_utils.fact_nodes_extraction_wo_prompt(clean_prompt, 'exst')
                func_name = init_fact_dict['Exst_FunctionDefinition'][0].func_name
                clean_output = online_api_code_generation(clean_prompt, model_name, func_name=func_name,
                                                          api_key=api_key,
                                                          dataset_name=dataset_name)
            else:
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
            prompt_ins = PromptInstance(prompt_id=prompt_id, orig_prompt=clean_prompt + "\nAnswer:\n", orig_output=clean_output,
                                        orig_fact_dict=orig_fact_dict, pass_rate=orig_pass_rate, syntax=syntax_correct)

            prompt_start = datetime.now().timestamp()

            logger.info(f"Conduct baseline mutation")
            if args.baseline in ["tp", "rewrite"]:
                if args.baseline == "tp":
                    attacked_file_path = f"./attacks/{dataset_name}/tp/prompt_{prompt_id}.txt"
                elif args.baseline == "rewrite":
                    attacked_file_path = f"./attacks/{dataset_name}/rewrite/prompt_{prompt_id}.txt"

                if not os.path.exists(attacked_file_path):
                    skiped_ids.append(prompt_id)
                    logger.error(f"The baseline can't generate the attacked prompt for {prompt_id}")
                    prompt_id += 1
                    continue

                attack_prompt = read_attack_prompt(attacked_file_path)
                # 改为 codex 和 chatgpt 的输出
                if dataset_name == 'humaneval':
                    attacked_output = online_api_code_generation(attack_prompt, model_name, func_name=func_name,
                                                                 api_key=api_key,
                                                                 dataset_name=dataset_name)
                else:
                    attacked_output = online_api_code_generation(attack_prompt, model_name, api_key=api_key,
                                                                 dataset_name=dataset_name)
                attacked_content = attack_prompt + attacked_output if dataset_name == "humaneval" else attacked_output

                attack_ins = AttackInstance(prompt=attack_prompt + "\nAnswer:\n", output=attacked_output, order=1)
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
            else:
                raise ValueError("Currenly not support baseline CAT.")

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
