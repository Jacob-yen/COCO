import sys
import pickle
import os
sys.path.append(os.getcwd())
import os
import json
import datasets
from tqdm import tqdm
from utils.common_utils import walk_path
import utils.model_utils as m_utils
from datetime import datetime 
from tqdm import tqdm

dataset_dict = {"humaneval": "prompt", "apps": "question"}

class DataItem:
    def __init__(self,**kw) -> None:
        self.dataset_name = kw["dataset_name"]
        self.prompt_id = kw["prompt_id"]
        self.task_id = None
        self.clean_prompt = None
        self.entry_point=None
        self.tests=None
        self.prob_path=None
        self.starter_path=None
        self.test_case_path=None
        self.groundtruth = None

        if self.dataset_name == "humaneval":
            prompt_item = kw["dataset"][self.prompt_id]
            self.task_id = prompt_item["task_id"]
            # self.clean_prompt = prompt_item["prompt"]
            with open(f"attacks/humaneval/origin/prompt_{self.prompt_id}.txt","r") as fr:
                self.clean_prompt = fr.read()
            self.entry_point = prompt_item["entry_point"]
            self.tests = kw["humaneval_tests"][self.prompt_id]
            self.groundtruth = prompt_item["canonical_solution"]
        elif self.dataset_name == "APPS":
            self.prob_path = kw["dataset"][self.prompt_id]
            self.test_case_path = os.path.join(self.prob_path, "input_output.json")
            self.prompt_path = os.path.join(self.prob_path, "question.txt")
            self.starter_path = os.path.join(self.prob_path, "starter_code.py")
            self.solution_path = os.path.join(self.prob_path, "solutions.json")
            if not os.path.exists(self.starter_path):
                self.starter_path = None
            if not os.path.exists(self.test_case_path) or not os.path.exists(self.prompt_path):
                raise FileExistsError(f"File not exist for APPS data item. Prompt id: {self.prompt_id}")
            self.clean_prompt, _ = APPS_prompt(self.test_case_path, self.prompt_path, self.starter_path)
            if not os.path.exists(self.solution_path):
                self.groundtruth = None
            else:
                with open(self.solution_path, 'r') as f:
                    self.groundtruth = json.load(f)
            
        else:
            raise NotImplementedError(f"Unsupported dataset {self.dataset_name}")



def yc_huggingface_datasets(d_name):
    if d_name == "humaneval":
        return datasets.load_dataset("openai_humaneval",
                            split='test',
                            cache_dir=".././datasets")
    elif d_name == "APPS":
        # load apps problem paths
        with open('data/APPS-5000.pkl', "rb") as fr:
            data = pickle.load(fr)
        return data
    elif d_name == "concode":
        # dev.json or test.json, but test.json has no groundtruth
        # ["code":"...", "nl":"..."]
        return open("finetune/concode/dev.json").readlines()
    else:
        raise NotImplementedError(f"{d_name} is not supported")


def huggingface_datasets(d_name):
    if d_name == "humaneval":
        return datasets.load_dataset("openai_humaneval",
                            split='test',
                            cache_dir="./datasets")
    elif d_name == "APPS":
        # load apps problem paths
        with open('data/APPS-5000.pkl', "rb") as fr:
            data = pickle.load(fr)
        return data
    elif d_name == "concode":
        # dev.json or test.json, but test.json has no groundtruth
        # ["code":"...", "nl":"..."]
        return open("finetune/concode/dev.json").readlines()
    else:
        raise NotImplementedError(f"{d_name} is not supported")



def APPS_prompt(test_case_path, prompt_path, starter_path=None):
    _input = "\nQUESTION:\n"
    with open(prompt_path, "r") as f:
        data = f.readlines()
        data = "".join(data)
    _input += data
    if starter_path != None:
        with open(starter_path, "r") as f:
            data = f.readlines()
            data = "".join(data)
            data = "\n" + data #+ "\n"
        _input += data

    with open(test_case_path, "r") as f:
        data = json.load(f)
    if not data.get("fn_name"):
        _input += "\nUse Standard Input format"#\n"
    else:
        _input += "\nUse Call-Based format"#\n"
    
    _input += "\nANSWER:\n"
    sample_sol = None
    return _input, sample_sol


def load_dataset(d_name):
    assert d_name in ["humaneval", "apps"], f"Not not support dataset {d_name}"
    d_path = os.path.join("./benchmarks",d_name)
    return walk_path([], d_path)

if __name__ == "__main__":

    # load apps problem paths
    with open(os.path.join('finetune/APPS', "test.json"), "r") as f:
        problems = json.load(f)
    problems = sorted(problems)

    apps_list = []
    for prompt_id in tqdm(range(len(problems))):
        problem = problems[prompt_id]
        prob_path = problem
        prompt_idntfr = f"prompt_{prompt_id}"
        test_case_path = os.path.join(prob_path, "input_output.json")
        prompt_path = os.path.join(prob_path, "question.txt")
        starter_path = os.path.join(prob_path, "starter_code.py")
        if not os.path.exists(starter_path):
            starter_path = None
        if not os.path.exists(test_case_path) or not os.path.exists(prompt_path):
            continue
        
        # Read the question in
        clean_prompt, sample_sol = APPS_prompt(test_case_path, prompt_path, starter_path)
        prompt_item = dict()
        prompt_item["id"] = prompt_id
        prompt_item["prompt"] = clean_prompt
        prompt_item["prob_path"] = prob_path
        apps_list.append(prompt_item)
    print(123)
    with open("data/APPS-5000.pkl","wb") as fw:
        pickle.dump(apps_list,fw)


    # benchmark_dir = "./benchmarks"
    # if not os.path.exists(benchmark_dir):
    #     os.makedirs(benchmark_dir)

    # for dataset_name, key in dataset_dict.items():
    #     d_dir = os.path.join(benchmark_dir,dataset_name)
    #     if not os.path.exists(d_dir):
    #         os.makedirs(d_dir)
    #     dataset = huggingface_datasets(dataset_name)
    #     d_size = len(dataset)
    #     for i in tqdm(range(d_size)):
    #         prompt = dataset[i][key]

    #         with open(os.path.join(d_dir,f"prompt_{i}.txt"), "w") as fw:
    #             fw.write(prompt)
    
    # save humaneval results
    # root_path = os.getcwd()
    # # load model
    # model_name = "Salesforce/codegen-350M-multi"
    # model_idntfr = "codegen-350M-multi"
    # tokenizer, model = m_utils.load_model(model_name)
    # # load dataset
    # dataset_name = "humaneval"



    
    # prompt_paths = load_dataset(dataset_name)
    # dataset = huggingface_datasets(dataset_name)
    # # for each prompt, load clean prompt, generate attack prompt, and save attack prompt
    # attack_status = []
    # s0 = datetime.now()
    # dataset_list = []

    # for prompt_id in tqdm(range(len(dataset))):
    #     data_dict = dict()
    #     # for prompt_id in tqdm(range(3)):
    #     # prompt_name is like `prompt_69`
    #     prompt_item = dataset[prompt_id]
    #     # print(123)
    #     # read the prompt from file and get original prediction
    #     clean_prompt, entry_point = prompt_item["prompt"], prompt_item["entry_point"]
    #     # output is only the prediction of model and solution is equal to `prompt + output`
    #     clean_outputs, clean_solutions = generate_code_v2(clean_prompt, tokenizer, model)
    #     # TODO: currently we don't check whether the solution is a valid program or not. 
    #     data_dict["prompt_id"] = prompt_id
    #     data_dict["prompt"] = clean_prompt
    #     data_dict["solution"] = clean_solutions[0]
    #     data_dict["entry_point"] = entry_point
    #     dataset_list.append(data_dict)

    # with open("human_eval_solutions.pkl", "wb") as wb:
    #     pickle.dump(dataset_list, wb)

    # with open("human_eval_solutions.pkl","rb") as rb:
    #     data = pickle.load(rb)
    #     print(123)




