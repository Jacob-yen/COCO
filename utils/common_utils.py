import os
from datetime import datetime
import time
import logging
# import pickle
import dill as pickle

def strcmp(s1, s2):
    l1, l2 = len(s1), len(s2)
    max_len = max(l1, l2)
    for i in range(max_len):
        if i >= min(l1,l2):
            return i
        else:
            if s1[i] != s2[i]:
                return i
    else:
        return -1

def read_pkl(f):
    with open(f,"rb") as fr:
        return pickle.load(fr)


def write_pkl(f,obj):
    import dill as pickle
    with open(f,"wb") as fw:
        return pickle.dump(obj,fw)
    
def read_file(f):
    with open(f) as fr:
        return fr.read()

def write_file(f,s):
    with open(f,"w") as fw:
        return fw.write(s)

def validity_checker(prog):
    return NotImplementedError

def walk_path(files,root_path):
    if os.path.isdir(root_path):
        sub_paths = [os.path.join(root_path,f) for f in os.listdir(root_path)]
        for sub_path in sub_paths:
            files = walk_path(files,sub_path)
    else:
        files.append(root_path)
    return files

def parse_leet_code(file_path):
    files = walk_path([], file_path)
    files = [f for f in files if f.endswith(".py")]
    save_path = ""
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for idx,f in enumerate(files):
        try:
            with open(f,"r") as fr:
                content = fr.read()
            with open(os.path.join(save_path,f"{idx}.py"), "w") as fw:
                fw.write(content)
            
        except UnicodeDecodeError:
            print(f)
            continue

def get_HH_mm_ss(td):
    days, seconds = td.days, td.seconds
    hours = days * 24 + seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return hours, minutes, secs

def get_est_time(s_sec,e_sec,prgrs,total):
    est_sec = (e_sec - s_sec)/prgrs * (total - prgrs)
    hours = est_sec // 3600
    minutes = (est_sec % 3600) // 60
    secs = est_sec % 60
    return int(hours), int(minutes), int(secs)

def time_check(start_time,time_limitation):
    hours, minutes, _ = get_HH_mm_ss(datetime.now() - start_time)
    total_minutes = hours * 60 + minutes
    if total_minutes < time_limitation:
        return True
    else:
        return False

def smart_sleep(start_time):
    td = datetime.now() - start_time
    seconds = td.total_seconds()
    mintues = seconds // 60
    if mintues != 0 and mintues % 5 == 0:
        logging.getLogger("mylogger").info("Smart sleep for 30 seconds.")
        time.sleep(30)
        

class AttackInstance:
    def __init__(self,prompt,output,pass_rate=None,syntax=None,last_fact=None,applied_facts=None, order=None) -> None:
        self.prompt = prompt
        self.pass_rate = pass_rate
        self.output = output
        self.syntax = syntax
        self.last_fact = last_fact
        self.order = order
        self.applied_facts = applied_facts
        self.success_stat = []

class PromptInstance:
    def __init__(self,prompt_id,orig_prompt,orig_output,orig_fact_dict,pass_rate,syntax) -> None:
        self.prompt_id = prompt_id
        self.orig_prompt = orig_prompt
        self.orig_output = orig_output
        self.pass_rate = pass_rate
        self.orig_fact_dict = orig_fact_dict
        self.syntax = syntax
        self.attack_instances = []

    def updtae_attack_instance(self,ins):
        self.attack_instances.append(ins)

    def update_fact_nodes(self,nodes):
        self.fact_nodes = nodes

if __name__ == "__main__":
    pass