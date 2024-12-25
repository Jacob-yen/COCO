import sys
import os
from tkinter.messagebox import NO
sys.path.append(os.getcwd())
import pickle
import os
import ast
import numpy as np
import random
from utils.logging_utils import MyLogger
from utils.common_utils import read_file
from utils.python_visitor import CodeVisitor
import utils.python_node as pynode
from utils.attack_utils import HumanEvalAttack
from utils.mutation_utils import Mutant,Mutator,Selection
import utils.model_utils as m_utils
from datetime import datetime 
from tqdm import tqdm
import traceback
from collections import defaultdict
from typing import List
import logging

# This is the execution path.
root_path = os.getcwd()

# def select_fact_by_mutator(mutator:Mutator,mutant:Mutant,orig_facts:dict,previous_attack_str:set):
#     # check whether we can select a fact node that had not been appied already.
#     # check whether the attacked str had been generated already
#     # return the fact; else return none.
#     facts = orig_facts[mutator.name]
#     idxs = np.arange(len(facts))
#     np.random.shuffle(idxs)
#     fact_strs = {str(fact) for fact in mutant.applied_facts}
#     for idx in idxs:
#         tmp_facts = [facts[idx], *mutant.applied_facts]
#         if str(facts[idx]) not in fact_strs and pynode.parse_facts(tmp_facts) not in previous_attack_str:
#             return facts[idx]
#     else:
#         return None

def select_mutator_and_fact(selection:Selection,mutant:Mutant,orig_facts:dict,previous_attack_str:set,last_used_mutator):
    aborted_mutators = set()
    logger = logging.getLogger('mylogger')
    while True:
        if len(aborted_mutators) == selection.mutator_count:
            logger.warn("All mutators have been aborted. Skip this mutant.")
            break
        mutator = selection.choose_mutator(last_used_mutator=last_used_mutator)
        mutator_name = mutator.name
        # check whether we can select a fact node that had not been appied already.
        if len(mutant.applied_facts_dict[mutator_name]) == len(orig_facts[mutator_name]):
            # logger.warn(f"No facts of mutator {mutator_name} can be utilized. Skip it.")
            aborted_mutators.add(mutator_name)
            continue

        # check whether the attacked str had been generated already
        # return the fact; else return none.
        facts = orig_facts[mutator_name]
        idxs = np.arange(len(facts))
        np.random.shuffle(idxs)
        fact_strs = {str(fact) for fact in mutant.applied_facts}
        for idx in idxs:
            tmp_facts = [facts[idx], *mutant.applied_facts]
            attack_str = pynode.parse_facts(tmp_facts)
            if str(facts[idx]) not in fact_strs and attack_str not in previous_attack_str:
                return mutator, facts[idx], attack_str
        else:
            aborted_mutators.add(mutator_name)

    return None, None, None

def ast_node_analysis(nodes:List[pynode.SyntaxNode],prompt_end_line):
    treenodes = []
    for node in nodes:
        if node.lineno < prompt_end_line:
            continue
        treenodes.append(node)
    return treenodes


def ast_node_analysis_wo_prompt(nodes:List[pynode.SyntaxNode]):
    treenodes = []
    for node in nodes:
        treenodes.append(node)
    return treenodes


if __name__ == "__main__":
    pass

