import os
import sys
sys.path.append(os.getcwd())
from numpy import deprecate
from utils.common_utils import read_file, write_file
import ast

class HumanEvalAttack:

    @staticmethod
    def get_docstring_node(node, entry):
        for child_node in ast.walk(node):
            if isinstance(child_node, ast.FunctionDef) and child_node.name == entry:
                return child_node

    @staticmethod
    @deprecate
    def set_docstring(node,t):
        node_body_value = node.body[0].value
        if isinstance(node_body_value, ast.Str):
            node.body[0].value.s = t
        elif isinstance(node_body_value, ast.Constant) and isinstance(node_body_value.value, str):
            node.body[0].value.value = t
        else:
            raise ValueError(f"Can't handle {type(node_body_value)}")


    @staticmethod
    def insert_perturbation(original_prompt, docstring, pertur):
        """
        Please note that in the Python 3.9, the ast module provides a function `unparse` which can parse 
        the ast object to code strings. However, the produced code string will not necessarily be equal to the
        original code that generated the ast.AST object. Thus, we decide to directly replace the prompt with 
        perturbed examples. 
        """
        return original_prompt.replace(docstring, pertur)


    @staticmethod
    def attack(prompt,entry_point, attack_str):
        root_code = ast.parse(prompt,type_comments=True)
        target_node = HumanEvalAttack.get_docstring_node(root_code,entry_point)
        doc_string = ast.get_docstring(target_node, clean=False)
        if doc_string is None:
            return prompt
        else:
            perturbed_string = " " + attack_str + " " + doc_string
            attack_prompt = HumanEvalAttack.insert_perturbation(prompt, doc_string, perturbed_string)
            return attack_prompt


class APPSAttack:


    @staticmethod
    def attack(prompt,entry_point,attack_str):
        return prompt.replace("QUESTION:\n", f"QUESTION:\n{attack_str}\n")

attacker_dict = {
    "humaneval":HumanEvalAttack,
    "APPS":APPSAttack
}

if __name__ == "__main__":
    # pass
    content = read_file("benchmarks/humaneval/prompt_0.txt")
    
    entry="has_close_elements"
    print(content)
    root_code = ast.parse(content, type_comments=True)
    # parse prompt using ast
    target_node = HumanEvalAttack.get_docstring_node(root_code,entry)
    doc_string = ast.get_docstring(target_node,clean=False)
    print(doc_string)
    noised_doc_string = "XXXXXX" + doc_string
    attack_prompt = HumanEvalAttack.insert_perturbation(content, doc_string, noised_doc_string)
    # write_file("test_attack_32.txt", attack_prompt)
    print(content)
    print("\n========================\n")
    print(attack_prompt)


