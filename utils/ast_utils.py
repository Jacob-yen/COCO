import os
import sys
sys.path.append(os.getcwd())
import numpy as np
from typing import List
from collections import defaultdict
import ast
from utils.python_visitor import CodeVisitor
from utils import feature_utils,python_visitor,python_node as pynode
import traceback


def extract_lines_from_str(line):
    line_num = line.split("\n")
    return line_num

def fact_nodes_extraction(last_prompt_line,solution,node_type):
    """
    Absence fact and extension fact
    """
    assert node_type in ["exst","absnt","mix"], f"Unsupported node type : {node_type}."
    code = ast.parse(solution,type_comments=True)
    visitor = CodeVisitor()
    visitor.visit(code)
    exst_nodes = visitor.nodes
    exst_nodes = feature_utils.ast_node_analysis(exst_nodes,last_prompt_line)
    exst_nodes.append(pynode.Exst_PLNode(lang="python"))

    if node_type == "exst":
        unique_exst_nodes,nodes_dict = python_visitor.node_division(exst_nodes,pynode.TOTAL_EXST_NODES)
        return unique_exst_nodes, nodes_dict
    elif node_type == "absnt":
        absnt_nodes = python_visitor.generate_absnt_node(exst_nodes)
        unique_absnt_nodes,nodes_dict = python_visitor.node_division(absnt_nodes,pynode.TOTAL_ABSNT_NODES)
        return unique_absnt_nodes, nodes_dict
    else:
        absnt_nodes = python_visitor.generate_absnt_node(exst_nodes)
        absnt_nodes.extend(exst_nodes)
        unique_nodes,nodes_dict = python_visitor.node_division(absnt_nodes,pynode.TOTAL_NODES)
        return unique_nodes, nodes_dict

def fact_nodes_extraction_wo_prompt(solution,node_type):
    assert node_type in ["exst","absnt","mix"], f"Unsupported node type : {node_type}."
    code = ast.parse(solution,type_comments=True)
    visitor = CodeVisitor()
    visitor.visit(code)
    exst_nodes = visitor.nodes
    # exst_nodes.append(pynode.Exst_PLNode(lang="python"))

    if node_type == "exst":
        unique_exst_nodes, nodes_dict = python_visitor.node_division(exst_nodes,pynode.TOTAL_EXST_NODES)
        return unique_exst_nodes, nodes_dict
    elif node_type == "absnt":
        absnt_nodes = python_visitor.generate_absnt_node(exst_nodes)
        unique_absnt_nodes,nodes_dict = python_visitor.node_division(absnt_nodes,pynode.TOTAL_ABSNT_NODES)
        return unique_absnt_nodes, nodes_dict
    else:
        absnt_nodes = python_visitor.generate_absnt_node(exst_nodes)
        absnt_nodes.extend(exst_nodes)
        unique_nodes,nodes_dict = python_visitor.node_division(absnt_nodes,pynode.TOTAL_NODES)
        return unique_nodes, nodes_dict


def syntax_correct(s):
    try :
        ast.dump(ast.parse(s,type_comments=True))
    except SyntaxError:
        return False
    except Exception:
        traceback.print_exc()
        print("Found unexpected Exception!")
        return False
    return True



if __name__ == "__main__":
    pass
