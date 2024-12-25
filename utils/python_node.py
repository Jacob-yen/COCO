import os
import sys
from typing import List
import inspect
from collections import defaultdict


class SyntaxNode:
    def __init__(self,lineno, end_lineno) -> None:
        
        self.lineno = lineno
        self.end_lineno = end_lineno

class FactNode(SyntaxNode):
    def __init__(self,lineno, end_lineno) -> None:
        super().__init__(lineno, end_lineno)

    @classmethod
    def template(cls):
        pass
    
    @property
    def common_desc(self):
        pass

    @property
    def prefix_desc(self):
        return " ".join([self.template()["prefix"], self.template()["adv_ins"]])

    def __str__(self) -> str:
        return f"{self.prefix_desc} {self.common_desc}."

class ExistentNode(FactNode):
    def __init__(self,lineno, end_lineno) -> None:
        super().__init__(lineno, end_lineno)

class AbsentNode(FactNode):
    def __init__(self,lineno, end_lineno) -> None:
        super().__init__(lineno, end_lineno)


###########################################
############ DEPENDENCY LEVEL #############
###########################################

class DependencyNode(SyntaxNode):
    def __init__(self,lineno, end_lineno) -> None:
        super().__init__(lineno, end_lineno)

class Exst_PLNode(DependencyNode,ExistentNode):
    def __init__(self,lang,lineno=None, end_lineno=None) -> None:
        super().__init__(lineno, end_lineno)
        self.pl = lang
        self.rank = 0

    @classmethod
    def template(cls):
        return {
            "adv_type":"modal_verb",
            "adv_ins":"is",
            "prefix":"The code",
        }

    @property
    def common_desc(self):
        return f"implemented in {self.pl}"

class Absnt_PLNode(DependencyNode,AbsentNode):
    def __init__(self,lineno=None, end_lineno=None) -> None:
        super().__init__(lineno, end_lineno)
        self.pl = "C++"
        self.rank = 0

    @classmethod
    def template(cls):
        return {
            "adv_type":"modal_verb",
            "adv_ins":"is not",
            "prefix":"The code",
        }

    @property
    def common_desc(self):
        return f"implemented in {self.pl}"

class Exst_PackageNode(DependencyNode,ExistentNode):
    def __init__(self,package,lineno=None, end_lineno=None) -> None:
        super().__init__(lineno, end_lineno)
        self.package = package
        self.rank = 1

    @classmethod
    def template(cls):
        return {
            "adv_type":"modal_verb",
            "adv_ins":"is",
            "prefix":"The code",
        }

    @property
    def common_desc(self):
        return f"dependent on package '{self.package}'"

class Absnt_PackageNode(DependencyNode,AbsentNode):
    def __init__(self,lineno=None, end_lineno=None) -> None:
        super().__init__(lineno, end_lineno)
        self.rank = 1

    @classmethod
    def template(cls):
        return {
            "adv_type":"verb",
            "adv_ins":"should not have any",
            "prefix":"The code",
        }

    @property
    def common_desc(self):
        return "imported module dependencies"

###########################################
############### CLASS LEVEL ###############
###########################################

class ClassNode(SyntaxNode):
    def __init__(self,lineno=None, end_lineno=None) -> None:
        super().__init__(lineno, end_lineno)

class Exst_ClassDefinition(ClassNode,ExistentNode):
    def __init__(self,cls_name, lineno=None, end_lineno=None) -> None:
        super().__init__(lineno, end_lineno)
        self.cls_name = cls_name
        self.rank = 0

    @classmethod
    def template(cls):
        return {
            "adv_type":"verb",
            "adv_ins":"includes",
            "prefix":"The code",
        }

    @property
    def common_desc(self):
        return f"one class named '{self.cls_name}'"

class Absnt_ClassDefinition(ClassNode,AbsentNode):
    def __init__(self,lineno=None, end_lineno=None) -> None:
        super().__init__(lineno, end_lineno)
        self.rank = 0

    @classmethod
    def template(cls):
        return {
            "adv_type":"verb",
            "adv_ins":"should not have any",
            "prefix":"The code",
        }

    @property
    def common_desc(self):
        return "class definitions"

class Exst_ClassVariable(ClassNode,ExistentNode):
    def __init__(self,cls_name,mem_var,lineno=None, end_lineno=None) -> None:
        super().__init__(lineno, end_lineno)
        self.cls_name = cls_name
        self.mem_var = mem_var
        self.rank = 1

    @classmethod
    def template(cls):
        return {
            "adv_type":"verb",
            "adv_ins":"includes",
            "prefix":"",
        }

    @property
    def common_desc(self):
        return f"one member variable named '{self.mem_var}'"

    @property
    def prefix_desc(self):
        return f"The class '{self.cls_name}' in the code {Exst_ClassVariable.template()['adv_ins']}"

class Absnt_ClassVariable(ClassNode,AbsentNode):
    def __init__(self,cls_name,lineno=None, end_lineno=None) -> None:
        super().__init__(lineno, end_lineno)
        self.cls_name = cls_name
        self.rank = 1

    @classmethod
    def template(cls):
        return {
            "adv_type":"verb",
            "adv_ins":"should not have any",
            "prefix":"",
        }

    @property
    def common_desc(self):
        return f"member variables"

    @property
    def prefix_desc(self):
        return f"The class '{self.cls_name}' in the code {Absnt_ClassVariable.template()['adv_ins']}"

class Exst_ClassFunction(ClassNode,ExistentNode):
    def __init__(self,cls_name,func_name,lineno=None, end_lineno=None) -> None:
        super().__init__(lineno, end_lineno)
        self.cls_name = cls_name
        self.func_name = func_name
        self.rank = 2

    @classmethod
    def template(cls):
        return {
            "adv_type":"verb",
            "adv_ins":"includes",
            "prefix":"",
        }

    @property
    def common_desc(self):
        return f"one member function called '{self.func_name}'"

    @property
    def prefix_desc(self):
        return f"The class '{self.cls_name}' in the code {Exst_ClassFunction.template()['adv_ins']}"

class Absnt_ClassFunction(ClassNode,AbsentNode):
    def __init__(self,cls_name, lineno=None, end_lineno=None) -> None:
        super().__init__(lineno, end_lineno)
        self.cls_name = cls_name
        self.rank = 2

    @classmethod
    def template(cls):
        return {
            "adv_type":"verb",
            "adv_ins":"should not have any",
            "prefix":"",
        }

    @property
    def common_desc(self):
        return f"member functions"

    @property
    def prefix_desc(self):
        return f"The class '{self.cls_name}' in the code {Absnt_ClassFunction.template()['adv_ins']}"

###########################################
############# FUNCTION LEVEL ##############
###########################################

class FunctionNode(SyntaxNode):
    def __init__(self,lineno=None, end_lineno=None) -> None:
        super().__init__(lineno, end_lineno)

class Exst_FunctionDefinition(FunctionNode,ExistentNode):
    def __init__(self,func_name, lineno=None, end_lineno=None) -> None:
        super().__init__(lineno, end_lineno)
        self.func_name = func_name
        self.rank = 3

    @classmethod
    def template(cls):
        return {
            "adv_type":"verb",
            "adv_ins":"includes",
            "prefix":"The code",
        }

    @property
    def common_desc(self):
        return f"one function named '{self.func_name}'"

class Absnt_FunctionDefinition(FunctionNode,AbsentNode):
    def __init__(self,lineno=None, end_lineno=None) -> None:
        super().__init__(lineno, end_lineno)
        self.rank = 3

    @classmethod
    def template(cls):
        return {
            "adv_type":"verb",
            "adv_ins":"should not have any",
            "prefix":"The code",
        }

    @property
    def common_desc(self):
        return "function definitions"

class Exst_FunctionArgument(FunctionNode,ExistentNode):
    def __init__(self,func_name,var_names:List[str],lineno=None, end_lineno=None) -> None:
        super().__init__(lineno, end_lineno)
        self.func_name = func_name
        self.var_names = var_names
        self.rank = 4

    @classmethod
    def template(cls):
        return {
            "adv_type":"verb",
            "adv_ins":"takes in",
            "prefix":"",
        }

    @property
    def var_str(self):
        if self.var_names is None or len(self.var_names) == 0:
            raise ValueError(f"The variables numbers of {self.func_name} is zero or None. It should have Existent Fact like this.")
        else:
            var_names = [f"'{v}'" for v in self.var_names]
            if len(var_names) == 1:
                return var_names[0]
            elif len(var_names) == 2:
                return f"{var_names[0]} and {var_names[1]}"
            else:
                return ", ".join(var_names[:-1]) + " and " + var_names[-1]

    @property
    def common_desc(self):
        if self.var_names is None or len(self.var_names) == 0:
            raise ValueError(f"The variables numbers of {self.func_name} is zero or None. It should have Existent Fact like this.")
        elif len(self.var_names) == 1:
            arg_str = "argument"
        else:
            arg_str = "arguments"
        return f"takes in {self.var_str} as {arg_str}"

    @property
    def prefix_desc(self):
        return f"The function '{self.func_name}' in the code"

class Absnt_FunctionArgument(FunctionNode,AbsentNode):
    def __init__(self,func_name, lineno=None, end_lineno=None) -> None:
        super().__init__(lineno, end_lineno)
        self.func_name = func_name
        self.rank = 4

    @classmethod
    def template(cls):
        return {
            "adv_type":"verb",
            "adv_ins":"should not have any",
            "prefix":"The function",
        }

    @property
    def common_desc(self):
        return f"{Absnt_FunctionArgument.template()['adv_ins']} arguments"

    @property
    def prefix_desc(self):
        return f"The function '{self.func_name}' in the code"

class Exst_FunctionReturn(FunctionNode,ExistentNode):
    def __init__(self,func_name, ret, lineno=None, end_lineno=None) -> None:
        super().__init__(lineno, end_lineno)
        self.func_name = func_name
        self.ret = ret
        self.rank = 5

    @classmethod
    def template(cls):
        return {
            "adv_type":"verb",
            "adv_ins":"returns",
            "prefix":"The function",
        }

    @property
    def common_desc(self):
        return f"returns one {self.ret}"


    @property
    def prefix_desc(self):
        return f"The function '{self.func_name}' in the code"


class Exst_CodeReturn(ExistentNode):
    def __init__(self, ret, lineno=None, end_lineno=None) -> None:
        super().__init__(lineno, end_lineno)
        self.ret = ret
        self.rank = 5

    @classmethod
    def template(cls):
        return {
            "adv_type":"verb",
            "adv_ins":"returns",
            "prefix":"The code",
        }

    @property
    def common_desc(self):
        return f"returns one {self.ret}"


    @property
    def prefix_desc(self):
        return f"The code"



class Absnt_FunctionReturn(FunctionNode,AbsentNode):
    def __init__(self,func_name, lineno=None, end_lineno=None) -> None:
        super().__init__(lineno, end_lineno)
        self.func_name = func_name
        self.rank = 5

    @classmethod
    def template(cls):
        return {
            "adv_type":"verb",
            "adv_ins":"returns",
            "prefix":"",
        }

    @property
    def common_desc(self):
        return f"returns nothing"

    @property
    def prefix_desc(self):
        return f"The function '{self.func_name}' in the code"
    
###########################################
############# STATEMENT LEVEL #############
###########################################

class StatementExpressionNode(SyntaxNode):
    def __init__(self,lineno=None, end_lineno=None) -> None:
        super().__init__(lineno, end_lineno)
        self.rank = 1

class Exst_StatementExpressionNode(StatementExpressionNode, ExistentNode):
    def __init__(self,stmt, lineno=None, end_lineno=None) -> None:
        super().__init__(lineno, end_lineno)
        self.stmt = stmt
        self.rank = 1

    @classmethod
    def template(cls):
        return {
            "adv_type":"verb",
            "adv_ins":"has at least",
            "prefix":"The code",
        }

    @property
    def common_desc(self):
        return f"one {self.stmt}"


class Absnt_StatementExpressionNode(StatementExpressionNode, AbsentNode):
    def __init__(self,stmt, lineno=None, end_lineno=None) -> None:
        super().__init__(lineno, end_lineno)
        self.stmt = stmt
        self.rank = 1


    @classmethod
    def template(cls):
        return {
            "adv_type":"verb",
            "adv_ins":"should not have any",
            "prefix":"The code",
        }

    @property
    def common_desc(self):
        return f"{self.stmt}"

class Exst_Statement(Exst_StatementExpressionNode):
    def __init__(self,stmt, lineno=None, end_lineno=None) -> None:
        super().__init__(lineno, end_lineno)
        self.stmt = stmt

class Absnt_Statement(Absnt_StatementExpressionNode):
    def __init__(self,stmt, lineno=None, end_lineno=None) -> None:
        super().__init__(lineno, end_lineno)
        self.stmt = stmt

###########################################
############# EXPRESSION LEVEL ############
###########################################

class Exst_Expression(Exst_StatementExpressionNode):
    def __init__(self,stmt, lineno=None, end_lineno=None) -> None:
        super().__init__(lineno, end_lineno)
        self.stmt = stmt

class Absnt_Expression(Absnt_StatementExpressionNode):
    def __init__(self,stmt, lineno=None, end_lineno=None) -> None:
        super().__init__(lineno, end_lineno)
        self.stmt = stmt

###########################################
############## VARIABLE LEVEL #############
###########################################

class VariableNode(SyntaxNode):
    def __init__(self,lineno=None, end_lineno=None) -> None:
        super().__init__(lineno, end_lineno)

class Exst_VariableDefinition(VariableNode,ExistentNode):
    def __init__(self,obj,lineno=None, end_lineno=None) -> None:
        super().__init__(lineno, end_lineno)
        self.obj = obj
        self.rank = 2

    @classmethod
    def template(cls):
        return {
            "adv_type":"verb",
            "adv_ins":"has",
            "prefix":"The code",
        }

    @property
    def common_desc(self):
        return f"one variable named '{self.obj}'"


class Absnt_VariableDefinition(VariableNode,ExistentNode):
    def __init__(self,lineno=None, end_lineno=None) -> None:
        super().__init__(lineno, end_lineno)
        self.rank = 2

    @classmethod
    def template(cls):
        return {
            "adv_type":"verb",
            "adv_ins":"should not have any",
            "prefix":"The code",
        }

    @property
    def common_desc(self):
        return "variable definitions"


class Exst_VariableAttribute(VariableNode,ExistentNode):
    def __init__(self,desc,lineno=None, end_lineno=None) -> None:
        super().__init__(lineno, end_lineno)
        self.desc=desc

    @classmethod
    def template(cls):
        return {
            "adv_type":"verb",
            "adv_ins":"contains at least",
            "prefix":"The code",
        }
    @property
    def common_desc(self):
        return f"one {self.desc}"


class Absnt_VariableAttribute(VariableNode,ExistentNode):
    def __init__(self,lineno=None, end_lineno=None) -> None:
        super().__init__(lineno, end_lineno)
        self.rank = 2

    @classmethod
    def template(cls):
        return {
            "adv_type":"verb",
            "adv_ins":"should not have any",
            "prefix":"The code",
        }
    @property
    def common_desc(self):
        return "attribute accesses"
    
def is_child(n,parents):
    for parent in parents:
        if isinstance(n,parent):
            return True
    else:
        return False

def is_instance(n,classes):
    for _clas in classes:
        if isinstance(n,_clas):
            return True
    else:
        return False


def merge_multi_siblings(fact_nodes,siblings,conj="and",sort_list=False):
    """
    All the multi sblings should have the same adv
    """
    sent = ""
    facts = []
    for factnode in fact_nodes:
        if is_child(factnode,siblings):
            facts.append(factnode)
    if len(facts) == 0:
        return sent
    if sort_list:
        facts = sorted(facts,key=lambda x:x.rank)
    if len(facts) == 1:
        sent = f"{facts[0].prefix_desc} {facts[0].common_desc}"
    elif len(facts) == 2:
        sent = f"{facts[0].prefix_desc} {facts[0].common_desc} {conj} {facts[1].common_desc}"
    else:
        common_descs = [fact.common_desc for fact in facts]
        middle = ", ".join(common_descs[:-1])
        sent = f"{facts[0].prefix_desc} {middle} {conj} {common_descs[-1]}"
        
    return sent + "." if sent != "" else ""


def parse_exst_modal_verb(fact_nodes):
    sent = merge_multi_siblings(fact_nodes,[Exst_PLNode,Exst_PackageNode],sort_list=True)
    return sent


def parse_exst_verb_includes(fact_nodes):
    """
    Only parse the class definition and function definition node
    """
    sent = merge_multi_siblings(fact_nodes,[Exst_ClassDefinition,Exst_FunctionDefinition,Exst_VariableDefinition],sort_list=True)
    return sent

def parse_exst_verb_have_at_least(fact_nodes):
    """
    Only parse the class definition and function definition node
    """
    sent = merge_multi_siblings(fact_nodes,[Exst_StatementExpressionNode],sort_list=True)
    return sent

def parse_exst_class_func_info(fact_nodes):
    """
    Only parse the class definition and function definition node
    """
    cls_classes=[Exst_ClassFunction,Exst_ClassVariable]
    func_classes = [Exst_FunctionArgument,Exst_FunctionReturn]
    cls_dict = node_attr_division(fact_nodes,attr_name="cls_name",target_classes=cls_classes)
    cls_sents = []
    for _, cls_nodes in cls_dict.items():
        cls_sents.append(merge_multi_siblings(cls_nodes,cls_classes))
    cls_sent = " ".join(cls_sents)

    func_dict = node_attr_division(fact_nodes,attr_name="func_name",target_classes=func_classes)
    func_sents = []
    for _, func_nodes in func_dict.items():
        func_sents.append(merge_multi_siblings(func_nodes,func_classes,sort_list=True))
    func_sent = " ".join(func_sents)

    sent = cls_sent
    if sent == "" and func_sent != "":
        sent = func_sent
    elif sent != "" and func_sent != "":
        func_sent = func_sent.replace("The function ","Also, the function ")
        sent = sent + " " + func_sent
    return sent

# def parse_exst_verb_utilize(fact_nodes):
#     """
#     Only parse the class definition and function definition node
#     """
#     sent = merge_multi_siblings(fact_nodes,[Exst_VariableAttribute])
#     return sent


def exst_parser(fact_nodes):
    sent1 = parse_exst_modal_verb(fact_nodes)
    sent2 = parse_exst_verb_have_at_least(fact_nodes)
    # sent3 = parse_exst_verb_utilize(fact_nodes)
    sent3 = ""
    sent4 = parse_exst_verb_includes(fact_nodes)
    if sent3 == "" and sent4 != "":
        sent3_4 = sent4
    elif sent3 != "" and sent4 == "":
        sent3_4 = sent3
    elif sent3 != "" and sent4 != "":

        sent3_4 = sent3[:-1] + ", and " + sent4.replace("The code ", "")
    else:
        sent3_4 = ""
    sent5 = parse_exst_class_func_info(fact_nodes)
    sent = sent1 

    if sent == "" and sent2 != "":
        sent = sent2
    elif sent != "" and  sent2!="":
        sent = sent[:-1] + sent2.replace("The code ", ", which ")

    if sent == "" and sent3_4 != "":
        sent = sent3_4
    elif sent != "" and  sent3_4!="":
        sent = sent + sent3_4.replace("The code ", " It ")
        
    if sent == "" and sent5 !="":
        sent = sent5
    elif sent != "" and sent5 != "":
        sent = sent + " " + sent5

    return sent

def parse_absnt_modal_verb(fact_nodes):
    sent = merge_multi_siblings(fact_nodes,[Absnt_PLNode],conj="and",sort_list=True)
    return sent


def parse_absnt_verb_does_not_have(fact_nodes):
    """
    Only parse the class definition and function definition node
    """

    sent = merge_multi_siblings(fact_nodes,[Absnt_PackageNode,Absnt_ClassDefinition,Absnt_FunctionDefinition,
                                            Absnt_StatementExpressionNode,Absnt_VariableDefinition],conj="or",sort_list=True)
    return sent

def node_attr_division(fact_nodes,attr_name,target_classes):
    select_nodes = [node for node in fact_nodes if is_child(node,target_classes)]
    cls_dict = defaultdict(list)
    for node in select_nodes:
        cls_dict[getattr(node,attr_name)].append(node)
    return cls_dict

def parse_absnt_class_func_info(fact_nodes):
    """
    Only parse the class definition and function definition node
    """
    cls_classes=[Absnt_ClassFunction,Absnt_ClassVariable]
    func_classes = [Absnt_FunctionArgument,Absnt_FunctionReturn]
    cls_dict = node_attr_division(fact_nodes,attr_name="cls_name",target_classes=cls_classes)
    cls_sents = []
    for _, cls_nodes in cls_dict.items():
        cls_sents.append(merge_multi_siblings(cls_nodes,cls_classes,conj="or"))
    cls_sent = " ".join(cls_sents)

    func_dict = node_attr_division(fact_nodes,attr_name="func_name",target_classes=func_classes)
    func_sents = []
    for _, func_nodes in func_dict.items():
        func_sents.append(merge_multi_siblings(func_nodes,func_classes,conj="or",sort_list=True))
    func_sent = " ".join(func_sents)

    sent = cls_sent
    if sent == "" and func_sent != "":
        sent = func_sent
    elif sent != "" and func_sent != "":
        func_sent = func_sent.replace("The function ","Also, the function ")
        sent = sent + " " + func_sent
    return sent

def absnt_parser(fact_nodes):
    sent1 = parse_absnt_modal_verb(fact_nodes)
    sent2 = parse_absnt_verb_does_not_have(fact_nodes)
    sent3 = parse_absnt_class_func_info(fact_nodes)
    sent = sent1
    if sent == "" and sent2 != "":
        sent = sent2
    elif sent != "" and  sent2!="":
        sent = sent + sent2.replace("The code ", " It ")
        
    if sent == "" and sent3 !="":
        sent = sent3
    elif sent != "" and sent3 != "":
        sent = sent + " " + sent3

    return sent

def parse_facts(fact_nodes):
    sent1 = exst_parser(fact_nodes)
    sent2 = absnt_parser(fact_nodes)
    sent = sent1
    if sent == "" and sent2 != "":
        sent = sent2
    elif sent != "" and  sent2!="":
        sent = sent + sent2.replace("The code ", " However, the code ")
    return sent

TOTAL_EXST_NODES = [
        # Exst_PLNode, 
        Exst_PackageNode,Exst_ClassDefinition,Exst_ClassFunction,Exst_ClassVariable,
        Exst_FunctionArgument,Exst_FunctionDefinition,Exst_FunctionReturn,
        Exst_Statement,Exst_Expression,Exst_VariableDefinition,Exst_CodeReturn
]

TOTAL_ABSNT_NODES = [
        # Absnt_PLNode, 
        Absnt_PackageNode,Absnt_ClassDefinition,Absnt_ClassFunction,Absnt_ClassVariable,
        Absnt_FunctionArgument,Absnt_FunctionDefinition,Absnt_FunctionReturn,
        Absnt_Statement,Absnt_Expression,Absnt_VariableDefinition
]

TOTAL_NODES = [
        # Exst_PLNode, 
        Exst_PackageNode,Exst_ClassDefinition,Exst_ClassFunction,Exst_ClassVariable,
        Exst_FunctionArgument,Exst_FunctionDefinition,Exst_FunctionReturn,
        Exst_Statement,Exst_Expression,Exst_VariableDefinition,Exst_CodeReturn,
        # Absnt_PLNode, 
        Absnt_PackageNode,Absnt_ClassDefinition,Absnt_ClassFunction,Absnt_ClassVariable,
        Absnt_FunctionArgument,Absnt_FunctionDefinition,Absnt_FunctionReturn,
        Absnt_Statement,Absnt_Expression,Absnt_VariableDefinition       
]
TOTAL_EXST_NODES_NAMES = [c.__name__ for c in TOTAL_EXST_NODES]


if __name__ == "__main__":
    pass
    import random
    import numpy as np
    def generate_instrance(cls_ins):
        node_instances = []
        for node_cls in cls_ins:
            init_parameters = inspect.signature(node_cls.__init__).parameters
            para_dict = {}
            for para_name in init_parameters:
                if para_name != "self":
                    para_dict[para_name] = para_template[para_name]
            node_instances.append(node_cls(**para_dict))
        return node_instances

    print(TOTAL_EXST_NODES_NAMES)

    total_exst = [
        Exst_PLNode, Exst_PackageNode,Exst_ClassDefinition,Exst_ClassFunction,Exst_ClassVariable,
        Exst_FunctionArgument,Exst_FunctionDefinition,Exst_FunctionReturn,
        Exst_Statement,Exst_Expression,Exst_VariableDefinition,Exst_CodeReturn
    ]

    total_absnt = [
        Absnt_PLNode, Absnt_PackageNode,Absnt_ClassDefinition,Absnt_ClassFunction,Absnt_ClassVariable,
        Absnt_FunctionArgument,Absnt_FunctionDefinition,Absnt_FunctionReturn,
        Absnt_Statement,Absnt_Expression,Absnt_VariableDefinition
    ]

    para_template = {'var_names':["a","b","c"], 'lang':"Python", 'obj':"obj_a", 'attr':"split", 'stmt':"stmt_expr", 'ret':"ret_stmt", 'cls_name':"People", 'mem_var':'age', 'func_name':"my_func", 'package':"numpy"}
    
    

    exst_facts = [Exst_PLNode(lang="Python"), Exst_PackageNode(package="numpy"), 
            Exst_ClassDefinition(cls_name="People"),Exst_ClassFunction(cls_name="People",func_name="study"), Exst_ClassVariable(cls_name="People",mem_var="age"),
            Exst_FunctionDefinition(func_name="Sum"), Exst_FunctionArgument(func_name="Sum",var_names=["a","b","c"]),Exst_FunctionArgument(func_name="Sub",var_names=["a","b"]),Exst_FunctionReturn(func_name="Sum",ret="a composition list"),
            Exst_Statement("while loop"),Exst_Statement("if branch"),Exst_Expression("lambda expression"),Exst_Expression("bitwise 'XOR' operation expression"), Exst_Expression("bool expression"),
            Exst_Expression("comparison expression"),Exst_Expression("list comprehension expression"),Exst_Expression("division expression"),Exst_Expression("named expression"),
            Exst_Expression("unary operation expression"),Exst_VariableDefinition("obj_a")
            ]
    absnt_facts = [Absnt_PLNode(), Absnt_PackageNode(),
            Absnt_ClassDefinition(),Absnt_ClassFunction('People'),Absnt_ClassVariable('People'),
            Absnt_FunctionArgument("PrintMsg"),Absnt_FunctionDefinition(),Absnt_FunctionReturn("PrintMsg"),
            Absnt_Statement("while loops"),Absnt_Statement("match blocks"),Absnt_Expression("lambda expressions"),Absnt_Expression("left shift expressions"), Absnt_Expression("bool expressions"),
            Absnt_Expression("comparison expression"),Absnt_Expression("list comprehension expressions"),Absnt_Expression("add operation expressions"),Absnt_Expression("named expressions"),
            Absnt_Expression("unary operation expression"),Absnt_VariableDefinition()
            ]
    cnt = 0
    exst_facts.extend(absnt_facts)
    # for fact in exst_facts:
    #     print(str(fact))
    for i in range(100):
        selected_facts = np.random.choice(exst_facts,size=5,replace=False)
        total_sent = exst_parser(selected_facts)
        print(f"{i}: {total_sent}")
        # fw.write(f"{i}: {total_sent}\n")
    # exst_facts = [Exst_ClassDefinition(cls_name="Animal"),Exst_ClassFunction(cls_name="Animal",func_name="eat"), 
    #             Exst_ClassDefinition(cls_name="People"),Exst_ClassFunction(cls_name="People",func_name="study"), 
    #             Exst_ClassVariable(cls_name="People",mem_var="age"), Exst_ClassVariable(cls_name="People",mem_var="gender"),]

    
    
    # with open("results/codefact_template.txt","w+") as fw:
    #     for i in range(100):
    #         selected_facts = np.random.choice(exst_facts,size=5,replace=False)
    #         total_sent = parser(selected_facts)
    #         print(f"{i}: {total_sent}")
    #         fw.write(f"{i}: {total_sent}\n")



