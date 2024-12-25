
import sys
import os
import numpy as np
sys.path.append(os.getcwd())
from utils.logging_utils import MyLogger
import logging
from collections import defaultdict
import ast
from utils.common_utils import read_file
import utils.python_node as pynode
from tqdm import tqdm
from typing import Dict, List, Tuple, DefaultDict


STATEMENTS = {"delete statement":"delete statements","continue statement":"continue statements",
        "assignment statement":"assignment statements", "for loop":"for loops", "while loop":"while loops", "if branch":"if branches",
        "with block":"with blocks","match block":"match blocks","raise statement":"raise statements","try block":"try blocks",
        "assertion statement":"assertion statements","global statement":"global statements","nonlocal statement":"nonlocal statements",
        "pass statement":"pass statements","break statement":"break statements"}

EXPRESSIONS = {"lambda expression":"lambda expressions","left shift expression":"left shift expressions","right shift expression":"right shift expressions",
        "bitwise 'XOR' expression":"bitwise 'XOR' expressions","bitwise 'And' expression":"bitwise 'And' expressions","bitwise 'Or' expression":"bitwise 'Or' expressions","bool expression":"bool expressions",
        "comparison expression": "comparison expressions","list comprehension expression":"list comprehension expressions","set comprehension expression":"set comprehension expressions",
        "dict comprehension expression":"dict comprehension expressions","addition expression":"addition expressions","subtraction expression":"subtraction expressions",
        "multiplication expression":"multiplication expressions","division expression":"division expressions","modulo expression":"modulo expressions",
        "matrix multiplication expression":"matrix multiplication expressions","power calculation expression":"power calculation expressions",
        "named expression":"named expressions","generator expresssion":"generator expresssions","await expression":"await expressions",
        "yield expression":"yield expressions", "comparison expression": "comparison expressions","unary operation expression":"unary operation expressions",
        "ternary operation expression":"ternary operation expressions","dict":"dicts","set":"sets","list":"lists","f-string expression":"f-string expressions","subscript expression":"subscript expressions",
        "starred expression":"starred expressions","slicing expression":"slicing expressions"
        
        }


class CodeVisitor(ast.NodeVisitor):
    """This expands on the ast module's NodeVisitor class
    to remove any implicit visits.
    Refer to  ExplicitNodeVisitor of astor
    https://github.com/berkerpeksag/astor
    """

    def __init__(self,) -> None:
        super().__init__()
        self.nodes = []
        # the flag is used to identify the function entry point for HumanEval
        self.flag= []
        self.mapping = None
        self.var_dict = None
        self.stmt_dict = None
        self.expr_dict = None

    @property
    def node_mapping(self)->DefaultDict:
        """
        Map the ast node to line number

        Returns:
            DefaultDict: The mapping dict. lineno: [node1, node2]
        """
        if self.mapping is None:
            self.mapping = defaultdict(list)
            for node in self.nodes:
                lineno = node.lineno
                self.mapping[lineno].append(node)
        return self.mapping

    def extract_lineno(node):
        return node.lineno, node.end_lineno

    def analysis_binop(op):
        """
        NOTE: Currently I use the first op to indicate the type of the binop expression. 
        """
        if isinstance(op,ast.Add):
            return "addition expression"
        elif isinstance(op,ast.Sub):
            return "subtraction expression"
        elif isinstance(op,ast.Mult):
            return "multiplication expression"
        elif isinstance(op,ast.Div):
            return "division expression"
        elif isinstance(op,ast.FloorDiv):
            return "division expression"
        elif isinstance(op,ast.Mod):
            return "modulo expression"
        elif isinstance(op,ast.MatMult):
            return "matrix multiplication expression"
        elif isinstance(op,ast.Pow):
            return "power calculation expression"
        elif isinstance(op,ast.LShift):
            return "left shift expression"
        elif isinstance(op,ast.RShift):
            return "right shift expression"
        elif isinstance(op,ast.BitXor):
            return "bitwise 'XOR' expression"
        elif isinstance(op,ast.BitAnd):
            return "bitwise 'And' expression"
        elif isinstance(op,ast.BitOr):
            return "bitwise 'Or' expression"
        else:
            raise ValueError(f"Unexpected Binop {op}")

    def get_node_name(self, node):
        return node.__class__.__name__

    def abort_visit(self, node):  
        msg = "Oops! I just don't know how to conduct one_step_visit for %s. Use default visitor"
        # print(msg % self.get_node_name(node))
        raise ValueError(msg % self.get_node_name(node))


    ########## Import statement visitor ##########
    
    def visit_Import(self, node):
        lineno, elineno = CodeVisitor.extract_lineno(node)
        for alias in node.names:
            self.nodes.append(pynode.Exst_PackageNode(alias.name,lineno=lineno,end_lineno=elineno))            
     
    def visit_ImportFrom(self, node):
        module_name = node.module
        lineno, elineno = CodeVisitor.extract_lineno(node)
        for alias in node.names:
            self.nodes.append(pynode.Exst_PackageNode(".".join([module_name,alias.name]),lineno=lineno,end_lineno=elineno))

    ########## Class visitor ##########
    def visit_ClassDef(self, node):

        def _extract_member_variables(trgt_node:ast.FunctionDef,_mem_vars):
            if isinstance(trgt_node,ast.Tuple):
                for elt in trgt_node.elts:
                    _mem_vars = _extract_member_variables(elt,_mem_vars)
            elif isinstance(trgt_node,List):
                 for sub_node in trgt_node:
                    _mem_vars = _extract_member_variables(sub_node,_mem_vars)
            elif isinstance(trgt_node,ast.Attribute) and hasattr(trgt_node,"value") and \
                isinstance(trgt_node.value,ast.Name) and getattr(trgt_node.value,"id") == "self":
                _mem_vars.add(trgt_node.attr)
            return _mem_vars

        class_name = node.name
        self.flag.append(class_name)
        lineno, elineno = CodeVisitor.extract_lineno(node)
        self.nodes.append(pynode.Exst_ClassDefinition(cls_name=class_name,lineno=lineno,end_lineno=elineno))
        for inner_node in node.body:
            # for class instance, we only consider : 1) class definition; 2) class member variable 3) class member function
            if isinstance(inner_node,ast.Import):
                self.visit_Import(inner_node)
            if isinstance(inner_node,ast.ImportFrom):
                self.visit_ImportFrom(inner_node)
            if isinstance(inner_node,ast.FunctionDef):
                if inner_node.name == "__init__":
                    # get class member variable
                    init_body = inner_node.body
                    mem_vars = set()
                    for line_node in init_body:
                        if isinstance(line_node,ast.Assign):
                            mem_vars = _extract_member_variables(line_node.targets, mem_vars)
                        elif isinstance(line_node,ast.AugAssign) or isinstance(line_node,ast.AnnAssign):
                            mem_vars = _extract_member_variables(line_node.target, mem_vars)
                    
                    if mem_vars is not None:
                        for mem_var in mem_vars:
                            self.nodes.append(pynode.Exst_ClassVariable(cls_name=class_name,mem_var=mem_var))            
                else:
                     # get class member function
                    self.nodes.append(pynode.Exst_ClassFunction(cls_name=class_name,func_name=inner_node.name,lineno=lineno,end_lineno=elineno))             
        # visit the child node.
        self.generic_visit(node)
        # we are leaving the class
        # clean the class flag.
        self.flag.pop()

    def visit_FunctionDef(self,node):
        def _extract_args(args_node):
            _args = []
            for arg in args_node.args:
                if arg.arg == "self":
                    # we don't take `self` into account.
                    continue
                _args.append(arg.arg)
            return _args
        lineno, elineno = CodeVisitor.extract_lineno(node)
        func_name = node.name
        if func_name == "__init__":
            # we have visit __init__ func in `visit_ClassDef`
            return
        # for function instance, we only consider : 1) function definition; 2) function arguments 3) return
        self.flag.append(func_name)
        # print(node.name)
        # print(self.flag)
        # NOTE: the parser can't recognize the arguments **kwargs
        func_args = _extract_args(node.args)
        if len(func_args) != 0:
            # self.nodes.append(pynode.Exst_FunctionArgument(func_name=".".join(self.flag),var_names=func_args,lineno=lineno, end_lineno=elineno))
             self.nodes.append(pynode.Exst_FunctionArgument(func_name=self.flag[-1],var_names=func_args,lineno=lineno, end_lineno=elineno))
        # print(self.flag)
        self.nodes.append(pynode.Exst_FunctionDefinition(func_name=self.flag[-1],lineno=lineno, end_lineno=elineno))
        self.generic_visit(node)
        self.flag.pop()

    def visit_Return(self, node):
        lineno, elineno = CodeVisitor.extract_lineno(node)
        return_value = node.value
        if return_value is not None:
            method = 'visit_' + self.get_node_name(return_value)
            visitor = getattr(self, method, self.abort_visit)
            ret_str = visitor(return_value)
            if ret_str is None:
                raise ValueError(f"Unexpected return content {ret_str} in parsing {ast.unparse(node)}")
            else:
                # this is the return statement in the function definition
                if len(self.flag) > 0:
                    self.nodes.append(pynode.Exst_FunctionReturn(func_name=self.flag[-1], ret=ret_str, lineno=lineno, end_lineno=elineno))
                else:
                    # this is the return statement in the code
                    self.nodes.append(pynode.Exst_CodeReturn(ret=ret_str, lineno=lineno, end_lineno=elineno))
            self.generic_visit(node) 

    ########## Statement visitor ##########

    def visit_Delete(self, node):
        lineno, elineno = CodeVisitor.extract_lineno(node)
        self.nodes.append(pynode.Exst_Statement(stmt="delete statement",lineno=lineno,end_lineno=elineno))
        return self.generic_visit(node)   

    def fetch_names_recursively(node, name_list:list):
        if isinstance(node, ast.List) or isinstance(node, ast.Tuple):
            elts = node.elts
            for elt in elts:
                name_list = CodeVisitor.fetch_names_recursively(elt, name_list)
        elif isinstance(node, ast.Name):
            name_list.append(node.id)
        else:
            logger = logging.getLogger("mylogger")
            logger.warn("The target in current assign node may be too complicated. Just skip it.")
            # raise ValueError(f"Unexpected type for Assign target: {node}. expect Name/List/Tuple")
        return name_list
        
    def visit_Assign(self, node):
        lineno, elineno = CodeVisitor.extract_lineno(node)
        var_list = []
        for target in node.targets:
            var_list = CodeVisitor.fetch_names_recursively(target, var_list)
            for target_var in var_list:
                self.nodes.append(pynode.Exst_VariableDefinition(obj=target_var,lineno=lineno,end_lineno=elineno))
        self.nodes.append(pynode.Exst_Statement(stmt="assignment statement",lineno=lineno,end_lineno=elineno))
        self.generic_visit(node)  
        
    def visit_AugAssign(self, node):
        lineno, elineno = CodeVisitor.extract_lineno(node)
        target = node.target
        if isinstance(target, ast.Name):
            target_var = target.id
            self.nodes.append(pynode.Exst_VariableDefinition(obj=target_var,lineno=lineno,end_lineno=elineno))
        else:
            logger = logging.getLogger("mylogger")
            logger.warn("The target fact in current node may be too complicated. Just skip it.")
        
        # AugAssign Can be regard as one kind of binary operation
        op = node.op
        aug_expr = CodeVisitor.analysis_binop(op)
        self.nodes.append(pynode.Exst_Expression(stmt=aug_expr,lineno=lineno,end_lineno=elineno))
        self.nodes.append(pynode.Exst_Statement(stmt="assignment statement",lineno=lineno,end_lineno=elineno))

        self.generic_visit(node)  

    def visit_AnnAssign(self, node):
        lineno, elineno = CodeVisitor.extract_lineno(node)
        target = node.target
        if isinstance(target, ast.Name):
            target_var = target.id
            self.nodes.append(pynode.Exst_VariableDefinition(obj=target_var,lineno=lineno,end_lineno=elineno))
        else:
            logger = logging.getLogger("mylogger")
            logger.warn("The target fact in current node may be too complicated. Just skip it.")

        self.nodes.append(pynode.Exst_Statement(stmt="assignment statement",lineno=lineno,end_lineno=elineno))
        self.generic_visit(node)  

    def visit_For(self, node):
        lineno, elineno = CodeVisitor.extract_lineno(node)
        self.nodes.append(pynode.Exst_Statement(stmt="for loop",lineno=lineno,end_lineno=elineno))
        self.generic_visit(node)      

    def visit_AsyncFor(self, node):
        lineno, elineno = CodeVisitor.extract_lineno(node)
        self.nodes.append(pynode.Exst_Statement(stmt="for loop",lineno=lineno,end_lineno=elineno))
        self.generic_visit(node) 

    def visit_While(self, node):
        lineno, elineno = CodeVisitor.extract_lineno(node)
        self.nodes.append(pynode.Exst_Statement(stmt="while loop",lineno=lineno,end_lineno=elineno))
        self.generic_visit(node)    

    def visit_If(self, node):
        lineno, elineno = CodeVisitor.extract_lineno(node)
        self.nodes.append(pynode.Exst_Statement(stmt="if branch",lineno=lineno,end_lineno=elineno))
        self.generic_visit(node)

    def visit_With(self, node):
        lineno, elineno = CodeVisitor.extract_lineno(node)
        self.nodes.append(pynode.Exst_Statement(stmt="with block",lineno=lineno,end_lineno=elineno))
        self.generic_visit(node)

    def visit_AsyncWith(self, node):
        lineno, elineno = CodeVisitor.extract_lineno(node)
        self.nodes.append(pynode.Exst_Statement(stmt="with block",lineno=lineno,end_lineno=elineno))
        self.generic_visit(node)

    def visit_Match(self, node):
        lineno, elineno = CodeVisitor.extract_lineno(node)
        self.nodes.append(pynode.Exst_Statement(stmt="match block",lineno=lineno,end_lineno=elineno))
        self.generic_visit(node)

    def visit_Raise(self, node):
        lineno, elineno = CodeVisitor.extract_lineno(node)
        self.nodes.append(pynode.Exst_Statement(stmt="raise statement",lineno=lineno,end_lineno=elineno))
        self.generic_visit(node)

    def visit_Try(self, node):
        lineno, elineno = CodeVisitor.extract_lineno(node)
        self.nodes.append(pynode.Exst_Statement(stmt="try block",lineno=lineno,end_lineno=elineno))
        self.generic_visit(node)

    def visit_Assert(self, node):
        lineno, elineno = CodeVisitor.extract_lineno(node)
        self.nodes.append(pynode.Exst_Statement(stmt="assertion statement",lineno=lineno,end_lineno=elineno))
        self.generic_visit(node)

    def visit_Global(self, node):
        lineno, elineno = CodeVisitor.extract_lineno(node)
        self.nodes.append(pynode.Exst_Statement(stmt="global statement",lineno=lineno,end_lineno=elineno))
        self.generic_visit(node)

    def visit_Nonlocal(self, node):
        lineno, elineno = CodeVisitor.extract_lineno(node)
        self.nodes.append(pynode.Exst_Statement(stmt="nonlocal statement",lineno=lineno,end_lineno=elineno))
        self.generic_visit(node)

    def visit_Expr(self, node):
        self.generic_visit(node)

    def visit_Pass(self, node):
        lineno, elineno = CodeVisitor.extract_lineno(node)
        self.nodes.append(pynode.Exst_Statement(stmt="pass statement",lineno=lineno,end_lineno=elineno))
        self.generic_visit(node)

    def visit_Break(self, node):
        lineno, elineno = CodeVisitor.extract_lineno(node)
        self.nodes.append(pynode.Exst_Statement(stmt="break statement",lineno=lineno,end_lineno=elineno))
        self.generic_visit(node)

    def visit_Continue(self, node):
        lineno, elineno = CodeVisitor.extract_lineno(node)
        self.nodes.append(pynode.Exst_Statement(stmt="continue statement",lineno=lineno,end_lineno=elineno))
        self.generic_visit(node) 


    ########## Expression visitor ##########

    def visit_BoolOp(self, node):
        expr_type = "bool expression"
        lineno, elineno = CodeVisitor.extract_lineno(node)
        self.nodes.append(pynode.Exst_Expression(stmt=expr_type,lineno=lineno,end_lineno=elineno))
        self.generic_visit(node)
        return expr_type

    def visit_NamedExpr(self, node):
        expr_type = "named expression"
        lineno, elineno = CodeVisitor.extract_lineno(node)
        self.nodes.append(pynode.Exst_Expression(stmt=expr_type,lineno=lineno,end_lineno=elineno))
        self.generic_visit(node)
        return expr_type

    def visit_BinOp(self, node):
        expr_type = CodeVisitor.analysis_binop(node.op)
        lineno, elineno = CodeVisitor.extract_lineno(node)
        self.nodes.append(pynode.Exst_Expression(stmt=expr_type,lineno=lineno,end_lineno=elineno))
        self.generic_visit(node)
        return expr_type

    def visit_UnaryOp(self, node):
        expr_type = "unary operation expression"
        lineno, elineno = CodeVisitor.extract_lineno(node)
        self.nodes.append(pynode.Exst_Expression(stmt=expr_type,lineno=lineno,end_lineno=elineno))
        self.generic_visit(node)
        return expr_type

    def visit_Lambda(self, node):
        expr_type = "lambda expression"
        lineno, elineno = CodeVisitor.extract_lineno(node)
        self.nodes.append(pynode.Exst_Expression(stmt=expr_type,lineno=lineno,end_lineno=elineno))
        self.generic_visit(node)
        return expr_type

    def visit_IfExp(self, node) :
        expr_type = "ternary operation expression"
        lineno, elineno = CodeVisitor.extract_lineno(node)
        self.nodes.append(pynode.Exst_Expression(stmt=expr_type,lineno=lineno,end_lineno=elineno))
        self.generic_visit(node)
        return expr_type

    def visit_Dict(self, node):
        expr_type = "dict"
        lineno, elineno = CodeVisitor.extract_lineno(node)
        self.nodes.append(pynode.Exst_Expression(stmt=expr_type,lineno=lineno,end_lineno=elineno))
        self.generic_visit(node)
        return expr_type

    def visit_Set(self, node):
        expr_type = "set"
        lineno, elineno = CodeVisitor.extract_lineno(node)
        self.nodes.append(pynode.Exst_Expression(stmt=expr_type,lineno=lineno,end_lineno=elineno))
        self.generic_visit(node)
        return expr_type

    def visit_ListComp(self, node):
        expr_type = "list comprehension expression"
        lineno, elineno = CodeVisitor.extract_lineno(node)
        self.nodes.append(pynode.Exst_Expression(stmt=expr_type,lineno=lineno,end_lineno=elineno))
        self.generic_visit(node)
        return expr_type

    def visit_SetComp(self, node):
        expr_type = "set comprehension expression"
        lineno, elineno = CodeVisitor.extract_lineno(node)
        self.nodes.append(pynode.Exst_Expression(stmt=expr_type,lineno=lineno,end_lineno=elineno))
        self.generic_visit(node)
        return expr_type

    def visit_DictComp(self, node):
        expr_type = "dict comprehension expression"
        lineno, elineno = CodeVisitor.extract_lineno(node)
        self.nodes.append(pynode.Exst_Expression(stmt=expr_type,lineno=lineno,end_lineno=elineno))
        self.generic_visit(node)
        return expr_type

    def visit_GeneratorExp(self, node):
        expr_type = "generator expresssion"
        lineno, elineno = CodeVisitor.extract_lineno(node)
        self.nodes.append(pynode.Exst_Expression(stmt=expr_type,lineno=lineno,end_lineno=elineno))
        self.generic_visit(node)
        return expr_type

    def visit_Await(self, node):
        expr_type ="await expression"
        lineno, elineno = CodeVisitor.extract_lineno(node)
        self.nodes.append(pynode.Exst_Expression(stmt=expr_type,lineno=lineno,end_lineno=elineno))
        self.generic_visit(node)
        return expr_type

    def visit_Yield(self, node):
        expr_type = "yield expression"
        lineno, elineno = CodeVisitor.extract_lineno(node)
        self.nodes.append(pynode.Exst_Expression(stmt=expr_type,lineno=lineno,end_lineno=elineno))
        return expr_type

    def visit_YieldFrom(self, node):
        expr_type = "yield expression"
        lineno, elineno = CodeVisitor.extract_lineno(node)
        self.nodes.append(pynode.Exst_Expression(stmt=expr_type,lineno=lineno,end_lineno=elineno))
        self.generic_visit(node)
        return expr_type

    def visit_Compare(self, node):
        # EQ, NOTEQ, LT, LTE, GT, GTE, IS, ISNOT, IN, NOTIN
        expr_type =  "comparison expression"
        lineno, elineno = CodeVisitor.extract_lineno(node)
        self.nodes.append(pynode.Exst_Expression(stmt=expr_type,lineno=lineno,end_lineno=elineno))
        self.generic_visit(node)
        return expr_type

    def visit_Call(self, node):
        func_object = node.func
        lineno, elineno = CodeVisitor.extract_lineno(node)
        # only three situations: ...f.a(x) and ...f(x).a(x) and a(x)
        # in ast.Attribute, the value is the object and the attr is the attribution
        # Case 1: a(x)
        if isinstance(func_object, ast.Name):
            func_name = func_object.id
            self.generic_visit(node)
            return_str =  f"function call '{func_name}'"
        # Case 2-3: f.a(x) and f(x).a(x)
        elif isinstance(func_object,ast.Attribute):
            attr_func = func_object.attr
            if isinstance(attr_func, str):
                return_str =  f"function call '{attr_func}'"
            else:
                raise TypeError(f"The type of attr_func is supposed to be Str, but got {type(attr_func)}")
        elif isinstance(func_object,ast.Call) and isinstance(func_object.func,ast.Name):
            func_name = func_object.func.id
            return_str = f"function call '{func_name}'"
        else:
            # Such as Conv[x](i), we return the Conv
            # TODO: to add one generic `call` code feature
            raise ValueError(f"Unexpected func type in ast.Call: {type(func_object)}")
            # return_str = None
            # pass
            # return_str =  f"function call '{attr_func}'"
        if return_str is not None:
            self.nodes.append(pynode.Exst_Expression(stmt=return_str,lineno=lineno,end_lineno=elineno))
        self.generic_visit(node)
        return return_str

    def visit_FormattedValue(self, node):
        """For such a expression we didn't update the status dict since it's too common."""
        self.generic_visit(node)

    def visit_JoinedStr(self, node):
        expr_type = "f-string expression"
        lineno, elineno = CodeVisitor.extract_lineno(node)
        self.nodes.append(pynode.Exst_Expression(stmt=expr_type,lineno=lineno,end_lineno=elineno))
        self.generic_visit(node)
        return expr_type

    def visit_Constant(self, node):
        # expr_type = "constant expression"
        # lineno, elineno = CodeVisitor.extract_lineno(node)
        # self.nodes.append(pynode.Exst_Expression(stmt=expr_type,lineno=lineno,end_lineno=elineno))
        self.generic_visit(node)
        return f"constant {node.value}"


    def visit_Attribute(self, node):
        # only two situations: ...f.a and ...f(x).a
        # in ast.Attribute, the value is the object and the attr is the attribution
        node_attr = node.attr

        if isinstance(node_attr, str):
            expr_type =  f"accessing of attribute '{node_attr}'"
        else:
            raise TypeError(f"The type of node_attr is supposed to be Str, but got {type(node_attr)}")
        self.generic_visit(node)
        return expr_type


    def visit_Subscript(self, node):
        expr_type ="subscript expression"
        lineno, elineno = CodeVisitor.extract_lineno(node)
        self.nodes.append(pynode.Exst_Expression(stmt=expr_type,lineno=lineno,end_lineno=elineno))
        self.generic_visit(node)
        return expr_type

    def visit_Starred(self, node):
        expr_type ="starred expression"
        lineno, elineno = CodeVisitor.extract_lineno(node)
        self.nodes.append(pynode.Exst_Expression(stmt=expr_type,lineno=lineno,end_lineno=elineno))
        self.generic_visit(node)
        return expr_type

    def visit_Name(self, node):
        # """For such a expression we didn't update the status dict since it's too common."""
        # return f"variable {node.id}"
        expr_type = f"variable '{node.id}'"
        # lineno, elineno = CodeVisitor.extract_lineno(node)
        # self.nodes.append(pynode.Exst_VariableDefinition(obj=node.id,lineno=lineno,end_lineno=elineno))
        self.generic_visit(node)
        return expr_type

    def visit_List(self, node):
        expr_type ="list"
        lineno, elineno = CodeVisitor.extract_lineno(node)
        self.nodes.append(pynode.Exst_Expression(stmt=expr_type,lineno=lineno,end_lineno=elineno))
        self.generic_visit(node)
        return expr_type

    def visit_Tuple(self, node):
        # NOTE: tuple is too common in python code. For code `for i,j in range(10)`, the `i,j` would be parsed into tuple.
        # Thus, such description is not natural. We remove tuple from the entire set.
        expr_type ="tuple"
        # lineno, elineno = CodeVisitor.extract_lineno(node)
        # self.nodes.append(pynode.Exst_Expression(stmt=expr_type,lineno=lineno,end_lineno=elineno))
        self.generic_visit(node)
        return expr_type

    def visit_Slice(self, node):
        expr_type ="slicing expression"
        lineno, elineno = CodeVisitor.extract_lineno(node)
        self.nodes.append(pynode.Exst_Expression(stmt=expr_type,lineno=lineno,end_lineno=elineno))
        self.generic_visit(node)
        return expr_type

    ########## Visitor End ##########

def node_division(nodes,all_nodes):
    # node_dict = {nc.__name__:[] for nc in all_nodes}
    node_dict = defaultdict(list)
    visited_desc = set()
    unique_facts = []
    for node in nodes:
        for cls_name in all_nodes:
            if isinstance(node,cls_name) and str(node) not in visited_desc:
                unique_facts.append(node)
                node_dict[cls_name.__name__].append(node)
                visited_desc.add(str(node))
    return unique_facts, node_dict

def generate_absnt_node(exst_nodes):
    # dependency-level absnt node
    _, node_dict = node_division(exst_nodes,pynode.TOTAL_EXST_NODES)
    absnt_nodes = []
    # absnt_nodes.append(pynode.Absnt_PLNode())
    if len(node_dict["Exst_PackageNode"]) == 0:
        absnt_nodes.append(pynode.Absnt_PackageNode())
    if len(node_dict["Exst_ClassDefinition"]) == 0:
        absnt_nodes.append(pynode.Absnt_ClassDefinition())
    else:
        # we check whether we have class member variables/function definition
        # for each class, we add the absnt facts if there is no variables and funtion definitions
        cls_set = set()
        for cls_node in node_dict["Exst_ClassDefinition"]:
            cls_set.add(cls_node.cls_name)
        cls_func = defaultdict(list)
        cls_var = defaultdict(list)
        for cls_func_node in node_dict["Exst_ClassFunction"]:
            cls_func[cls_func_node.cls_name].append(cls_func_node)
        for cls_var_node in node_dict["Exst_ClassVariable"]:
            cls_var[cls_var_node.cls_name].append(cls_var_node)
        for cls_name in cls_set:
            if len(cls_func[cls_name]) == 0:
                absnt_nodes.append(pynode.Absnt_ClassFunction(cls_name))
            if len(cls_var[cls_name]) == 0:
                absnt_nodes.append(pynode.Absnt_ClassVariable(cls_name))
    if len(node_dict['Exst_FunctionDefinition']) == 0:
        absnt_nodes.append(pynode.Absnt_FunctionDefinition())
    else:
        func_set = set()
        for func_node in node_dict["Exst_FunctionDefinition"]:
            func_set.add(func_node.func_name)
        func_arg = defaultdict(list)
        func_ret = defaultdict(list)
        for func_arg_node in node_dict["Exst_FunctionArgument"]:
            func_arg[func_arg_node.func_name].append(func_arg_node)
        for func_ret_node in node_dict["Exst_FunctionReturn"]:
            func_ret[func_ret_node.func_name].append(func_ret_node)
        for func_name in func_set:
            if len(func_arg[func_name]) == 0:
                absnt_nodes.append(pynode.Absnt_FunctionArgument(func_name))
            if len(func_ret[func_name]) == 0:
                absnt_nodes.append(pynode.Absnt_FunctionReturn(func_name))

    exist_stmts = {s.stmt for s in node_dict['Exst_Statement']}
    absnt_stmts = set(list(STATEMENTS.keys())) - exist_stmts
    for absnt_stmt in absnt_stmts:
        absnt_nodes.append(pynode.Absnt_Statement(stmt=STATEMENTS[absnt_stmt]))

    exist_exprs = {s.stmt for s in node_dict['Exst_Expression']}
    absnt_exprs = set(list(EXPRESSIONS.keys())) - exist_exprs
    for absnt_expr in absnt_exprs:
        absnt_nodes.append(pynode.Absnt_Expression(stmt=EXPRESSIONS[absnt_expr]))

    if len(node_dict['Exst_VariableDefinition']) == 0:
        absnt_nodes.append(pynode.Absnt_VariableDefinition())
    # if len(node_dict['Exst_VariableAttribute']):
    #     absnt_nodes.append(pynode.Absnt_VariableAttribute())

    return absnt_nodes


if __name__ == "__main__":
    import random
    mylogger = MyLogger(log_path="./logs/test.log")
    root_path = os.getcwd()
    # content_path = os.path.join(root_path,"data/20230227-manual_check/humaneval_codex/origin/prompt_32.txt")
    # # content_path = os.path.join(root_path,"test_scripts/template_code.txt")
    # content = read_file(content_path)
    content = """
n = int(input())
a = [list(map(int, input().split())) for i in range(n)]

def is_square(m, n):
    s = sum(a[i][j] for i in range(m) for j in range(n))
    return s == (m + n) * (m + n) / 2 and s % 2 == 0

for i in range(n):
    if not any(a[i][j] == 0 for j in range(n)):
        if is_square(n, n):
            return n ** 2

print(-1)"""
    print(content)
    code = ast.parse(content,type_comments=True)
    visitor = CodeVisitor()
    visitor.visit(code)
    nodes = visitor.nodes
    absnt_nodes = generate_absnt_node(nodes)




    nodes.extend(absnt_nodes)
    # for node in nodes:
    #     print(node.__class__.__name__, node.lineno,": ", str(node))
    
    selected_nodes = np.random.choice(nodes,size=5)
    unique_set = set()
    unique_facts = []
    for n in nodes:
        # print(n.__class__.__name__,str(n))
        if str(n) not in unique_set:
            unique_set.add(str(n))
            unique_facts.append(n)
    # for s in unique_set:
    #     print(s)
    unique_set = sorted(list(unique_set))
    for s in unique_set:
        print(s)

    parsed_str = pynode.parse_facts(selected_nodes)
    print(parsed_str)


    # # pass
    # dataset_names = ["humaneval_codex", "humaneval_codegen-350M-multi"]
    # mylogger = MyLogger(log_path="./logs/test.log")
    # root_path = os.getcwd()

    
    # for dataset_name in dataset_names:
    #     content_path = os.path.join(root_path,f"data/20230227-manual_check/{dataset_name}/origin")
    #     files = os.listdir(content_path)
    #     random.shuffle(files)
    #     cnt = 0
    #     with open(f"./logs/{dataset_name}_attacks.txt","w") as fw:
    #         for file in files:
    #             cnt += 1
    #             if cnt > 20:
    #                 break
    #             if not file.endswith(".txt"):
    #                 continue
    #             prompt_id = file.split("_")[-1][:-4]
                
    #             fw.write(f"{'='*10}{prompt_id}{'='*10}\n")
    #             file_path = os.path.join(content_path, file)
    #             content = read_file(file_path)
    #             fw.write(f"{content}\n")
    #             code = ast.parse(content,type_comments=True)
    #             visitor = CodeVisitor()
    #             visitor.visit(code)
    #             nodes = visitor.nodes
    #             absnt_nodes = generate_absnt_node(nodes)
    #             nodes.extend(absnt_nodes)
            
    #             unique_set = set()
    #             unique_facts = []
    #             for n in nodes:
    #                 if str(n) not in unique_set:
    #                     unique_set.add(str(n))
    #                     unique_facts.append(n)
    #             unique_set = sorted(list(unique_set))
    #             for s in unique_set:
    #                 fw.write(f"{s}\n")
    #             fw.write(f"{'='*30}\n")




