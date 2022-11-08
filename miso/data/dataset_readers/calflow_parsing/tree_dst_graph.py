# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from collections import defaultdict, Counter
from typing import List, Any
import pdb 
import copy 
import re 
import json
import numpy as np
from overrides import overrides

import networkx as nx
from allennlp.data.vocabulary import DEFAULT_PADDING_TOKEN, DEFAULT_OOV_TOKEN

from miso.data.dataset_readers.decomp_parsing.decomp import SourceCopyVocabulary
from miso.data.dataset_readers.calflow_parsing.calflow_graph import CalFlowGraph, NOBRACK, PROGRAM_SEP, PAD_EDGE, DETOKENIZER, SPACY_NLP, CARD_GEX, CARD_DETOK_GEX
from dataflow.core.program import Program, Expression, BuildStructOp, ValueOp, CallLikeOp, TypeName
from dataflow.core.lispress import (parse_lispress, 
                                    program_to_lispress, 
                                    lispress_to_program,  
                                    render_compact, 
                                    render_pretty)


PHONE_NUM_GEX = re.compile("[0-9]{11}")

class TreeDSTGraph(CalFlowGraph):
    def __init__(self, 
                src_str: str,
                tgt_str: str,
                use_program: bool = False,
                use_agent_utterance: bool = False, 
                use_context: bool = False,
                fxn_of_interest: str = None,
                line_idx: int = None,
                expression_probs: List[float] = None):
        super(TreeDSTGraph, self).__init__(src_str=src_str, 
                                           tgt_str=tgt_str,
                                           use_program=use_program,
                                           use_agent_utterance=use_agent_utterance,
                                           use_context=use_context,
                                           fxn_of_interest=fxn_of_interest,
                                           line_idx=line_idx,
                                           expression_probs=expression_probs)
    @overrides
    def fill_lists_from_program(self, program: Program):            
        def get_arg_num(eid):
            parent_edges = list(self.dep_chart.in_edges(eid))
            parent_edge = parent_edges[-1]

            edge_order_idx = self.dep_chart.edges[parent_edge]['order_idx']
            return edge_order_idx

        # node lookup dict takes expression IDs and turns them into node ids of their parent nodes 
        parent_node_lookup = defaultdict(list)
        program_new = copy.deepcopy(program)

        # helper function for adding type_args recursively
        def add_type_name(type_name, parent_nidx, arg_idx):
            curr_idx = self.node_idx_list[-1] + 1 
            self.node_idx_list.append(curr_idx) 
            self.node_name_list.append(type_name.base)
            self.edge_head_list.append(parent_nidx)
            self.edge_type_list.append(f"type_arg-{arg_idx}")
            for i, ta in enumerate(type_name.type_args):
                add_type_name(ta, curr_idx + self.n_reentrant, i)

        # rules for adding 
        def add_struct_op(e: Expression, eidx: int, nidx: int):
            id = int(NOBRACK.sub("",e.id))
            try:
                parent_node_idx, argn = parent_node_lookup[id][-1]
            except (KeyError, IndexError) as ex:
                # add edge to dummy root 
                parent_node_idx, argn = 0, 0
                parent_node_lookup[id].append((parent_node_idx + self.n_reentrant, argn))

            # add node idx, with repeats for re-entrancy
            for i, (parent_node_idx, reent_argn) in enumerate(parent_node_lookup[id]):
                self.node_idx_list.append(nidx)
                # add schema 
                self.node_name_list.append(e.op.op_schema) 
                # add an edge from the expression to its parent 
                self.edge_head_list.append(parent_node_idx)

                try:
                    fxn_argn = get_arg_num(id)
                except IndexError:
                    fxn_argn = 0

                self.edge_type_list.append(f"fxn_arg-{fxn_argn}")
                # update lookup 
                self.node_idx_to_expr_idx[nidx] = eidx
                self.expr_idx_to_node_idx[eidx] = nidx
                # increment counter so that edge indices are correct 
                if i > 0:
                    self.n_reentrant += 1

            # add type args
            type_args = e.type_args
            if type_args is not None:
                for i, ta in enumerate(type_args): 
                    add_type_name(ta, nidx + self.n_reentrant, i)

            op_fields = e.op.op_fields
            ## add op nodes only once 
            try:
                assert(len(op_fields) == len(e.arg_ids))
            except AssertionError:
                assert(not e.op.empty_base)
                # if there's a base, prepend "base" to op fields 
                op_fields = ["nonEmptyBase"] + op_fields
                assert(len(op_fields) == len(e.arg_ids))

            fields_added = 0
            for i, (field, dep) in enumerate(zip(op_fields, e.arg_ids)):
                if field is not None:
                    field_node_idx = nidx + fields_added + 1
                    fields_added += 1
                    # self.node_idx_list.append(field_node_idx)
                    # TODO (elias): what about re-entrancy 
                    self.node_idx_list.append(self.node_idx_list[-1] + 1)
                    self.node_name_list.append(field)
                    self.edge_head_list.append(nidx + self.n_reentrant)
                    self.edge_type_list.append(f"arg-{i}")

                arg_id = int(NOBRACK.sub("", dep))
                parent_node_lookup[arg_id].append((nidx+ self.n_reentrant, 0))

        def tokenize_target(underlying):
            toked = [x.text for x in SPACY_NLP(underlying)]
            # TODO handle numbers like 22nd, 1st, 4th, etc.
            to_ret = []
            for tok in toked:
                m = CARD_GEX.match(tok)
                if m is not None:
                    # split into number and suffix 
                    num = m.group(1)
                    suffix = m.group(2)
                    to_ret.append(num)
                    to_ret.append(suffix)
                else:
                    to_ret.append(tok)
            return to_ret 

        def add_value_op(e: Expression, eidx: int, nidx: int):
            id = int(NOBRACK.sub("",e.id))
            reentrant = False
            if len(parent_node_lookup[id]) > 1:
                reentrant = True

            if len(parent_node_lookup[id]) == 0:
                parent_node_idx, argn = 0, 0
                parent_node_lookup[id].append((parent_node_idx + self.n_reentrant, argn))
            #if reentrant:
            #    nidx = self.expr_idx_to_node_idx[eidx]

            op_value_dict = json.loads(e.op.value)
            for i, (parent_node_idx, argn) in enumerate(parent_node_lookup[id]):
                # add type 
                self.node_name_list.append(op_value_dict['schema'])
                # add node idx
                self.node_idx_list.append(nidx)
                # add edge 
                self.edge_head_list.append(parent_node_idx)

                try:
                    val_argn = get_arg_num(id)
                except IndexError:
                    val_argn = 0

                self.edge_type_list.append(f"val_arg-{val_argn}")
                # update lookup 
                self.node_idx_to_expr_idx[nidx] = eidx
                self.expr_idx_to_node_idx[eidx] = nidx
                # increment counter so that edge indices are correct 
                if i > 0:
                    self.n_reentrant += 1 

            # add underlying 
            nested = False
            underlying = op_value_dict['underlying']
            try:
                # NOTE (elias): here is where we tokenize the underlying value 
                underlying = " ".join(tokenize_target(underlying))
                # underlying = " ".join([t.text for t in underlying_tokens])
            except TypeError:
                pass 

            try:
                value_words = underlying.strip().split(" ")
            except AttributeError:
                try:
                    assert(type(underlying) in [int, float, bool, list])
                except AssertionError:
                    pdb.set_trace() 
                # deal with underlying lists
                if type(underlying) == list:                         
                    if len(underlying) >= 1:                            
                        # nested 
                        nested = True
                        outer_value_words = []
                        for word_or_list in underlying[0][0]:
                            if type(word_or_list) == str:
                                outer_value_words.append(word_or_list)
                            elif type(word_or_list) == list:
                                if type(word_or_list[0]) == list:
                                    outer_value_words.append([str(x) for x in word_or_list[0]])
                                else:
                                    outer_value_words.append([str(x) for x in word_or_list])

                        #outer_value_words = [str(x) for x in underlying[0][0]]
                        try:
                            inner_value_words = [str(x) for x in underlying[1]]
                        except IndexError:
                            inner_value_words = []
                else:
                    value_words = [str(underlying)]

            if not nested:
                for i, word in enumerate(value_words): 
                    self.node_name_list.append(word)
                    self.node_idx_list.append(self.node_idx_list[-1] + 1) 
                    self.edge_head_list.append(nidx + self.n_reentrant)
                    self.edge_type_list.append(f"arg-{i}")
            else:
                # reverse so that the function type comes before the type constraints, e.g. [roleConstraint, [[Constraint, DateTime]], '^']
                outer_value_words = outer_value_words[::-1]
                outer_parent_idx = -1
                for i, word_or_list in enumerate(outer_value_words): 
                    if type(word_or_list) == str:
                        if i == 0:
                            self.node_name_list.append(word_or_list)
                            self.node_idx_list.append(self.node_idx_list[-1] + 1) 
                            self.edge_head_list.append(nidx + self.n_reentrant)
                            self.edge_type_list.append(f"arg-{i}")
                            outer_parent_idx = nidx + i + 1
                        else:
                            assert(word_or_list == "^")
                            # don't add this, just sugar 
                            continue
                    elif type(word_or_list) == list:
                        for j, word in enumerate(word_or_list):
                            self.node_name_list.append(word)
                            self.node_idx_list.append(self.node_idx_list[-1] + 1)
                            # set parent to outer function type 
                            self.edge_head_list.append(outer_parent_idx)
                            self.edge_type_list.append(f"type_arg-{j}")

                for i, word in enumerate(inner_value_words): 
                    self.node_name_list.append(word)
                    self.node_idx_list.append(self.node_idx_list[-1] + 1) 
                    self.edge_head_list.append(outer_parent_idx)
                    self.edge_type_list.append(f"inner_arg-{i}")

        def add_call_like_op(e: Expression, eidx: int, nidx: int):
            id = int(NOBRACK.sub("",e.id))
            try:
                parent_node_idx, argn = parent_node_lookup[id][-1]
            except IndexError:
                # attach to root 
                parent_node_lookup[id] = [(0,0)]

            for i, (parent_node_idx, argn) in enumerate(parent_node_lookup[id]):
                # NOTE (elias): tree DST specific: for lambdas, only add one, not 2 
                #if e.op.name == "lambda_arg" and self.node_name_list[-1] == "lambda_arg":
                    # pdb.set_trace()
                    # skip the second lambda arg, we'll add it in post-hoc 
                    # continue 

                # add type 
                self.node_name_list.append(e.op.name) 
                # add node idx
                self.node_idx_list.append(nidx)
                # add edge 
                self.edge_head_list.append(parent_node_idx)

                try:
                    call_argn = get_arg_num(id)
                except IndexError:
                    call_argn = 0

                self.edge_type_list.append(f"call_arg-{call_argn}")
                # update lookup 
                self.node_idx_to_expr_idx[nidx] = eidx
                self.expr_idx_to_node_idx[eidx] = nidx
                # increment counter so that edge indices are correct 
                if i > 0:
                    self.n_reentrant += 1

            # add type args
            type_args = e.type_args
            if type_args is not None:
                for i, ta in enumerate(type_args): 
                    add_type_name(ta, nidx + self.n_reentrant, i)

            for i, dep in enumerate( e.arg_ids):
                field_node_idx = nidx 
                arg_id = int(NOBRACK.sub("", dep))
                parent_node_lookup[arg_id].append((field_node_idx + self.n_reentrant, i))

        # add dummy root node 
        self.node_name_list.append("@ROOT@")
        self.edge_head_list.append(0)
        self.edge_type_list.append('root')
        self.node_idx_list.append(0)

        # reverse to get correct order 
        program_new.expressions.reverse() 
        for eidx, expr in enumerate(program_new.expressions):
            try:
                node_idx = self.node_idx_list[-1] + 1
            except IndexError:
                # 0 is reserved for root 
                node_idx = 1
            expr_idx = int(NOBRACK.sub("", expr.id))
            if isinstance(expr.op, BuildStructOp):
                add_struct_op(expr, expr_idx, node_idx)
            elif isinstance(expr.op, ValueOp):
                add_value_op(expr, expr_idx, node_idx)
            elif isinstance(expr.op, CallLikeOp):
                if len(expr.arg_ids) > len(set(expr.arg_ids)):
                    # we have double for re-entrancy, which will cause problems; replace
                    new_expr = Expression(id = expr.id, op = expr.op, arg_ids = list(set(expr.arg_ids)), type_args = expr.type_args)
                    expr = new_expr 
                add_call_like_op(expr, expr_idx, node_idx)
            else:
                raise ValueError(f"Unexpected Expression: {expr}")

    @staticmethod
    def prediction_to_program(node_name_list: List[str],
                              node_idx_list: List[int], 
                              edge_head_list: List[int], 
                              edge_type_list: List[str],
                              node_probs_list: List[float]): 

        graph = TreeDSTGraph.lists_to_ast(node_name_list, node_idx_list, edge_head_list, edge_type_list, node_probs_list=node_probs_list)

        def get_arg_children(op_node):
            outgoing_edges = [e for e in graph.edges if e[0] == op_node and graph.edges[e]['type'].startswith("arg")]
            outgoing_edges = sorted(outgoing_edges, key = lambda e: int(graph.edges[e]['type'].split("-")[1]))
            return [e[1] for e in outgoing_edges]

        def get_inner_arg_children(op_node):
            outgoing_edges = [e for e in graph.edges if e[0] == op_node and graph.edges[e]['type'].startswith("inner_arg")]
            outgoing_edges = sorted(outgoing_edges, key = lambda e: int(graph.edges[e]['type'].split("-")[1]))
            return [e[1] for e in outgoing_edges]

        def get_type_arg_children(op_node):
            outgoing_edges = [e for e in graph.edges if e[0] == op_node and graph.edges[e]['type'].startswith("type_arg")]
            outgoing_edges = sorted(outgoing_edges, key = lambda e: int(graph.edges[e]['type'].split("-")[1]))

            return [e[1] for e in outgoing_edges]

        def get_fxn_children(op_node):
            # ordering = fxn_arg, call_arg, value_arg
            order_augment = {"fxn_arg": 0,
                             "call_arg": 0,
                             "val_arg": 0}

            def get_order_num(edge_type):
                etype, argn = edge_type.split("-")
                argn=int(argn)
                eord = order_augment[etype]
                return eord + argn

            outgoing_edges = [e for e in graph.edges if e[0] == op_node and (graph.edges[e]['type'].startswith("fxn_arg")
                                                                            or graph.edges[e]['type'].startswith("val_arg")
                                                                            or graph.edges[e]['type'].startswith("call_arg"))]

            outgoing_edges = sorted(outgoing_edges, key = lambda e: get_order_num(graph.edges[e]['type']), reverse=False)
            return [e[1] for e in outgoing_edges]
            
        def get_parent(op_node):
            incoming_edges = [e for e in graph.edges if e[1] == op_node]
            return [e[0] for e in incoming_edges][0]

        def get_type_args(node):
            nested = []
            type_arg_node_idxs = get_type_arg_children(node)
            for ta_node in type_arg_node_idxs:
                name = graph.nodes[ta_node]['node_name']
                #ta_subchildren = get_type_arg_children(ta_node)
                type_args = get_type_args(ta_node)
                if type_args is None:
                    type_args = []
                type_name = TypeName(base=name, type_args = type_args)
                nested.append(type_name)
            if len(nested)>0:
                return nested
            return None

        node_id_to_expr_id = {}
        is_expr = [True if (not et.startswith("arg") \
                            and not et.startswith("inner_arg") \
                            and not et.startswith("type_arg")) else False for et in edge_type_list]
        n_expr = sum(is_expr)

        n_reentrant = 0
        for i, node_idx in enumerate(node_idx_list):
            if i - n_reentrant != node_idx:
                n_reentrant += 1

        n_expr -= n_reentrant

        curr_expr = n_expr
        for i in range(len(edge_type_list)):
            # if it's either a build, value, or call
            if is_expr[i]: 
                node_id = node_idx_list[i]
                if node_id not in node_id_to_expr_id.keys():
                    node_id_to_expr_id[node_id] = curr_expr + 1
                    curr_expr -= 1

        def do_detokenize(str_list):
            detokenized = DETOKENIZER.detokenize(str_list)
            detokenized = re.sub(r" - ", "-", detokenized)
            m = CARD_DETOK_GEX.search(detokenized)
            if m is not None: 
                # remove space 
                with_space = m.group(0)
                without_space = with_space.replace(" ", "")
                detokenized = re.sub(with_space, without_space, detokenized)
            detokenized = f" {detokenized.strip()} "
            return detokenized

        expressions = []
        expression_probs = []
        add_to_next = []
        is_lambda = False
        for node in sorted(graph.nodes):
            op = None
            op_prob = 1.0
            if graph.nodes[node]['function_type'] is not None: 
                if graph.nodes[node]['function_type'] == "build":
                    empty_base = True
                    children = get_fxn_children(node) 
                    field_children = get_arg_children(node)
                    name = graph.nodes[node]['node_name']

                    if len(field_children) == 0:
                        field_names = [None for i in range(len(children))]
                    else:
                        field_names = [graph.nodes[child]['node_name'] for child in field_children]
                        fxn_children = get_fxn_children(node) 
                        if len(field_names) != len(fxn_children):
                            # add difference 
                            field_names = [None for i in range(len(fxn_children) - len(field_names))] + field_names


                    op = BuildStructOp(op_schema = name, 
                                       op_fields = field_names,
                                       empty_base = empty_base, 
                                       push_go = True)

                    if graph.nodes[node]['node_prob'] is not None:
                        # construct prob taking into account all children 
                        op_prob = graph.nodes[node]['node_prob']
                        for child in children + field_children:
                            op_prob *= graph.nodes[child]['node_prob']

                elif graph.nodes[node]['function_type'] == "value": 
                    children = get_arg_children(node)
                    inner_gchildren = []
                    for n in children:
                        inner_gchildren += get_inner_arg_children(n) 
                    name = graph.nodes[node]['node_name']
                    child_names = [graph.nodes[child]['node_name'] for child in children]
                        # underlying = " ".join(child_names) 
                    # NOTE (elias): adding this to detokenize apostrophes 
                    underlying = do_detokenize(child_names)


                    ## check for ints 
                    if len(child_names) == 1:
                        if child_names[0].isdigit() and not PHONE_NUM_GEX.match(child_names[0]):
                            underlying = int(child_names[0])
                        elif re.match("-?\d+\.\d+", child_names[0]) is not None: 
                            underlying = float(child_names[0])
                        elif child_names[0].strip() in ["True", "False"]:
                            underlying = True if child_names[0].strip() == "True" else False
                        elif child_names[0].strip() == "[]":
                            underlying = []
                        else:
                            # needs to not have spaces for sugaring to work 
                            parent_name = graph.nodes[node]['node_name']
                            grandparent_idx = get_parent(node)
                            grandparent_name = graph.nodes[grandparent_idx]['node_name']
                            if parent_name == "Path" and grandparent_name == "get": 
                                underlying = underlying.strip()

                        # exception: LocationKeyphrases are always str                        
                        if name == "String" and type(underlying) == int: 
                            underlying =  " ".join(child_names) 

                    inner_dict = {"schema": name, "underlying": underlying}
                    op = ValueOp(value=json.dumps(inner_dict))

                    if graph.nodes[node]['node_prob'] is not None:
                        # construct prob taking into account all children
                        op_prob = graph.nodes[node]['node_prob']
                        for child in children: 
                            op_prob *= graph.nodes[child]['node_prob']

                elif graph.nodes[node]['function_type'] == "call": 
                    name = graph.nodes[node]['node_name']
                    op = CallLikeOp(name=name)

                    if graph.nodes[node]['node_prob'] is not None:
                        op_prob = graph.nodes[node]['node_prob']
            else:
                continue

            if op is not None:
                fxn_children = get_fxn_children(node) 
                eid = node_id_to_expr_id[node]
                expr_ids = [node_id_to_expr_id[n] for n in fxn_children]
                # add check for lambda
                if isinstance(op, CallLikeOp) and op.name == "lambda" and len(expr_ids) == 1:
                    expr_ids = [expr_ids] + [expr_ids]
                    children_ids = [f"{e}" for e in expr_ids]
                    is_lambda = True
                # elif isinstance(op, CallLikeOp) and op.name == "lambda" and len(expr_ids) == 2:
                    # expr_ids[1] -= 1
                    # add_to_next.append(expr_ids[0])
                    # children_ids = [f"[{e}]" for e in expr_ids]
                    # is_lambda = True
                else:
                    # if len(add_to_next) > 0 and hasattr(op, "name") and op.name == "&":
                        # expr_ids = add_to_next + expr_ids
                        # add_to_next = []
                    children_ids = [f"[{e}]" for e in expr_ids]
                type_args = get_type_args(node)

                curr_expr = Expression(id=f"[{eid}]", op = op, arg_ids = children_ids, type_args = type_args)
                expressions.append(curr_expr)
                expression_probs.append(op_prob)

        if is_lambda: 
            # decrease all expression ids by 1 
            for i, expr in enumerate(expressions):
                expr_id = int(re.sub("[\[\]]", "", expr.id))
                child_ids = [int(re.sub("[\[\]]", "", child_id)) for child_id in expr.arg_ids]
                expr_id -= 1 
                child_ids = [child_id - 1 for child_id in child_ids]

                new_expr = Expression(id=f"[{expr_id}]", op = expr.op, arg_ids = [f"[{child_id}]" for child_id in child_ids], type_args = expr.type_args)
                expressions[i] = new_expr
                expression_probs[i] = op_prob

        expressions.reverse()
        expression_probs.reverse()
        return Program(expressions), expressions, expression_probs

    @staticmethod
    def lists_to_ast(node_name_list: List[str], 
                     node_idx_list: List[str],
                     edge_head_list: List[int], 
                     edge_type_list: List[int],
                     node_probs_list: List[float]) -> List[Any]:
        """
        convert predicted lists back to an AST 
        """
        # use digraph to store data and then convert 
        graph = nx.DiGraph() 

        # start with 1-to-1 mapping 
        node_idx_to_list_idx_mapping = {k:k for k in range(len(node_name_list))}

        def update_mapping_after_n(n):
            for k in node_idx_to_list_idx_mapping.keys():
                if k > n:
                    node_idx_to_list_idx_mapping[k] -= 1

        offset = 0
        for i, (node_name, node_idx, edge_head, edge_type, node_prob) in enumerate(zip(node_name_list, node_idx_list, edge_head_list, edge_type_list, node_probs_list)):
            if edge_type.startswith("fxn_arg"):
                function_type = "build"
            elif edge_type.startswith("val_arg"):
                function_type = "value"
            elif edge_type.startswith("call_arg"):
                function_type = "call"
            else:
                function_type = None 

            reentrant = False
            if i - offset != node_idx:
                # reentrant 
                curr_name = graph.nodes[node_idx]['node_name']
                # check we're not renaming anything here 
                try:
                    assert(curr_name == node_name)
                except AssertionError:
                    pass
                offset += 1
                update_mapping_after_n(node_idx) 

            graph.add_node(node_idx, node_name = node_name, function_type= function_type, node_prob=node_prob)

            # root self-edges
            if edge_head < 0:
                edge_head = 0

            edge_head_idx = node_idx_list[edge_head] 
            graph.add_edge(edge_head_idx, node_idx, type=edge_type)

        return graph 
