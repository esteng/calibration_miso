import pytest 
import pdb 
import difflib 

from test_calflow_data import calflow_roundtrip
from dataflow.core.lispress import parse_lispress, program_to_lispress, lispress_to_program, render_compact, render_pretty
from miso.data.dataset_readers.calflow_parsing.calflow_graph import CalFlowGraph
from miso.data.dataset_readers.calflow_parsing.tree_dst_graph import TreeDSTGraph


def tree_dst_roundtrip(test_str):
    true_ls_prog, __ = lispress_to_program(parse_lispress(test_str),0)
    # print(true_ls_prog)
    true_lispress = program_to_lispress(true_ls_prog)

    calflow_graph = TreeDSTGraph(src_str="", tgt_str = test_str)
    program = TreeDSTGraph.prediction_to_program(calflow_graph.node_name_list, 
                                                calflow_graph.node_idx_list, 
                                                calflow_graph.edge_head_list, 
                                                calflow_graph.edge_type_list)

    pred_lispress = program_to_lispress(program)
    true_ls_prog, __ = lispress_to_program(parse_lispress(test_str),0)
    true_lispress = program_to_lispress(true_ls_prog)
    true_lispress_str = render_pretty(true_lispress)
    pred_lispress_str = render_pretty(pred_lispress)
    print([(i, x[0], x[1], x[2]) for i, x in enumerate(zip(calflow_graph.node_name_list, calflow_graph.edge_head_list, calflow_graph.edge_type_list))])
    pdb.set_trace()
    assert(pred_lispress_str == true_lispress_str)

@pytest.fixture
def load_tree_dst_lispress():
    return """( plan ( ^ ( Hotel ) Find :focus ( Hotel.location_? ( ^ ( String ) always ) ) :object ( Hotel.hotelName_? ( ?= " Moody Moon " ) ) ) )"""

@pytest.fixture
def load_lambda_lispress():
    return """( plan ( revise ( ^ ( Unit ) Path.apply " Book " ) ( ^ ( ( Constraint Taxi ) ) Path.apply " object " ) ( lambda ( ^ ( Constraint Taxi ) x0 ) ( & x0 ( Taxi.serviceType_? ( List.exists_? ( ?= ( ServiceType.Executive ) ) ) ) ) ) ) )"""

@pytest.fixture
def load_phone_num_lispress():
    return """( plan ( ^ ( Message ) Create :object ( & ( Message.recipients_? ( Contact.phoneNumber_? ( ?= " 07764716039 " ) ) ) ( Message.textContent_? ( ?= " what\'s up " ) ) ) ) )"""

@pytest.fixture
def load_broken_lispress():
    return """( plan ( revise ( ^ ( Unit ) Path.apply " Create " ) ( ^ ( Unit ) Path.apply "  " ) ( lambda ( ^ Unit x0 ) x0 ) ) )"""

@pytest.fixture
def load_two_lambdas_lispress():
    return """( plan ( revise ( ^ ( Unit ) Path.apply " Book " ) ( ^ ( ( Constraint Taxi ) ) Path.apply " object " ) ( lambda ( ^ ( Constraint Taxi ) x0 ) ( & x0 ( Taxi.serviceType_? ( List.exists_? ( ?= ( ServiceType.Executive ) ) ) ) ) ) ) )"""

@pytest.fixture
def load_all_valid_tgt_str():
    data_path = "/brtx/601-nvme1/estengel/resources/data/tree_dst.agent.data/valid.tgt"
    with open(data_path) as f1:
       lines = f1.readlines() 
    return lines


def test_lambda_roundtrip(load_lambda_lispress):
    tree_dst_roundtrip(load_lambda_lispress) 

def test_tree_dst_roundtrip(load_tree_dst_lispress):
    tree_dst_roundtrip(load_tree_dst_lispress)

def test_phone_num_roundtrip(load_phone_num_lispress):
    tree_dst_roundtrip(load_phone_num_lispress)

def test_broken_roundtrip(load_broken_lispress):
    tree_dst_roundtrip(load_broken_lispress)

def test_two_lambdas_roundtrip(load_two_lambdas_lispress):
    tree_dst_roundtrip(load_two_lambdas_lispress)

def test_calflow_roundtrip_valid(load_all_valid_tgt_str):
    all_lines = load_all_valid_tgt_str
    skipped = 0
    mistakes = 0
    for i, line in enumerate(all_lines):
        try:
            line=line.strip()
            lispress_from_line = parse_lispress(line)
            clean_true_lispress_str = render_compact(lispress_from_line) 
            calflow_graph = TreeDSTGraph(src_str="", tgt_str = clean_true_lispress_str)
            program = TreeDSTGraph.prediction_to_program(calflow_graph.node_name_list, 
                                                        calflow_graph.node_idx_list,
                                                        calflow_graph.edge_head_list, 
                                                        calflow_graph.edge_type_list) 
            pred_lispress = program_to_lispress(program)
            pred_lispress = program_to_lispress(lispress_to_program(pred_lispress, 0)[0])
            # run through again to get sugaring
            true_ls_prog, __ = lispress_to_program(lispress_from_line,0)
            true_lispress = program_to_lispress(true_ls_prog)
            true_lispress_str = render_pretty(true_lispress)
            pred_lispress_str = render_pretty(pred_lispress)
        except AssertionError:
            # pdb.set_trace()
            mistakes += 1
        try:
            assert(pred_lispress_str.strip() == true_lispress_str.strip())

        except (AssertionError) as e:
            progress = i/len(all_lines)
            print(progress)
            print(f"pred: {pred_lispress_str}")
            print(f"true: {true_lispress_str}")
            print("diff")
            # pdb.set_trace()
            print(" ".join(difflib.ndiff(pred_lispress_str.splitlines(keepends=True), true_lispress_str.splitlines(keepends=True))))
            mistakes += 1
    print(f"there were  {mistakes} mistakes out of {len(all_lines)} lines")
    assert(1==2)