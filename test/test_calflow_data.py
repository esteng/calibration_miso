import pytest
import sys 
import os 
import pdb
import traceback

from allennlp.data.token_indexers.token_indexer import TokenIndexer

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, path) 

#from task_oriented_dialogue_as_dataflow_synthesis.src.dataflow.core.dialogue import Dialogue
from dataflow.core.lispress import parse_lispress, program_to_lispress, lispress_to_program, render_compact, render_pretty, _sugar_gets, _desugar_gets
from dataflow.core.dialogue import Dialogue
from dataflow.core.io import load_jsonl_file
from miso.data.dataset_readers.calflow_parsing.calflow_graph import CalFlowGraph
from miso.data.dataset_readers.calflow_parsing.calflow_sequence import CalFlowSequence
from miso.data.dataset_readers.calflow import CalFlowDatasetReader
from miso.data.tokenizers import MisoTokenizer

def assert_dict(produced, expected):
    for key in expected:
        assert(produced[key] == expected[key])

@pytest.fixture
def load_test_lispress():
    return """( Yield ( PersonFromRecipient ( Execute ( refer ( extensionConstraint ( RecipientWithNameLike ( ^ ( Recipient ) EmptyStructConstraint ) ( PersonName.apply "Darby" ) ) ) ) ) ) )"""

@pytest.fixture
def load_long_lispress():
    return """( Yield ( UpdateCommitEventWrapper ( UpdatePreflightEventWrapper ( Event.id ( singleton ( QueryEventResponse.results ( FindEventWrapperWithDefaults ( Event.attendees? ( AttendeeListHasRecipientConstraint ( RecipientWithNameLike ( ( ^ ( Recipient ) EmptyStructConstraint ) ) ( PersonName.apply "Matthew" ) ) ) ) ) ) ) ) ( Event.start? ( ?= ( DateAtTimeWithDefaults ( NextDOW ( Thursday ) ) ( DateTime.time ( Event.end ( singleton ( QueryEventResponse.results ( FindEventWrapperWithDefaults ( Event.attendees? ( AttendeeListHasRecipientConstraint ( RecipientWithNameLike ( ( ^ ( Recipient ) EmptyStructConstraint ) ) ( PersonName.apply "Jeremy" ) ) ) ) ) ) ) ) ) ) ) ) ) ) )"""

@pytest.fixture
def load_let_lispress():
    return """( let ( x0 ( Execute ( refer ( extensionConstraint ( ^ ( Event ) EmptyStructConstraint ) ) ) ) ) ( Yield ( CreateCommitEventWrapper ( CreatePreflightEventWrapper ( Event.start? ( OnDateBeforeTime ( DateTime.date ( Event.start x0 ) ) ( DateTime.time ( Event.start x0 ) ) ) ) ) ) ) )"""

@pytest.fixture
def load_path_lispress():
    return """( Yield ( Execute ( ReviseConstraint ( refer ( ( ^ ( Dynamic ) roleConstraint ) ( Path.apply "output" ) ) ) ( ( ^ ( Event ) ConstraintTypeIntension ) ) ( Event.attendees? ( AttendeeListHasRecipient ( Execute ( refer ( extensionConstraint ( RecipientWithNameLike ( ( ^ ( Recipient ) EmptyStructConstraint ) ) ( PersonName.apply "Pam" ) ) ) ) ) ) ) ) ) )"""

@pytest.fixture
def load_do_singleton_lispress():
    return """(do (singleton (QueryEventResponse.results (FindEventWrapperWithDefaults (& (& (Event.subject? (?~= "lunchdate")) (Event.start? (DateTime.date? (?= (NextDOW (Thursday)))))) (Event.attendees? (AttendeeListHasRecipientConstraint (RecipientWithNameLike ((^(Recipient) EmptyStructConstraint)) (PersonName.apply "Lisa")))))))) (Yield (UpdateCommitEventWrapper (UpdatePreflightEventWrapper (Event.id (Execute (refer (extensionConstraint ((^(Event) EmptyStructConstraint)))))) (Event.start? (DateTime.date? (?= (ClosestDayOfWeek (DateTime.date (Event.start (Execute (refer (extensionConstraint ((^(Event) EmptyStructConstraint))))))) (Friday)))))))))"""

@pytest.fixture
def load_underlying_lispress():
    return """( Yield ( Execute ( NewClobber ( refer ( ^ ( Dynamic ) ActionIntensionConstraint ) ) ( ^ ( ( Constraint DateTime ) ) roleConstraint ( Path.apply "time" ) ) ( intension ( DateTime.date? ( ?= ( Tomorrow ) ) ) ) ) ) )"""

@pytest.fixture
def load_variable_order_lispress():
    return """( let ( x0 ( PersonName.apply "Elli Parker" ) ) ( do ( Yield ( Execute ( ChooseCreateEventFromConstraint ( ^ ( Event ) EmptyStructConstraint ) ( refer ( ^ ( Dynamic ) ActionIntensionConstraint ) ) ) ) ) ( Yield ( > ( size ( QueryEventResponse.results ( FindEventWrapperWithDefaults ( EventOnDate ( Tomorrow ) ( Event.attendees? ( & ( AttendeeListHasRecipientConstraint ( RecipientWithNameLike ( ^ ( Recipient ) EmptyStructConstraint ) x0 ) ) ( AttendeeListHasPeople ( FindTeamOf ( Execute ( refer ( extensionConstraint ( RecipientWithNameLike ( ^ ( Recipient ) EmptyStructConstraint ) x0 ) ) ) ) ) ) ) ) ) ) ) ) 0L ) ) ) )"""

@pytest.fixture
def load_reentrant_expression_lispress():
    return """( let ( x0 ( NextDOW ( Friday ) ) ) ( Yield ( CreateCommitEventWrapper ( CreatePreflightEventWrapper ( EventAllDayOnDate ( EventAllDayOnDate ( Event.subject? ( ?= "spending" ) ) x0 ) ( nextDayOfWeek x0 ( Sunday ) ) ) ) ) ) )"""

@pytest.fixture
def load_inf_loss_lispress():
    return """( let ( x0 ( DateAtTimeWithDefaults ( Execute ( refer ( extensionConstraint ( ^ ( Date ) EmptyStructConstraint ) ) ) ) ( Noon ) ) ) ( do "Shiro's sushi" ( Yield ( CreateCommitEventWrapper ( CreatePreflightEventWrapper ( & ( & ( & ( & ( & ( Event.subject_? ( ?= "lunch date" ) ) ( Event.start_? ( ?= x0 ) ) ) ( Event.end_? ( ?= ( TimeAfterDateTime x0 ( NumberPM 2L ) ) ) ) ) ( Event.location_? ( ?= ( LocationKeyphrase.apply "Shiro's sushi" ) ) ) ) ( Event.showAs_? ( ?= ( ShowAsStatus.OutOfOffice ) ) ) ) ( Event.attendees_? ( AttendeeListHasRecipient ( Execute ( refer ( extensionConstraint ( RecipientWithNameLike ( ^ ( Recipient ) EmptyStructConstraint ) ( PersonName.apply "Kate" ) ) ) ) ) ) ) ) ) ) ) ) )"""

@pytest.fixture
def load_all_valid_tgt_str():
    data_path = os.path.join(path, "data", "smcalflow.full.data", "valid.tgt") 
    with open(data_path) as f1:
       lines = f1.readlines() 
    return lines 

@pytest.fixture
def load_all_train_tgt_str():
    data_path = os.path.join(path, "data", "smcalflow.full.data", "train.tgt") 
    with open(data_path) as f1:
       lines = f1.readlines() 
    return lines 

def test_tgt_str_to_list_short(load_test_lispress_short):
    calflow_graph = CalFlowGraph(src_str="", tgt_str = load_test_lispress_short)
    assert(calflow_graph.node_name_list == ['Yield', 'output', 'Execute', 'intension', 'ConfirmAndReturnAction'])
    assert(calflow_graph.node_idx_list == [0, 1, 2, 3, 4])
    assert(calflow_graph.edge_head_list == [0, 0, 1, 2, 3])
    assert(calflow_graph.edge_type_list == ["fxn_arg-0", "arg-0", "fxn_arg-0", "arg-0", "fxn_arg-0"])

def calflow_roundtrip(test_str):
    try:
        true_ls_prog, __ = lispress_to_program(parse_lispress(test_str),0)
        true_lispress = program_to_lispress(true_ls_prog)

        calflow_graph = CalFlowGraph(src_str="", tgt_str = test_str)
        program = CalFlowGraph.prediction_to_program(calflow_graph.node_name_list, 
                                                    calflow_graph.node_idx_list, 
                                                    calflow_graph.edge_head_list, 
                                                    calflow_graph.edge_type_list)

        pred_lispress = program_to_lispress(program)
        true_ls_prog, __ = lispress_to_program(parse_lispress(test_str),0)
        true_lispress = program_to_lispress(true_ls_prog)
        true_lispress_str = render_pretty(true_lispress)
        pred_lispress_str = render_pretty(pred_lispress)
        assert(pred_lispress_str == true_lispress_str)
    except (AssertionError,KeyError,IndexError, AttributeError, TypeError) as error:
        pytest.set_trace() 
        traceback.print_exc()
        print(error)

def test_calflow_roundtrip_base(load_test_lispress):
    calflow_roundtrip(load_test_lispress)
    
def test_calflow_roundtrip_long(load_long_lispress):
    calflow_roundtrip(load_long_lispress)

def test_calflow_roundtrip_path(load_path_lispress):
    calflow_roundtrip(load_path_lispress)

def test_calflow_roundtrip_singleton(load_do_singleton_lispress):
    calflow_roundtrip(load_do_singleton_lispress)

def test_calflow_roundtrip_nested_underlying(load_underlying_lispress):
    calflow_roundtrip(load_underlying_lispress)

def test_calflow_roundtrip_let(load_let_lispress):
    calflow_roundtrip(load_let_lispress) 

def test_calflow_roundtrip_variable_order(load_variable_order_lispress):
    calflow_roundtrip(load_variable_order_lispress)

def test_calflow_roundtrip_expression_order(load_reentrant_expression_lispress):
    calflow_roundtrip(load_reentrant_expression_lispress)

def test_calflow_rountrip_inf_issue(load_inf_loss_lispress):
    calflow_roundtrip(load_inf_loss_lispress)

def test_calflow_roundtrip_valid(load_all_valid_tgt_str):
    all_lines = load_all_valid_tgt_str
    skipped = 0
    mistakes = 0
    for i, line in enumerate(all_lines):
        try:
            line=line.strip()
            lispress_from_line = parse_lispress(line)
            clean_true_lispress_str = render_compact(lispress_from_line) 
            calflow_graph = CalFlowGraph(src_str="", tgt_str = clean_true_lispress_str)
            program = CalFlowGraph.prediction_to_program(calflow_graph.node_name_list, 
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

            assert(pred_lispress_str == true_lispress_str)

        except (AssertionError, IndexError, KeyError) as e:
            progress = i/len(all_lines)
            print(progress)
            print(pred_lispress_str)
            print(true_lispress_str)
            pdb.set_trace() 
            mistakes += 1

def test_calflow_roundtrip_train(load_all_train_tgt_str):
    all_lines = load_all_train_tgt_str
    skipped = 0
    mistakes = 0
    for i, line in enumerate(all_lines):
        try:
            line=line.strip()
            lispress_from_line = parse_lispress(line)
            clean_true_lispress_str = render_compact(lispress_from_line) 
            calflow_graph = CalFlowGraph(src_str="", tgt_str = clean_true_lispress_str)
            program = CalFlowGraph.prediction_to_program(calflow_graph.node_name_list, 
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

            assert(pred_lispress_str == true_lispress_str)

        except (AssertionError, IndexError, KeyError) as e:
            progress = i/len(all_lines)
            print(progress)
            print(pred_lispress_str)
            print(true_lispress_str)
            pdb.set_trace() 
            mistakes += 1


@pytest.fixture
def load_seq_strings_basic():
    #src_str = """__User Darby __StartOfProgram ( Yield ( PersonFromRecipient ( Execute ( refer ( extensionConstraint ( RecipientWithNameLike ( ( ^ ( Recipient ) EmptyStructConstraint ) ) ( PersonName.apply "Darby" ) ) ) ) ) ) ) __User Dirty Dan __StartOfProgram"""
    src_str = """__User Darby __StartOfProgram"""
    tgt_str = """( Yield ( PersonFromRecipient ( Execute ( refer ( extensionConstraint ( RecipientWithNameLike ( ( ^ ( Recipient ) EmptyStructConstraint ) ) ( PersonName.apply "Darby" ) ) ) ) ) ) )"""
    return (src_str, tgt_str)

def test_calflow_get_list_data(load_seq_strings_basic):
    src_str, tgt_str = load_seq_strings_basic
    g = CalFlowGraph(src_str = src_str,
                     tgt_str = tgt_str)
    data = g.get_list_data(bos="@start@",
                           eos="@end@")



def test_calflow_sequence_basic(load_seq_strings_basic):
    src_str, tgt_str = load_seq_strings_basic
    seq_obj = CalFlowSequence(src_str, tgt_str, use_program=False)
    output = seq_obj.get_list_data(bos="@start@", eos="@end@")

    assert(output['src_copy_inds'] == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1])
    assert(output['tgt_tokens'][output['src_copy_inds'].index(2)] == "Darby")














































































































































































#def test_get_list_data(load_tiny_data):
#    # test concat-after 
#    d_graph = CalFlowGraph(load_dev_graphs['basic'], syntactic_method="concat-after")
#    list_data = d_graph.get_list_data(bos="@start@", eos="@end@", max_tgt_length = 100) 
#    expected = {"tgt_tokens": ["@start@", "@@ROOT@@", "comes", "AP", "the", "story", "this", "From",  '@syntax-sep@', 'comes', 'AP', 'story', ':', 'From', 'the', 'this', '@end@'],
#                "head_tags": ['dependency', 'dependency', 'dependency', 'EMPTY', 'dependency', 'EMPTY', 'EMPTY', 'SEP', 'root', 'nmod', 'nsubj', 'punct', 'case', 'det', 'det'],
#                "head_indices": [0, 1, 2, 3, 2, 5, 2, -1, 0, 9, 9, 9, 10, 10, 11]}
#
#    assert_dict(list_data, expected) 
#
#def test_get_list_data_concat_before(load_dev_graphs):
#    # test concat-after 
#    d_graph = DecompGraphWithSyntax(load_dev_graphs['basic'], syntactic_method="concat-before")
#    list_data = d_graph.get_list_data(bos="@start@", eos="@end@", max_tgt_length = 100) 
#    expected = {"tgt_tokens": ['@start@', 'comes', 'AP', 'story', ':', 'From', 'the', 'this', '@syntax-sep@', '@@ROOT@@', 'comes', 'AP', 'the', 'story', 'this', 'From', '@end@'],
#                "head_indices": [0, 1, 1, 1, 2, 2, 3, -1, 0, 9, 10, 11, 10, 13, 10],
#                "head_tags": ['root', 'nmod', 'nsubj', 'punct', 'case', 'det', 'det', 'SEP', 'dependency', 'dependency', 'dependency', 'EMPTY', 'dependency', 'EMPTY', 'EMPTY']} 
#
#    assert_dict(list_data, expected) 
#
#def test_get_list_data_syntax_basic(load_dev_graphs): 
#    # test 1: basic  
#    d_graph = DecompGraph(load_dev_graphs["basic"]) 
#    list_data = d_graph.get_list_data(bos="@start@", eos="@end@")
#    expected = {"tgt_tokens": ["@start@", "@@ROOT@@", "comes", "AP", "the", "story", "this", "From", "@end@"],
#                "tgt_indices": [0, 1, 2, 3, 4, 5, 6, 7, 8],
#                "tgt_attributes": [{}, {}, {'factuality-factual': {'confidence': 1.0, 'value': 0.967}, 'time-dur-weeks': {'confidence': 0.2564, 'value': -1.3247}, 'time-dur-decades': {'confidence': 0.2564, 'value': -1.1146}, 'time-dur-days': {'confidence': 0.2564, 'value': 0.8558}, 'time-dur-hours': {'confidence': 0.2564, 'value': 0.9952}, 'time-dur-seconds': {'confidence': 0.2564, 'value': 0.8931}, 'time-dur-forever': {'confidence': 0.2564, 'value': -1.4626}, 'time-dur-centuries': {'confidence': 0.2564, 'value': -1.1688}, 'time-dur-instant': {'confidence': 0.2564, 'value': -1.4106}, 'time-dur-years': {'confidence': 0.2564, 'value': 0.9252}, 'time-dur-minutes': {'confidence': 0.2564, 'value': -0.9337}, 'time-dur-months': {'confidence': 0.2564, 'value': -1.2142}, 'genericity-pred-dynamic': {'confidence': 0.627, 'value': -0.0469}, 'genericity-pred-hypothetical': {'confidence': 0.5067, 'value': -0.0416}, 'genericity-pred-particular': {'confidence': 1.0, 'value': 1.1753}}, {'genericity-arg-kind': {'confidence': 1.0, 'value': -1.1642}, 'genericity-arg-abstract': {'confidence': 1.0, 'value': -1.1642}, 'genericity-arg-particular': {'confidence': 1.0, 'value': 1.2257}}, {}, {'wordsense-supersense-noun.object': {'confidence': 1.0, 'value': -3.0}, 'wordsense-supersense-noun.Tops': {'confidence': 1.0, 'value': -3.0}, 'wordsense-supersense-noun.quantity': {'confidence': 1.0, 'value': -3.0}, 'wordsense-supersense-noun.feeling': {'confidence': 1.0, 'value': -3.0}, 'wordsense-supersense-noun.food': {'confidence': 1.0, 'value': -3.0}, 'wordsense-supersense-noun.shape': {'confidence': 1.0, 'value': -3.0}, 'wordsense-supersense-noun.event': {'confidence': 1.0, 'value': -3.0}, 'wordsense-supersense-noun.motive': {'confidence': 1.0, 'value': -3.0}, 'wordsense-supersense-noun.substance': {'confidence': 1.0, 'value': -3.0}, 'wordsense-supersense-noun.time': {'confidence': 1.0, 'value': -3.0}, 'wordsense-supersense-noun.person': {'confidence': 1.0, 'value': -3.0}, 'wordsense-supersense-noun.process': {'confidence': 1.0, 'value': -3.0}, 'wordsense-supersense-noun.attribute': {'confidence': 1.0, 'value': -3.0}, 'wordsense-supersense-noun.artifact': {'confidence': 1.0, 'value': -1.3996}, 'wordsense-supersense-noun.group': {'confidence': 1.0, 'value': -3.0}, 'wordsense-supersense-noun.animal': {'confidence': 1.0, 'value': -3.0}, 'wordsense-supersense-noun.location': {'confidence': 1.0, 'value': -3.0}, 'wordsense-supersense-noun.plant': {'confidence': 1.0, 'value': -3.0}, 'wordsense-supersense-noun.possession': {'confidence': 1.0, 'value': -3.0}, 'wordsense-supersense-noun.relation': {'confidence': 1.0, 'value': -3.0}, 'wordsense-supersense-noun.phenomenon': {'confidence': 1.0, 'value': -3.0}, 'wordsense-supersense-noun.cognition': {'confidence': 1.0, 'value': -3.0}, 'wordsense-supersense-noun.act': {'confidence': 1.0, 'value': -3.0}, 'wordsense-supersense-noun.state': {'confidence': 1.0, 'value': -3.0}, 'wordsense-supersense-noun.communication': {'confidence': 1.0, 'value': 0.2016}, 'wordsense-supersense-noun.body': {'confidence': 1.0, 'value': -3.0}, 'genericity-arg-kind': {'confidence': 0.7138, 'value': -0.035}, 'genericity-arg-abstract': {'confidence': 1.0, 'value': -1.1685}, 'genericity-arg-particular': {'confidence': 1.0, 'value': 1.2257}}, {}, {}, {}],
#            "tgt_copy_indices": [0, 0, 0, 0, 0, 0, 0, 0, 0],
#            "tgt_tokens_to_generate": ["@start@", "@@ROOT@@", "@@UNKNOWN@@", "@@UNKNOWN@@", "@@UNKNOWN@@", "@@UNKNOWN@@", "@@UNKNOWN@@", "@@UNKNOWN@@", "@end@"]
#            }
#    assert_dict(list_data, expected) 
#
#def test_get_list_data_syntax_reentrancy(load_dev_graphs): 
#    # test 2: corefferent/reentrant node 
#    d_graph = DecompGraph(load_dev_graphs["coref"])
#    list_data = d_graph.get_list_data(bos="@start@", eos="@end@")
#    expected = {"tgt_tokens": ["@start@", "@@ROOT@@", "nominated", "Bush", 
#                                "President", "Tuesday", "individuals", "two", 
#                                "on", "replace", "Bush", "President", "jurists", 
#                                "retiring", "on", "federal", "courts", "in", "the", 
#                                "Washington", "area", "to", "@end@"], 
#                "tgt_indices": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 3, 4, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
#                "tgt_tokens_to_generate": ["@start@", "@@ROOT@@", "@@UNKNOWN@@", "@@UNKNOWN@@", "@@UNKNOWN@@", 
#                                            "@@UNKNOWN@@", "@@UNKNOWN@@", "@@UNKNOWN@@", "@@UNKNOWN@@", "@@UNKNOWN@@", 
#                                            "@@UNKNOWN@@", "@@UNKNOWN@@", "@@UNKNOWN@@", "@@UNKNOWN@@", "@@UNKNOWN@@", 
#                                            "@@UNKNOWN@@", "@@UNKNOWN@@", "@@UNKNOWN@@", "@@UNKNOWN@@", "@@UNKNOWN@@", 
#                                            "@@UNKNOWN@@", "@@UNKNOWN@@", "@end@"],
#                "head_tags": ["dependency", "dependency", "dependency", "EMPTY", "dependency", "dependency", "EMPTY", 
#                              "EMPTY", "dependency", "dependency", "EMPTY", "dependency", "EMPTY", "EMPTY", "EMPTY", 
#                              "EMPTY", "EMPTY", "EMPTY", "EMPTY", "EMPTY", "EMPTY"], 
#                "head_indices": [0, 1, 2, 3, 2, 2, 6, 2, 1, 9, 10, 9, 12, 12, 12, 12, 12, 12, 12, 12, 9],
#                "node_name_list": ["@start@", "dummy-semantics-root", "ewt-dev-2-semantics-pred-5", "ewt-dev-2-semantics-arg-2", 
#                                  "ewt-dev-2-syntax-1", "ewt-dev-2-semantics-arg-4", "ewt-dev-2-semantics-arg-7", 
#                                  "ewt-dev-2-syntax-6", "ewt-dev-2-syntax-3", "ewt-dev-2-semantics-pred-9", 
#                                  "ewt-dev-2-semantics-arg-2", "ewt-dev-2-syntax-1", "ewt-dev-2-semantics-arg-11", 
#                                  "ewt-dev-2-syntax-10", "ewt-dev-2-syntax-12", "ewt-dev-2-syntax-13", 
#                                  "ewt-dev-2-syntax-14", "ewt-dev-2-syntax-15", "ewt-dev-2-syntax-16", 
#                                  "ewt-dev-2-syntax-17", "ewt-dev-2-syntax-18", "ewt-dev-2-syntax-8", "@end@"]
#                }
#
#    assert_dict(list_data, expected) 
#
##def test_get_list_data_syntax_long(load_dev_graphs): 
##    # test 3: long data
##    d_graph = DecompGraph(load_dev_graphs["long"])
##    list_data = d_graph.get_list_data(bos="@start@", eos="@end@")
##    
##    print(list_data) 
##    assert(2==1) 
#
#

@pytest.fixture
def load_indexers():
    source_params = {"source_token_characters": {
                        "type": "characters",
                        "min_padding_length": 5,
                        "namespace": "source_token_characters"
                    },
                    "source_tokens": {
                        "type": "single_id",
                        "namespace": "source_tokens"
                        }
                    }
    target_params = {"target_token_characters": {
                        "type": "characters",
                        "min_padding_length": 5,
                        "namespace": "target_token_characters"
                    },
                    "target_tokens": {
                        "type": "single_id",
                        "namespace": "target_tokens"
                        }
                    }

    source_token_indexers = {k: TokenIndexer(**params) for k, params in source_params.items()}
    target_token_indexers = {k: TokenIndexer(**params) for k, params in target_params.items()}

    return source_token_indexers, target_token_indexers


def test_calflow_dataset_reader(load_indexers):
    source_token_indexers, target_token_indexers = load_indexers
    data_path = os.path.join(path, "data", "smcalflow.full.data", "tiny.dataflow_dialogues.jsonl")
    generation_token_indexers = target_token_indexers
    tokenizer = MisoTokenizer()
    evaluation = False 
    line_limit = None
    lazy = False

    dataset_reader = CalFlowDatasetReader(source_token_indexers,
                                          target_token_indexers,
                                          generation_token_indexers,
                                          tokenizer,
                                          evaluation,
                                          line_limit,
                                          lazy)

    data = dataset_reader._read(data_path)

    assert(data)
