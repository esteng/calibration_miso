from calibration_sql_type_reader import TypeTopLogitFormatSequenceReader
import pdb 

path = "/home/estengel/semantic_parsing_with_constrained_lm/src/semantic_parsing_with_constrained_lm/trained_models/1.0/t5-small-lm-adapt_spider_past_none_db_val_all_lower_0.0001/checkpoint-5000/outputs/test_all.logits"
model = "t5-small"

reader = TypeTopLogitFormatSequenceReader(path,
                                        model_name=model)
preds_by_type, is_correct_by_type = reader.read()
pdb.set_trace()