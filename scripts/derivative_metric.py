import pathlib 
import json 
import pdb 

import pandas as pd 
import numpy as np 


def get_intent_data_from_dir(root_data_dir, fxn, seeds):
    all_data = pd.DataFrame(columns=["train", "function", "seed", "total_acc", "fxn_acc"], dtype=object)
    root_data_dir = pathlib.Path(root_data_dir).joinpath(str(fxn))
    for seed in seeds:
        data_dir = root_data_dir.joinpath(f"{seed}_seed")

        globs = [x for x in data_dir.glob("*/test_metrics.json")]
        globs = sorted(globs, key = lambda x: int(x.parent.name.split("_")[0]))

        for path in globs:
            try:
                data = json.load(open(path))
            except json.JSONDecodeError:
                data = {}
                data['acc'] = np.nan
                data[f'{fxn}_acc'] = np.nan

            setting = path.parent.name
            num_train, num_fxn = setting.split("_")
            num_train, num_fxn = int(num_train), int(num_fxn)

            to_add = {"train": str(num_train), "function": num_fxn, "seed": seed, 
                     "total_acc": data['acc'], "fxn_acc": data[f"{fxn}_acc"]}
            all_data = all_data.append(to_add, ignore_index=True)

    return all_data 


def prepare_intent(df, fxn_num): 
    # drop total acc
    df = df.drop(columns="total_acc")
    # rename column 
    df = df.rename(columns={"fxn_acc":"acc"})
    df = df[df['function'] == fxn_num]
    # multiply accuracy by factor so that it is roughly the same size as train 
    df['train'] = df['train'].astype(int)
    avg_train = df.mean()['train']
    avg_acc = df.mean()['acc']
    ratio = avg_train / avg_acc
    df['acc'] *= ratio
    return df 


def detect_missing(fxn, df, seeds=[12,31,64]):
    sub_dfs = []
    for seed in seeds:
        sub_df = df[df['seed'] == seed]
        sub_dfs.append(sub_df)

    for i,sdf_a in enumerate(sub_dfs):
        for j, sdf_b in enumerate(sub_dfs):
            if i == j:
                continue

            for train_setting in sdf_a['train']:
                b_vals = sdf_b[sdf_b['train'] == train_setting]
                if len(b_vals) == 0:
                    seed = sdf_a[sdf_a['train'] == train_setting].loc['seed']
                    seed = sdf_a[sdf_a['train'] == train_setting].loc['fxn']
                    print(f"{fxn} is missing {train_setting} for seed {seed}")
    
def compute_derivative_metric(df, average_first=False):
    # Assumed datatframe format: 
    # columns: "train", "function", "seed", "accuracy"
    # if average_first=True, then first take average across seeds, then take slope
    # otherwise take the average of slopes 
    timesteps = sorted(list(set([int(x) for x in df['train']])))
    slope_df = pd.DataFrame(columns=["end_train", "seed", "slope"], dtype=object)
    for i in range(len(timesteps)-1):
        start_ts = timesteps[i]
        end_ts = timesteps[i+1]

        start_rows = df[df['train'] == start_ts]
        end_rows = df[df['train'] == end_ts]

        # if average_first:
        #     start_acc = start_rows['acc'].mean()
        #     end_acc = end_rows['acc'].mean()
        #     slope = (end_acc - start_acc)/(end_ts - start_ts)
        # else:
        all_slopes = []
        for ((start_index, start_row), (end_index, end_row)) in zip(start_rows.iterrows(), end_rows.iterrows()):
            seed = start_row['seed']
            try:
                assert(start_row['seed'] == end_row['seed'])
                assert(int(start_row['train']) < int(end_row['train']))
                assert(start_row['function'] == end_row['function'])
            except AssertionError:
                pdb.set_trace() 
            start_acc = start_row['acc']
            end_acc = end_row['acc']
            single_slope = (end_acc - start_acc)/(end_ts - start_ts)
            all_slopes.append(single_slope)
            slope_df = slope_df.append({"end_train": end_ts, "seed": seed, "slope": single_slope}, ignore_index=True)
        #slope = np.mean(all_slopes)

    # pdb.set_trace()  
    sum_df = slope_df.set_index("seed").sum(level="seed")

    mean = sum_df.mean()['slope']
    stderr = sum_df.sem()['slope']

    print(f"mean: {mean} +/- {stderr}")
    return mean, stderr

def prepare_latex(paths_and_settings, functions = [50, 66], seeds = [12, 31, 64]):
    # columns: intent/function, #examples, setting, deriv_metric, min_acc, min_x, max_acc, max_x 
    latex_df = pd.DataFrame(columns=["function", "examples", "setting", "deriv_metric", "min_acc", "min_x", "max_acc", "max_x"],dtype=object)
    for path, setting in paths_and_settings:
        function, function_name, num_examples, model_name = setting
        df = get_intent_data_from_dir(path, function, seeds)
        prepped_df = prepare_intent(df, num_examples)
        detect_missing(function_name, prepped_df)
        mean, stddev = compute_derivative_metric(prepped_df)
        metric_str = f"${mean:.2f}\pm{stddev:.2f}$"
        acc_df = df[df['function'] == num_examples]
        acc_df = acc_df.set_index("train").mean(level="train")
        min_acc_idx = acc_df['fxn_acc'].argmin()
        min_acc_x = f"${acc_df.index[min_acc_idx]}$"
        min_acc = f"${acc_df['fxn_acc'].min():.2f}$"
        max_acc_idx = acc_df['fxn_acc'].argmax()
        max_acc_x = f"${acc_df.index[max_acc_idx]}$"
        max_acc = f"${acc_df['fxn_acc'].max():.2f}$"

        latex_df = latex_df.append({"function": function_name, "examples": num_examples, 
                                    "setting": model_name, "deriv_metric": metric_str, 
                                    "min_acc": min_acc, "min_x": min_acc_x,
                                    "max_acc": max_acc, "max_x": max_acc_x}, ignore_index=True)
    print(latex_df.to_latex(escape=False, float_format=".2f"))


if __name__ == "__main__":
    # paths_and_settings = [("/brtx/603-nvme1/estengel/intent/", (50, 15, "baseline")),
    #                       ("/brtx/603-nvme1/estengel/intent/", (50, 30, "baseline")),
    #                       ("/brtx/603-nvme1/estengel/intent/", (50, 75, "baseline")),
    #                       ("/brtx/604-nvme1/estengel/intent_no_source_triggers/", (50, 15, "remove source")),
    #                       ("/brtx/604-nvme1/estengel/intent_no_source_triggers/", (50, 30, "remove source")),
    #                       ("/brtx/604-nvme1/estengel/intent_no_source_triggers/", (50, 75, "remove source")),
    #                       ("/brtx/604-nvme1/estengel/intent_no_source_triggers/", (50, 15, "linear upsample")),
    #                       ("/brtx/604-nvme1/estengel/intent_no_source_triggers/", (50, 30, "remove source")),
    #                       ("/brtx/604-nvme1/estengel/intent_no_source_triggers/", (50, 75, "remove source"))]


    paths = {"baseline": "/brtx/603-nvme1/estengel/intent_fixed_test/intent/",
             "remove source": "/brtx/605-nvme1/estengel/intent_fixed_test/intent_no_source"}
            #  "upsample": "/brtx/605-nvme1/estengel/intent_fixed_test/intent_no_source" 
    # functions_and_names = [(50, "play_radio"), (66, "transit_traffic"), (15, "email_query"), (16, "email_querycontact"), (27, "general_quirky")]
    functions_and_names = [ (50, "play_radio"), (66, "transit_traffic"), (15, "email_query"), (16, "email_querycontact"), (27, "general_quirky")]
    # numbers = [15, 30, 75]
    numbers = [15, 30, 75]

    paths_and_settings = []

    for num in numbers:
        for fxn, name in functions_and_names:
            for model_name, path in paths.items():
                paths_and_settings.append((path, (fxn, name, num, model_name)))
    



    prepare_latex(paths_and_settings)


#    path = "/brtx/603-nvme1/estengel/intent/"
#    fxn = 50
#    seeds= [12, 31, 64]
#    df = get_intent_data_from_dir(path, fxn, seeds)
#    df = prepare_intent(df, 15)
#    mean, stddev = compute_derivative_metric(df)
#
#    path = "/brtx/604-nvme1/estengel/intent_no_source_triggers/"
#    df = get_intent_data_from_dir(path, fxn, seeds)
#    df = prepare_intent(df, 15)
#    mean, stddev = compute_derivative_metric(df)
#    #print(metric)
#
#
#
#
#