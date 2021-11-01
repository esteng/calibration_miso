
import sys
import pathlib

root_path = pathlib.Path(sys.argv[1]) 

fxns = ["FindManager", "Tomorrow", "DoNotConfirm", "PlaceHasFeature", "FenceAttendee"]

out_paths = ["smcalflow_samples_curated", "smcalflow_samples_curated", "smcalflow_samples_big", "smcalflow_samples", "smcalflow_samples"]

for fxn, path in zip(fxns, out_paths):
    for split in ["test", "dev"]: 
        out_path = root_path.joinpath(path, fxn) 
        fxn_source_path = out_path.joinpath(f"fxn_{split}_valid.src_tok") 
        fxn_target_path = out_path.joinpath(f"fxn_{split}_valid.tgt") 

        data_path = root_path.joinpath("smcalflow.agent.data") 
        source_path = data_path.joinpath(f"{split}_valid.src_tok") 
        target_path = data_path.joinpath(f"{split}_valid.tgt") 


        with open(source_path) as src_f, open(target_path) as tgt_f: 
            src_lines = src_f.readlines()
            tgt_lines = tgt_f.readlines()

        idxs=[]
        for i, tl in enumerate(tgt_lines):
            if fxn in tl.strip().split(" "):
                idxs.append(i) 

        fxn_src_lines = [src_lines[i] for i in idxs]
        fxn_tgt_lines = [tgt_lines[i] for i in idxs]

        with open(fxn_source_path, "w") as fsrc_f, open(fxn_target_path, "w") as ftgt_f:
            for sl, tl in zip(fxn_src_lines, fxn_tgt_lines):
                fsrc_f.write(sl.strip() + "\n") 
                ftgt_f.write(tl.strip() + "\n") 


        

            
