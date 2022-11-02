import pathlib 

path = pathlib.Path(".")
all_py_files = path.glob("**/*.py") 
all_sh_files = path.glob("**/*.sh") 

header = """# Copyright (c) Microsoft Corporation.\n# Licensed under the MIT license.\n\n"""
for py_file in all_py_files:
    if "add_header" in str(py_file):
        continue
    py_data = open(py_file).read()
    py_data = header + py_data
    with open(py_file,"w") as f1:
        f1.write(py_data) 
    
for sh_file in all_sh_files:
    sh_data = open(sh_file).read()
    sh_data = header + sh_data
    with open(sh_file,"w") as f1:
        f1.write(sh_data) 
