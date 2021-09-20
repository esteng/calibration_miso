# Using Amulet

I've been using [amulet](https://amulet-docs.azurewebsites.net/master/index.html) for submitting jobs at scale. 

## Setting up Amulet 

The steps I took to get this set up were: 

1. follow instructions on [installing amulet](https://amulet-docs.azurewebsites.net/master/setup.html#installation)
2. set up an [azure storage account](https://amulet-docs.azurewebsites.net/master/setup.html#azure-storage-account)
    - navigate to the storage account through the [azure portal](https://ms.portal.azure.com/#home) 
    - create a container called `miso`, which is where all models will be stored 
3. install [blobfuse](https://github.com/Azure/azure-storage-fuse)
4. mount the `miso` container locally as `~/amlt_models/`
    - `blobfuse ~/amlt_models --container-name=miso --tmp-path=/mnt/blobfusetmp` 
5. put all data in `~/resources/data` 

## Submitting jobs 
Jobs are submitted by running `amlt run <config>.yaml :<command_name> my_experiment`, specifying a config file, a command to run from the config, and an experiment name. 
Before submitting jobs, the data may need to be uploaded if it has changed. The data path is specified in the config file. The way I set up the config files, they all point
to the same data (`~/resources/data`). This has its pros and cons. On the one hand, we save time because we don't have to upload data for each experiment, only each time the
data changes. On the other hand, any time the data changes, we have to re-upload all of the data, which takes ~5 minutes. As long as we're submitting a lot of jobs with the 
same data, I think this is still effective. 

When you submit a job with the above command, you will get a bunch of prompts asking to confirm aspects of the job. This makes submitting jobs in a loop annoying, so the 
files I have for submitting jobs (`submit_amlt_dangerous.sh`) override the prompts with the `-r -y` flags. This is dangerous (hence the name) since the `-r -y` flags will 
skip the warning and confirmation step if the submitted job has the same name as an existing job, and will overwrite the existing job. 

In a given amulet config, there are a few commands. The most important is `:train` which sets all the environment variables and calls `experiments/calflow.sh -a train` to 
train a model. There is also a list of requirements, which amulet installs on the target instance. It uploads all of the code specified and then runs the specified commands. 
Unnecessary files can be ignored by modifying the `.amltignore` file. 

## Collecting results 
Running the `:train` command for a given config will train a model, then decode test and dev sets and write the results of that to `CHECKPOINT_DIR/translate_output`. 
To collect the results into a csv, I have a script called `scripts/collect_results.py`, that measures the exact match and coarse/fine-grained accuracy. 
