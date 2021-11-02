#!/bin/bash

name=$1
dest_num=$2


for num in 2 3 4 5 6
do 
    for intent in 15 16 27 50 66 
    do 
        for seed in 12 31 64
        do 
            source_path="/brtx/60${num}-nvme1/estengel/intent_fixed_test/${name}/${intent}/${seed}_seed"
            dest_path="/brtx/60${dest_num}-nvme1/estengel/intent_fixed_test/${name}/${intent}/${seed}_seed"
            # skip if destination already exists 
            [ -d ${dest_path} ] && continue
            
            # skip if the same  
            if [ "${source_path}" = "${dest_path}" ]; then
                continue
            fi

            #mkdir -p ${dest_path}.basename
            # only try to move if source exists  
            [ -d ${source_path} ] && echo "mv ${source_path} ${dest_path}" && mv ${source_path} ${dest_path}

        done
    done 
done
