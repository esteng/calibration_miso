#!/bin/bash

dest_num=$1


for num in 2 3 4 5 6
do 
    for ratio in 0.10_0.10  0.10_0.30  0.10_0.50  0.10_0.70  0.10_0.90  0.25_0.10  0.25_0.30  0.25_0.50  0.25_0.70  0.25_0.90  0.50_0.10  0.50_0.30  0.50_0.50  0.50_0.70  0.50_0.90  1.00_0.10  1.00_0.30  1.00_0.50  1.00_0.70  1.00_0.90  0.10_0.20  0.10_0.40  0.10_0.60  0.10_0.80  0.10_1.00  0.25_0.20  0.25_0.40  0.25_0.60  0.25_0.80  0.25_1.00  0.50_0.20  0.50_0.40  0.50_0.60  0.50_0.80  0.50_1.00  1.00_0.20  1.00_0.40  1.00_0.60  1.00_0.80  1.00_1.00  
    do 
        for seed in 12 31 64
        do 
            source_path="/brtx/60${num}-nvme1/estengel/intent_conj_diverse/${ratio}/${seed}/"
            dest_path="/brtx/60${dest_num}-nvme1/estengel/intent_conj_diverse/${ratio}/${seed}/"
            # skip if destination already exists 
            [ -d ${dest_path} ] && continue
            
            # skip if the same  
            if [ "${source_path}" = "${dest_path}" ]; then
                continue
            fi

            #mkdir -p ${dest_path}.basename
            # only try to move if source exists  
            mkdir -p "/brtx/60${dest_num}-nvme1/estengel/intent_conj_diverse/${ratio}/"
            [ -d ${source_path} ] && echo "mv ${source_path} ${dest_path}" && mv ${source_path} ${dest_path}
        done
    done 
done
