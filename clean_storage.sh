#!/bin/bash

IN_PATH=$1

for path in ${IN_PATH}/*
do
	if test -f ${path}/ckpt/model.tar.gz; 
	then
		echo "${path} has model";
		rm ${path}/ckpt/*.th;
	fi

done

