#!/usr/bin/env bash

gpuid=$1
inf=$2
outf=$3

if [[ -z "$1" ]]; then
	gpuid=0
	echo GPUID not chosen, defaulting to "$gpuid"
fi

if [[ -z "$2" ]]; then
	inf=/data/datasets/susy/SUSY.csv
	echo filename not chosen, defaulting to "$inf"
fi

if [[ -z "$3" ]]; then
	outf=output.txt
	echo Outputfile not chosen, defaulting to "$outf"
fi

THEANO_FLAGS=mode=FAST_RUN,device=gpu$gpuid,floatX=float32,force_device=True python Control.py $inf 2>&1 | tee $outf
