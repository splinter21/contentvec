#!/bin/bash

expdir=./tmp
mkdir -p $expdir

python -u ./fairseq/fairseq_cli/hydra_train.py --config-dir ./contentvec/config/contentvec --config-name contentvec hydra.run.dir=${expdir}