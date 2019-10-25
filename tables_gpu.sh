#!/bin/sh
SRC=/umbc/xfs1/cybertrn/cybertraining2019/team3/research/results/source/
PY=/umbc/xfs1/cybertrn/cybertraining2019/team3/research/miniconda3_cpu/bin/
ROOT=/umbc/xfs1/gobbert/group_saved/research/datascience/papers/Barajas_Thesis2019/tables/multipleGPUs_moreBatch/
mkdir -p $ROOT
# Delete current results
rm results-2013.txt 
rm results-2018.txt 
# Regather results
#cat 2013/*/result-* >> results-2013.txt
#cat 2018/*/result-* >> results-2018.txt
find ./2013/ -type f -name "result-*" -exec cat {} \; >> results-2013.txt
find ./2018/ -type f -name "result-*" -exec cat {} \; >> results-2018.txt
# Arrange tables
$PY/python $SRC/postProcess.py -j technical_report_2013.json \
    -f results-2013.txt -g epochs -c gpu data_multiplier \
    -eoc "for the 2013 GPUs with preaugmented data and forced parallelism" \
    -o $ROOT/gpu2013_timings_gnvdmgep.tex $ROOT/gpu2013_accuracy_gnvdmgep.tex \
    <<< -1
$PY/python $SRC/postProcess.py -j technical_report_2013.json \
    -f results-2013.txt -g epochs -c gpu batch_size \
    -eoc "for the 2013 GPUs with preaugmented data and forced parallelism" \
    -o $ROOT/gpu2013_timings_gnvbsgep.tex $ROOT/gpu2013_accuracy_gnvbsgep.tex \
    <<< -1
$PY/python $SRC/postProcess.py -j technical_report_2013.json \
    -f results-2013.txt -g gpu -c batch_size epochs \
    -eoc "for the 2013 GPUs with preaugmented data and forced parallelism" \
    -o $ROOT/gpu2013_timings_bsvepggn.tex $ROOT/gpu2013_accuracy_bsvepggn.tex \
    <<< -1
$PY/python $SRC/postProcess.py -j technical_report_2018.json \
    -f results-2018.txt -g epochs -c gpu data_multiplier \
    -eoc "for the 2018 GPUs with preaugmented data and forced parallelism" \
    -o $ROOT/gpu2018_timings_gnvdmgep.tex $ROOT/gpu2018_accuracy_gnvdmgep.tex \
    <<< -1
$PY/python $SRC/postProcess.py -j technical_report_2018.json \
    -f results-2018.txt -g epochs -c gpu batch_size \
    -eoc "for the 2018 GPUs with preaugmented data and forced parallelism" \
    -o $ROOT/gpu2018_timings_gnvbsgep.tex $ROOT/gpu2018_accuracy_gnvbsgep.tex \
    <<< -1
$PY/python $SRC/postProcess.py -j technical_report_2018.json \
    -f results-2018.txt -g gpu -c batch_size epochs \
    -eoc "for the 2018 GPUs with preaugmented data and forced parallelism" \
    -o $ROOT/gpu2018_timings_bsvepggn.tex $ROOT/gpu2018_accuracy_bsvepggn.tex \
    <<< -1
$PY/python $SRC/postProcess.py -j technical_report_2018.json \
    -f results-2018.txt -g data_multiplier -c gpu epochs \
    -eoc "for the 2018 GPUs with preaugmented data and forced parallelism" \
    -o $ROOT/gpu2018_timings_gnvepgdm.tex $ROOT/gpu2018_accuracy_gnvepgdm.tex \
    <<< 0
