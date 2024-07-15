#!/bin/bash

cd meta_evaluation
if [[ ! -d data ]]; then
mkdir data
fi
cd data

if [[ ! -d official_submissions ]]; then
wget https://www.comp.nus.edu.sg/~nlp/conll14st/official_submissions.tar.gz
tar -xvf official_submissions.tar.gz

wget https://www.comp.nus.edu.sg/~nlp/conll14st/conll14st-test-data.tar.gz
tar -xvf conll14st-test-data.tar.gz
cat conll14st-test-data/noalt/official-2014.combined.m2 | grep '^S' | cut -d ' ' -f 2- > conll14_src.txt
cp conll14_src.txt official_submissions/INPUT

rm official_submissions.tar.gz
rm conll14st-test-data.tar.gz
rm -r conll14st-test-data
fi

cd ../
python calc_score.py \
    --model_id gpt2 \
    --threshold 0.8 \
    --out system_conll14_official.txt \
    --src data/conll14_src.txt \
    --system_outputs data/official_submissions/AMU \
        data/official_submissions/CAMB \
        data/official_submissions/RAC \
        data/official_submissions/CUUI \
        data/official_submissions/POST \
        data/official_submissions/PKU \
        data/official_submissions/UMC \
        data/official_submissions/UFC \
        data/official_submissions/IITB \
        data/official_submissions/INPUT \
        data/official_submissions/SJTU \
        data/official_submissions/NTHU \
        data/official_submissions/IPN

python calc_corr.py \
    --human Grundkiewicz-2015.txt \
    --system system_conll14_official.txt

