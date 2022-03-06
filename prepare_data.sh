#!/bin/bash

mkdir data
cd data
wget https://www.comp.nus.edu.sg/~nlp/conll14st/official_submissions.tar.gz
tar -xvf official_submissions.tar.gz

wget https://www.comp.nus.edu.sg/~nlp/conll14st/conll14st-test-data.tar.gz
tar -xvf conll14st-test-data.tar.gz
cat conll14st-test-data/noalt/official-2014.combined.m2 | grep '^S' | cut -d ' ' -f 2- > conll14_src.txt
cp conll14_src.txt official_submissions/INPUT

rm official_submissions.tar.gz
rm conll14st-test-data.tar.gz
rm -r conll14st-test-data