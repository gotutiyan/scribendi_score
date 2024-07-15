# set -eu

cd meta_evaluation

if [[ ! -f scribendi.py ]]; then
ln -s ../scribendi.py
fi

if [[ ! -d data ]]; then
mkdir data
fi
cd data

if [[ ! -d SEEDA ]]; then
git clone https://github.com/tmu-nlp/SEEDA.git
fi

cd ../
python calc_score.py \
    --model_id gpt2 \
    --threshold 0.8 \
    --out system_seeda.txt \
    --src data/SEEDA/outputs/subset/INPUT.txt \
    --system_outputs data/SEEDA/outputs/subset/BART.txt \
        data/SEEDA/outputs/subset/BERT-fuse.txt \
        data/SEEDA/outputs/subset/GECToR-BERT.txt \
        data/SEEDA/outputs/subset/GECToR-ens.txt \
        data/SEEDA/outputs/subset/GPT-3.5.txt \
        data/SEEDA/outputs/subset/INPUT.txt \
        data/SEEDA/outputs/subset/LM-Critic.txt \
        data/SEEDA/outputs/subset/PIE.txt \
        data/SEEDA/outputs/subset/REF-F.txt \
        data/SEEDA/outputs/subset/REF-M.txt \
        data/SEEDA/outputs/subset/Riken-Tohoku.txt \
        data/SEEDA/outputs/subset/T5.txt \
        data/SEEDA/outputs/subset/TemplateGEC.txt \
        data/SEEDA/outputs/subset/TransGEC.txt \
        data/SEEDA/outputs/subset/UEDIN-MS.txt \

echo 'SEEDA-S (TrueSkill)'
python data/SEEDA/utils/corr_system.py \
    --human data/SEEDA/scores/human/TS_sent.txt \
    --metric system_seeda.txt \

echo 'SEEDA-E (TrueSkill)'
python data/SEEDA/utils/corr_system.py \
    --human data/SEEDA/scores/human/TS_edit.txt \
    --metric system_seeda.txt