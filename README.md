# Scribendi Score

This an unofficial repository that reproduces Scribendi Score, proposed in the following paper.

```
@inproceedings{islam-magnani-2021-end,
    title = "Is this the end of the gold standard? A straightforward reference-less grammatical error correction metric",
    author = "Islam, Md Asadul  and
      Magnani, Enrico",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.239",
    doi = "10.18653/v1/2021.emnlp-main.239",
    pages = "3009--3015",
}
```

Note that the human correlation experiments were not able to fully reproduce the results of the paper.

# Install
Confirmed that it works python3.11.0
```
pip install -r requirements.txt
```

# Usage
### CLI

```bash
python scribendi.py --src <source file> --pred <prediction file>
```

Note that multiple source and prediction files can be given, separated by
colons. If the colon-separated lists of files are of identical length, they
are matched directly. Alternatively, a single source file may be given along
with multiple prediction files, for instance from multiple systems.
Example:

```bash
python scribendi.py --src test.original --pred test.system1:test.system2
```

This is useful to save time with large language models and multiple small
evaluation files.

- Other options

`--no_cuda`   
`--model_id <str>`: Specify the id of GPT-2 related model of huggingface. (default: 'gpt2') You can also specify a local directory, from which the model will be loaded.
`--threshold <float>`: The threshold of a token sort ratio and a levenshtein distance ratio. (default: 0.8)  
`--batch_size <int>`: The batch size to compute perplexity of a GPT-2 model faster. (default: 32)  
`--example`: Show the perplexity, the token sort ration and the levenshtein distance ratio using the examples of Table 1 in the [paper](https://aclanthology.org/2021.emnlp-main.239/).

- Demo
```bash
python scribendi.py --src demo/src.txt --pred demo/pred.txt --no_cuda
```
### API  

```python
from scribendi import ScribendiScore
scorer = ScribendiScore()
src_sents = ['This is a sentence .', 'This is another sentence .']
pred_sents = ['This a is sentence .', 'This is another sentence .']
print(scorer.score(src_sents, pred_sents)) # -1 (-1 + 0)
```

# Meta-evaluation

### Correlations on the CoNLL-14 submissions

An example to calculate the correlation with [Grundkiewicz-2015](https://aclanthology.org/D15-1052/) (Human TrueSkill ranking).  
`Grundkiewicz-2015.txt`'s scores are arranged as follows: `[]`

```sh
bash meta_evaluation/conll14_official.sh
```

The results are:

|Correlation|Paper (Table 3)|Ours|
|:---|:--:|:--:|
|Pearson|0.951|0.913|
|Spearman|0.940|0.928|

### Correlations on SEEDA

An example to calculate the correlation using SEEDA [Kobayashi+ 2024](https://arxiv.org/abs/2403.02674).

```sh
bash meta_evaluation/seeda.sh
```

Outputs:
```
SEEDA-S
Pearson: 0.6308356098410074
Spearman: 0.6409817185295869
SEEDA-E
Pearson: 0.8323609711579115
Spearman: 0.847637026689399
```
