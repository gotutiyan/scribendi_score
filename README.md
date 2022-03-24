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
Confirmed that it works python3.7.10
```
pip install -r requirements.txt
```

# Usage
### CLI

```bash
python scribendi.py --src <source file> --pred <prediction file>
```

- Other options

`--no_cuda`   
`--model_id <str>`: Specify the id of GPT-2 related model of huggingface. (default: 'gpt2')  
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

# Reproducing of Human Correlation

It provides the scripts to reproduce a experiment, which calculate correlation between Scribendi Score's score and human's score. 

Note that I did not able to fully reproduce the results of the paper. The below table indicates the correlations with human score of [Grundkiewicz-2015](https://aclanthology.org/D15-1052/) (Human TrueSkill ranking).
|Correlation|Paper (Table 3)|Reproduced|
|:---|:--:|:--:|
|Pearson|0.951|0.913|
|Spearman|0.940|0.928|

### Procedure of Reproducing
```bash
bash prepare_data.sh
python correlation.py
```

The outputs will be like this:
```
Correlation with Grundkiewicz-2015
  Peason's correlation: 0.9139891180613645
  Spearman's correlation: 0.9285714285714285
```