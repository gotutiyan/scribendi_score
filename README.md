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
I used python3.7.10
```
pip install -r requirements.txt
```

# Usage
- CLI

```bash
python scribendi.py --src <source file> --pred <prediction file>
```

- API  

Please clone this repository then you can use in the following:
```python
from scribendi import ScribendiScore
scorer = ScribendiScore()
src_sents = ['This is a sentence .', 'This is a also sentence .']
pred_sents = ['This a is sentence .', 'This is a also sentence .']
print(scorer.score(src_sents, pred_sents)) # -1 (-1 + 0)
```

# Reproducing of Human Correlation

It provides the scripts to reproduce a experiment, which calculate correlation between Scribendi Score's rank and human's rank. 

Note that I did not able to fully reproduce the results of the paper. The below table indicates the correlations with [Grundkiewicz-2015](https://aclanthology.org/D15-1052/) (Human TrueSkill ranking).
|Correlation|Islam+ 2021 (Table 3)|Reproduced|
|:---|:--:|:--:|
|Pearson|0.951|0.913|
|Spearman|0.940|0.928|

### Procedure
Prepare data.
```bash
bash prepare_data.sh
```

Then, 
```
python correlation.py
```
The outputs will be like this:
```
Correlation with Grundkiewicz-2015
  Peason's correlation: 0.9139891180613645
  Spearman's correlation: 0.9285714285714285
```