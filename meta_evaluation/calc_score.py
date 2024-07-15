from scribendi import ScribendiScore
import argparse
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from typing import List, Dict, Tuple, Optional

def main(args) -> None:
    scorer = ScribendiScore(args.threshold, args.model_id, args.no_cuda)
    srcs = open(args.src).read().strip().split('\n')
    preds = [s.rstrip() for s in srcs]
    scores = []
    for out in args.system_outputs:
        preds = open(out).read().strip().split('\n')
        preds = [p.rstrip() for p in preds]
        score = scorer.score(src_sents=srcs, pred_sents=preds)
        scores.append(score)
    with open(args.out, 'w') as f:
        f.write('\n'.join(map(str, scores)))


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', default='gpt2')
    parser.add_argument('--threshold', type=float, default=0.8)
    parser.add_argument('--src', default='data/conll14_src.txt')
    parser.add_argument('--out', default='out.txt')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--system_outputs', nargs='+')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_parser()
    main(args)