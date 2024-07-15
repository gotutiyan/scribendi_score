import argparse
from scipy.stats import pearsonr, spearmanr

def main(args):
    h_scores = open(args.human).read().rstrip().split('\n')
    s_scores = open(args.system).read().rstrip().split('\n')
    h_scores = list(map(float, h_scores))
    s_scores = list(map(float, s_scores))
    pea, pea_p = pearsonr(h_scores, s_scores)
    spe, spe_p = spearmanr(h_scores, s_scores)
    print('Pearson Corr:', pea)
    print('Spearman Corr:', spe)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--human', required=True)
    parser.add_argument('--system', required=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_parser()
    main(args)