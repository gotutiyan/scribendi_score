from scribendi import ScribendiScore
import argparse
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from typing import List, Dict, Tuple, Optional

def main(args) -> None:
    scorer = ScribendiScore(args.threshold, args.model_id, args.no_cuda)
    src = load_file(args.src)
    system_outputs = dict()
    for file_name in args.system_outputs:
        system_name = file_name.split('/')[-1]
        system_outputs[system_name] = load_file(file_name)
    peason_crr, spearman_crr = calc_correlation(
        src, system_outputs, scorer,
        paper_id=args.paper_id,
        verbose=args.verbose
    )
    print('Correlation with {}:'.format(args.paper_id))
    print('  Peason\'s correlation:', peason_crr)
    print('  Spearman\'s correlation:', spearman_crr)

def calc_correlation(
    src_sents: List[str], 
    system_outputs: Dict[str, List[str]],
    scorer: ScribendiScore,
    paper_id: str='Grundkiewicz-2015',
    verbose:bool=False
) -> Tuple[float, float]:
    system_scores = dict()
    for sys_name, pred_sents in tqdm(system_outputs.items()):
        print(sys_name, end=': ')
        score = scorer.score(src_sents, pred_sents, batch_size=32, verbose=verbose)
        system_scores[sys_name] = score
    human_rank, human_score = load_human_rank(paper_id=paper_id)
    sorted_system_rank = sorted(system_scores.items(), key=lambda x:x[1], reverse=True)
    system_rank = [s[0] for s in sorted_system_rank]
    system_score = [s[1] for s in sorted_system_rank]
    if verbose:
        print('Human Rank ({}):'.format(paper_id), human_rank)
        print('Human Score ({}):'.format(paper_id), human_score)
        print('System Rank:', system_rank)
        print('System Score:', system_score)
    pearson_crr = pearsonr(
        [human_score[human_rank.index(sys_name)] for sys_name in human_rank],
        [system_score[system_rank.index(sys_name)] for sys_name in human_rank]
    )
    spearman_crr = spearmanr(
        [human_score[human_rank.index(sys_name)] for sys_name in human_rank],
        [system_score[system_rank.index(sys_name)] for sys_name in human_rank]
    )
    return pearson_crr[0], spearman_crr[0]
        
def load_human_rank(
    paper_id: str='Grundkiewicz-2015'
) -> Tuple[List[str], List[float]]:
    '''
    return: system ranking and its score.
    For Grundkiewicz-2015, it used Human TrueSkill ranking.
    '''
    if paper_id == 'Grundkiewicz-2015':
        return 'AMU CAMB RAC CUUI POST PKU UMC UFC IITB INPUT SJTU NTHU IPN'.split(),\
                [0.273, 0.182, 0.114, 0.105, 0.080, -0.001, -0.022, -0.041, -0.055, -0.062, -0.074, -0.142, -0.358]
    # elif paper_id == 'Napoles-2015':
    #     return 'CAMB AMU RAC CUUI INPUT POST UFC SJTU IITB PKU UMC NTHU IPN'.split()
    else:
        raise ValueError('{} is invalid for paper_id.'.format(paper_id))

def load_file(file_path: str) -> List[str]:
    sentences = []
    with open(file_path) as fp:
        for line in fp:
            sent = line.rstrip()
            sentences.append(sent)
    return sentences
    

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', default='gpt2')
    parser.add_argument('--threshold', type=float, default=0.8)
    parser.add_argument('--src', default='data/conll14_src.txt')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--paper_id', choices=['Grundkiewicz-2015'], default='Grundkiewicz-2015')
    parser.add_argument('--system_outputs', nargs='+', default=[
        'data/official_submissions/AMU',
        'data/official_submissions/CAMB',
        'data/official_submissions/CUUI',
        'data/official_submissions/IITB',
        'data/official_submissions/IPN',
        'data/official_submissions/NTHU',
        'data/official_submissions/PKU',
        'data/official_submissions/POST',
        'data/official_submissions/RAC',
        'data/official_submissions/SJTU',
        'data/official_submissions/UFC',
        'data/official_submissions/UMC',
        'data/official_submissions/INPUT',
    ])
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_parser()
    main(args)