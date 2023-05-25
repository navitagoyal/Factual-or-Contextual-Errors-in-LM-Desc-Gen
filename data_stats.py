import json
import pickle as pkl

from dataloader import DescGenProcessor
from mcqa_dataloader import MCQAProcessor, ClaimProcessor
from eval_utils import get_known_unknown

config = json.load(open('config.json', 'r'))


with open(config['known_file'], 'r') as f:
    known_entities = f.readlines()

known_entities = [i.rstrip() for i in known_entities]


def known_unknown_stats(data_proc, task, only_in_wiki=True):
    for split in ['train', 'valid', 'test']:
        if task=='descgen':
            texts, labels, TPs, indices = data_proc.create_data(split)
        else:
            texts, labels, label_classes, TPs, indices = data_proc.create_data(split)
        
        if only_in_wiki:

            wiki_file = f"../saved/wiki_text/articles_cnn_{split}.pkl"
            with open(wiki_file, 'rb') as f:
                wiki_text = pkl.load(f)

            TPs = [i for i in TPs if i in wiki_text]
        known_indices, unknown_indices = get_known_unknown(TPs, known_entities)
        print(f"Split: {split}, Known: {len(known_indices)}, Unknown: {len(unknown_indices)}")


def main():
    
    sentence_only=False
    max_sent=5
    total_sent=10
    mask_token='[MASK]'

    data_proc = DescGenProcessor('masked_data', 
                            clipped=not sentence_only, 
                            max_sent=max_sent, 
                            mask_token=mask_token)
    
    known_unknown_stats(data_proc, task='descgen')
    
    data_proc = ClaimProcessor('claim_mcqa', 
                              clipped=not sentence_only, 
                              max_sent=max_sent, 
                              total_sent=total_sent, 
                              mask_token=mask_token)
    known_unknown_stats(data_proc, task='claimid', only_in_wiki=False)
    
    data_proc = MCQAProcessor('masked_data', 'desc_mcqa', 
                              clipped=not sentence_only, 
                              max_sent=max_sent, 
                              total_sent=total_sent, 
                              mask_token=mask_token)
    known_unknown_stats(data_proc, task='descid', only_in_wiki=False)
        
    
    
    
if __name__=='__main__':
    main()