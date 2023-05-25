import nltk
import numpy as np
import pickle as pkl
from tqdm.auto import tqdm

import eval_metrics
import utils
import logging

metric_fns = {
    'rougeL_precision': eval_metrics.rouge_L,
    'bert_precision': eval_metrics.bertscore,
    'bleu_precision': eval_metrics.modified_unigram_precision
}


def remove_true_desc(text, label):
    text_tokens = nltk.word_tokenize(text)
    label_tokens = nltk.word_tokenize(label)
    text_tokens = [i for i in text_tokens if i not in label_tokens]
    return ' '.join(text_tokens)

    
def create_data_subset(output_labels, inputs, labels, TP, indices):
    output_labels_sub, inputs_sub, labels_sub, TP_sub  = list(zip(*[i for idx, i in enumerate(zip(output_labels, inputs, labels, TP)) if idx in indices]))
    
    return output_labels_sub, inputs_sub, labels_sub, TP_sub


def get_known_unknown(input_TPs, known_entities):
    known_indices = [idx for idx, i in enumerate(input_TPs) if i in known_entities]
    unknown_indices = [idx for idx, i in enumerate(input_TPs) if i not in known_entities]
    
    return known_indices, unknown_indices
    
    
    
class EvaluateUtils:
    def __init__(self, known_file, wiki_file):
        with open(known_file, 'r') as f:
            known_entities = f.readlines()
            
        self.known_entities = [i.rstrip() for i in known_entities]
        
        with open(wiki_file, 'rb') as f:
            self.wiki_text = pkl.load(f)
            
            
            
    def evaluate_wo_classes(self, output_labels, test_labels):
        
        ### Overlap between true and predicted description
        overlap = [metric(t[0], t[1]) for t in zip(test_labels, output_labels)]
        return overlap
            
        
        
    def evaluate_classes(self, output_labels, test_texts, test_labels, test_TP, metric):
        
        
        ### Overlap between true and predicted description
        overlap = [metric(t[0], t[1]) for t in tqdm(zip(test_labels, output_labels), total=len(output_labels))]
        
        ##### Remove true description from the ref_texts (wiki/context) before taking overlap

        ### Overlap between predicted description and wiki text of the person aka factual-non-contextual

        ref_texts = [' '.join(self.wiki_text[t]) for t in test_TP]
        ref_texts = [remove_true_desc(t[0], t[1]) for t in zip(ref_texts, test_labels)]
        wiki_overlap = [metric(t[0], t[1]) for t in tqdm(zip(ref_texts, output_labels), total=len(output_labels))]

        ### Overlap between predicted description and the rest of the article text aka contextual-non-factual

        test_texts = [remove_true_desc(t[0], t[1]) for t in zip(test_texts, test_labels)]

        context_overlap = [metric((t[0]), t[1]) for t in tqdm(zip(test_texts, output_labels), total=len(output_labels))]
        
        
        return overlap, wiki_overlap, context_overlap
        
        
    def evaluate_frac(self, true_overlap, wiki_overlap, context_overlap, metric):
        if metric not in ['word_error_rate']:
            class_labels = [np.argmax(i) for i in zip(true_overlap, wiki_overlap, context_overlap)]
        else:
            class_labels = [np.argmin(i) for i in zip(true_overlap, wiki_overlap, context_overlap)]
            
        return (*[[1 if i==c else 0 for i in class_labels] for c in range(3)], )
    
    
    def evaluate_mrr(self, true_overlap, wiki_overlap, context_overlap, metric):
        ### Rank of true, wiki, context class for each example
        if metric not in ['word_error_rate']:
            class_ranks = [[sorted(l, reverse=True).index(x)+1 for x in l] for l in zip(true_overlap, wiki_overlap, context_overlap)] 
        else:    
            class_ranks = [[sorted(l).index(x)+1 for x in l] for l in zip(true_overlap, wiki_overlap, context_overlap)] 
        
        ### list of reverse rank: [1, 0.33, 0.5] if true is rank=1, context is rank 2 and wiki is rank 3
        rev_class_ranks = [[1/i for i in example] for example in class_ranks] 
        
        ### Mean of reverse ranks across all examples
        return np.asarray(rev_class_ranks).T
        
        
        
class Evaluate(EvaluateUtils):
    
    def __init__(self, known_file, wiki_file):
        super().__init__(known_file, wiki_file)
    
    
    def evaluate(self, output_labels, test_texts, test_labels, test_TP, metric):
        
        metric_fn = metric_fns[metric]
                
        true_overlap, wiki_overlap, context_overlap = super().evaluate_classes(output_labels, 
                                                                             test_texts, 
                                                                             test_labels, 
                                                                             test_TP, 
                                                                             metric_fn)
        
        assert len(true_overlap) == len(wiki_overlap) == len(context_overlap) == len(output_labels)
        
        ### Overlap w.r.t. ground truth, wiki and context
        return true_overlap, wiki_overlap, context_overlap
        
    
    def collate(self, true_overlap, wiki_overlap, context_overlap, metric):
        logging.info(f'********** {metric} **********' )

        logging.info(
            f'Accurate: {round(np.mean(true_overlap), 4)}, '\
            f'Incongruous: {round(np.mean(wiki_overlap), 4)}, '\
            f'Nonfactual: {round(np.mean(context_overlap), 4)}'
        )
        
        true_known, wiki_known, context_known = zip(*[instance for idx, instance in enumerate(zip(true_overlap, wiki_overlap, context_overlap)) if idx in self.known_indices])
        
        true_unknown, wiki_unknown, context_unknown = zip(*[instance for idx, instance in enumerate(zip(true_overlap, wiki_overlap, context_overlap)) if idx in self.unknown_indices])
        
        logging.info(
            f'Accurate Known: {round(np.mean(true_known), 4)}, '\
            f'Incongruous Known: {round(np.mean(wiki_known), 4)}, '\
            f'Nonfactual Known: {round(np.mean(context_known), 4)}'
        )
        
        logging.info(
            f'Accurate Unknown: {round(np.mean(true_unknown), 4)}, '\
            f'Incongruous Unknown: {round(np.mean(wiki_unknown), 4)}, '\
            f'Nonfactual Unknown: {round(np.mean(context_unknown), 4)}'
        )
        
        
        logging.info(f'********** {metric}:Top **********')
        
        true_frac, wiki_frac, context_frac = super().evaluate_frac(true_overlap, wiki_overlap, context_overlap, metric)
        assert len(true_frac) == len(wiki_frac) == len(context_frac) == len(true_overlap)
        
        logging.info(
            f'Accurate: {round(np.mean(true_frac), 4)}, '\
            f'Incongruous: {round(np.mean(wiki_frac), 4)}, '\
            f'Nonfactual: {round(np.mean(context_frac), 4)}'
        )
        
        true_frac_known, wiki_frac_known, context_frac_known = zip(*[instance for idx, instance in enumerate(zip(true_frac, wiki_frac, context_frac)) if idx in self.known_indices])
        
        true_frac_unknown, wiki_frac_unknown, context_frac_unknown = zip(*[instance for idx, instance in enumerate(zip(true_frac, wiki_frac, context_frac)) if idx in self.unknown_indices])
        
        logging.info(
            f'Accurate Known: {round(np.mean(true_frac_known), 4)}, '\
            f'Incongruous Known: {round(np.mean(wiki_frac_known), 4)}, '\
            f'Nonfactual Known: {round(np.mean(context_frac_known), 4)}'
        )
        
        logging.info(
            f'Accurate Unknown: {round(np.mean(true_frac_unknown), 4)}, '\
            f'Incongruous Unknown: {round(np.mean(wiki_frac_unknown), 4)}, '\
            f'Nonfactual Unknown: {round(np.mean(context_frac_unknown), 4)}'
        )
        
        
        logging.info(f'********** {metric}:MRR **********')
        
        true_mrr, wiki_mrr, context_mrr = super().evaluate_mrr(true_overlap, wiki_overlap, context_overlap, metric)
        assert len(true_mrr) == len(wiki_mrr) == len(context_mrr) == len(true_overlap)
        
        logging.info(
            f'Accurate: {round(np.mean(true_mrr), 4)}, '\
            f'Incongruous: {round(np.mean(wiki_mrr), 4)}, '\
            f'Nonfactual: {round(np.mean(context_mrr), 4)}'
        )
        
        true_mrr_known, wiki_mrr_known, context_mrr_known = zip(*[instance for idx, instance in enumerate(zip(true_mrr, wiki_mrr, context_mrr)) if idx in self.known_indices])
        
        true_mrr_unknown, wiki_mrr_unknown, context_mrr_unknown = zip(*[instance for idx, instance in enumerate(zip(true_mrr, wiki_mrr, context_mrr)) if idx in self.unknown_indices])
        
        logging.info(
            f'Accurate Known: {round(np.mean(true_mrr_known), 4)}, '\
            f'Incongruous Known: {round(np.mean(wiki_mrr_known), 4)}, '\
            f'Nonfactual Known: {round(np.mean(context_mrr_known), 4)}'
        )
        
        logging.info(
            f'Accurate Unknown: {round(np.mean(true_mrr_unknown), 4)}, '\
            f'Incongruous Unknown: {round(np.mean(wiki_mrr_unknown), 4)}, '\
            f'Nonfactual Unknown: {round(np.mean(context_mrr_unknown), 4)}'
        )
        
        
        
    
    def main(self, output_labels, test_texts, test_labels, test_TP, log_file):
        if log_file:
            logging.basicConfig(filename=log_file, filemode='w', format='%(message)s')
        else:
            logging.basicConfig(format='%(message)s')
        logging.getLogger().setLevel(logging.INFO)
        
        print(f"{len([x for x in output_labels if x==''])} generated labels empty")
        
        print( len(output_labels), len(test_texts), len(test_labels), len(test_TP))
        
        
        ### Remove output labels that are empty
        output_labels, test_texts, test_labels, test_TP = zip(*[i for i in zip(output_labels, test_texts, test_labels, test_TP) if i[0]!=''])
        
        
        assert len(output_labels) == len(test_texts) == len(test_labels) == len(test_TP)
        
        ### Remove test instances without wiki text
        self.output_labels, self.test_texts, self.test_labels, self.test_TP = zip(*[i for i in zip(output_labels, test_texts, test_labels, test_TP) if i[3] in self.wiki_text])
        
        assert len(self.output_labels) == len(self.test_texts) == len(self.test_labels) == len(self.test_TP)
        
        
        
        self.known_indices, self.unknown_indices = get_known_unknown(self.test_TP, self.known_entities)
        
        print(f'Total: {len(self.test_labels)}, Known: {len(self.known_indices)}, Unknown {len(self.unknown_indices)}')
        
        # for metric in ['unigram_precision']:
        for metric in list(metric_fns.keys()):
            overlap, wiki_overlap, context_overlap = self.evaluate(self.output_labels, 
                                                                         self.test_texts, 
                                                                         self.test_labels, 
                                                                         self.test_TP, 
                                                                         metric)
            
            self.collate(overlap, wiki_overlap, context_overlap, metric)
             
                