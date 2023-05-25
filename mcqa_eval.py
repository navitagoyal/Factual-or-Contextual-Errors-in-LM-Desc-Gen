import logging
import numpy as np
import pickle as pkl
from sig_utils import significance_eval


def create_data_subset(output_labels, inputs, labels, label_classes, TP, indices):
    output_labels_sub, inputs_sub, labels_sub, label_classes_sub, TP_sub  = list(zip(*[i for idx, i in enumerate(zip(output_labels, inputs, labels, label_classes, TP)) if idx in indices]))
    
    return output_labels_sub, inputs_sub, labels_sub, label_classes_sub, TP_sub


def get_known_unknown(input_TPs, known_entities):
    known_indices = [idx for idx, i in enumerate(input_TPs) if i in known_entities]
    unknown_indices = [idx for idx, i in enumerate(input_TPs) if i not in known_entities]
    
    return known_indices, unknown_indices



def calculate_highest(predictions, labels, label_classes):
    pred = [np.argmax(output) for output in predictions]
    pred_class = [label_class[label] for label, label_class in zip(pred, label_classes)]
    
    return (*[[1 if i==c else 0 for i in pred_class] for c in range(4)], )


def calculate_mrr(predictions, labels, label_classes):
    predictions = np.asarray(predictions)
    ### rank 1 for highest prob and 4 for lowest prob
    pred_rank = 4 - predictions.argsort().argsort()
    
    ### store indices of each label class in the options
    label_classes_ind = [{label_class:idx for idx, label_class in enumerate(instance)} for instance in label_classes]

    ### sanity check
    # assert(labels==[i[0] for i in label_classes_ind])
    
    label_classes_rank = {}
    for label_class in [0, 1, 2, 3]:
        label_rank = []
        for pred, label_position in zip(pred_rank, label_classes_ind):
            label_rank.append(pred[label_position[label_class]])
        label_classes_rank[label_class] = label_rank  
        
    
    rev_class_ranks = [[1/i for i in label_classes_rank[label_class]] for label_class in range(4)]

    # mrr = {}
    # for label_class in [0, 1, 2, 3]:
    #     mr = np.mean([1/i for i in label_classes_rank[label_class]])
    #     mrr[label_class] = np.round(mr, 4)
        
    
    return (*rev_class_ranks, )


    
class Evaluate:
    def __init__(self, known_file, wiki_file):
        with open(known_file, 'r') as f:
            known_entities = f.readlines()
            
        self.known_entities = [i.rstrip() for i in known_entities]
        
        with open(wiki_file, 'rb') as f:
            self.wiki_text = pkl.load(f)
        
    
    
    def main(self, output_labels, test_texts, test_labels, test_label_classes, test_TP, log_file):
        if log_file:
            logging.basicConfig(filename=log_file, filemode='w', format='%(message)s')
        else:
            logging.basicConfig(format='%(message)s')
            
        logging.getLogger().setLevel(logging.INFO)
        
        print(np.asarray(output_labels)[:5])
        
        assert len(output_labels) == len(test_texts) == len(test_labels) == len(test_label_classes) == len(test_TP)
        
        ### Remove test instances without wiki text
        output_labels, test_texts, test_labels, test_label_classes, test_TP = zip(*[i for i in zip(output_labels, test_texts, test_labels, test_label_classes, test_TP) if i[-1] in self.wiki_text])
        
        assert len(output_labels) == len(test_texts) == len(test_labels) == len(test_label_classes) == len(test_TP)
        
        self.known_indices, self.unknown_indices = get_known_unknown(test_TP, self.known_entities)
        print(f'Total: {len(test_labels)}, Known: {len(self.known_indices)}, Unknown {len(self.unknown_indices)}')
        
        accurate_frac, incong_frac, nonfact_frac, both_frac = calculate_highest(output_labels, test_labels, test_label_classes)
        
        assert len(accurate_frac) == len(incong_frac) == len(nonfact_frac) == len(both_frac)
        
        logging.info(f'********** Top **********' )
        
        logging.info(
            f'Accurate: {round(np.mean(accurate_frac), 4)}, '\
            f'Incongruous: {round(np.mean(incong_frac), 4)}, '\
            f'Nonfactual: {round(np.mean(nonfact_frac), 4)}, '\
            f'Both: {round(np.mean(both_frac), 4)}'
        )
        
        accurate_frac_known, incong_frac_known, nonfact_frac_known, both_frac_known = zip(*[instance for idx, instance in enumerate(zip(accurate_frac, incong_frac, nonfact_frac, both_frac)) if idx in self.known_indices])
        
        assert len(accurate_frac_known) == len(incong_frac_known) == len(nonfact_frac_known) == len(both_frac_known)
        
        accurate_frac_unknown, incong_frac_unknown, nonfact_frac_unknown, both_frac_unknown = zip(*[instance for idx, instance in enumerate(zip(accurate_frac, incong_frac, nonfact_frac, both_frac)) if idx in self.unknown_indices])
        
        logging.info(
            f'Accurate Known: {round(np.mean(accurate_frac_known), 4)}, '\
            f'Incongruous Known: {round(np.mean(incong_frac_known), 4)}, '\
            f'Nonfactual Known: {round(np.mean(nonfact_frac_known), 4)}, '\
            f'Both Known: {round(np.mean(both_frac_known), 4)}'
        )
        
        logging.info(
            f'Accurate Unknown: {round(np.mean(accurate_frac_unknown), 4)}, '\
            f'Incongruous Unknown: {round(np.mean(incong_frac_unknown), 4)}, '\
            f'Nonfactual Unknown: {round(np.mean(nonfact_frac_unknown), 4)}, '\
            f'Both Unknown: {round(np.mean(both_frac_unknown), 4)}'
        )
     
        sig = significance_eval([accurate_frac_known, incong_frac_known, 
                          nonfact_frac_known, both_frac_known],
                          [accurate_frac_unknown, incong_frac_unknown, 
                          nonfact_frac_unknown, both_frac_unknown])
        
        logging.info(f'********** MRR **********' )
        
        accurate_mrr, incong_mrr, nonfact_mrr, both_mrr = calculate_mrr(output_labels, test_labels, test_label_classes)
        
        logging.info(
            f'Accurate: {round(np.mean(accurate_mrr), 4)}, '\
            f'Incongruous: {round(np.mean(incong_mrr), 4)}, '\
            f'Nonfactual: {round(np.mean(nonfact_mrr), 4)}, '\
            f'Both: {round(np.mean(both_mrr), 4)}'
        )
        
        accurate_mrr_known, incong_mrr_known, nonfact_mrr_known, both_mrr_known = zip(*[instance for idx, instance in enumerate(zip(accurate_mrr, incong_mrr, nonfact_mrr, both_mrr)) if idx in self.known_indices])
        
        accurate_mrr_unknown, incong_mrr_unknown, nonfact_mrr_unknown, both_mrr_unknown = zip(*[instance for idx, instance in enumerate(zip(accurate_mrr, incong_mrr, nonfact_mrr, both_mrr)) if idx in self.unknown_indices])
        
        assert len(accurate_mrr_unknown) == len(incong_mrr_unknown) == len(nonfact_mrr_unknown) == len(both_mrr_unknown)
        
        logging.info(
            f'Accurate Known: {round(np.mean(accurate_mrr_known), 4)}, '\
            f'Incongruous Known: {round(np.mean(incong_mrr_known), 4)}, '\
            f'Nonfactual Known: {round(np.mean(nonfact_mrr_known), 4)}, '\
            f'Both Known: {round(np.mean(both_mrr_known), 4)}'
        )
        
        logging.info(
            f'Accurate Unknown: {round(np.mean(accurate_mrr_unknown), 4)}, '\
            f'Incongruous Unknown: {round(np.mean(incong_mrr_unknown), 4)}, '\
            f'Nonfactual Unknown: {round(np.mean(nonfact_mrr_unknown), 4)}, '\
            f'Both Unknown: {round(np.mean(both_mrr_unknown), 4)}'
        )
        
        sig = significance_eval([accurate_mrr_known, incong_mrr_known, 
                          nonfact_mrr_known, both_mrr_known],
                          [accurate_mrr_unknown, incong_mrr_unknown, 
                          nonfact_mrr_unknown, both_mrr_unknown])
           
        