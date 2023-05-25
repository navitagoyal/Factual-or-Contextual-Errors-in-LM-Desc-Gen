import nltk
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
# , SmoothingFunction
# import evaluate
from torchmetrics.functional.text.bert import bert_score
import os
import collections
import numpy as np

from transformers import logging
logging.set_verbosity_error()

rouge = Rouge()
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def intersection_over_union(true_label, pred_label):
    true_tokens = nltk.word_tokenize(true_label)
    pred_tokens = nltk.word_tokenize(pred_label)
    
    overlap = list(set(true_tokens).intersection(pred_tokens))
    union = list(set(true_tokens).union(pred_tokens))
    
    if len(union)==0:
        return 0
    
    return len(overlap)/len(union)


def unigram_precision(true_label, pred_label):
    true_label = true_label.lower()
    pred_label = pred_label.lower()
        
    true_tokens = nltk.word_tokenize(true_label)
    pred_tokens = nltk.word_tokenize(pred_label)
    
    overlap = list(set(true_tokens).intersection(pred_tokens))
    # union = list(set(true_tokens).union(pred_tokens))
    total = list(set(pred_tokens))
    
    if len(total)==0:
        return 0
    
    return len(overlap)/len(total)


def rouge_L(true_label, pred_label):
    true_label = true_label.lower()
    pred_label = pred_label.lower()
            
    score = rouge.get_scores(pred_label, true_label)
    return score[0]['rouge-l']['p']


def modified_unigram_precision(true_label, pred_label):
    true_label = true_label.lower()
    pred_label = pred_label.lower()
        
    true_tokens = nltk.word_tokenize(true_label)
    pred_tokens = nltk.word_tokenize(pred_label)
    
    if len(pred_tokens)==0:
        return 0
    
    pred_count = collections.Counter(pred_tokens)
    true_count = collections.Counter(true_tokens)
    
    clipped_count = collections.defaultdict()
    
    for token in pred_count:
        if token in true_count:
            clipped_count[token] = min(pred_count[token], true_count[token])
        else:
            clipped_count[token] = 0
            
    overlap = np.sum([v for k,v in clipped_count.items()])
    
    return overlap/len(pred_tokens)


def bleu(true_label, pred_label):
    true_label = true_label.lower()
    pred_label = pred_label.lower()
        
    # true_tokens = nltk.word_tokenize(true_label)
    # pred_tokens = nltk.word_tokenize(pred_label)
    
    score = sentence_bleu([[true_label]], [pred_label], weights=(1, 0, 0, 0))

    return score
    


def bleurt(true_label, pred_label):
    scorer = evaluate.load("bleurt", module_type="metric")
    
    score = scorer.compute(predictions = [pred_label], 
                           references = [true_label])
    
    return score['scores'][0]


def wer(true_label, pred_label):
    scorer = evaluate.load("wer")
    
    score = scorer.compute(predictions = [pred_label], 
                           references = [true_label])
    
    return score


def bertscore(true_label, pred_label):
    # scorer = evaluate.load("bertscore")
    # scorer = BERTScore()
    
    score = bert_score([pred_label], [true_label], model_name_or_path='bert-base-uncased')
    
    return score['precision']