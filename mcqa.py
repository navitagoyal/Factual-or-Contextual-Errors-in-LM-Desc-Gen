#!/usr/bin/env python
# coding: utf-8


import torch
import nltk
import math
import json
import argparse
import numpy as np
import regex as re
import pickle as pkl
from tqdm import tqdm

from mcqa_dataloader import MCQA
import mcqa_eval as eval_utils

from transformers import AutoModelForMultipleChoice, AutoTokenizer
from transformers import TrainingArguments, Trainer


#### Config


config = json.load(open('config.json', 'r'))


def compute_metrics(eval_predictions):
    predictions, label_ids = eval_predictions
    preds = np.argmax(predictions, axis=1)
    return {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}


class Predictor:
    
    def __init__(self, args):
        self.args = args
    
    
    def load_model(self, model_name, model_checkpoint):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForMultipleChoice.from_pretrained(model_checkpoint)
            
        self.model = self.model.cuda()
        
        
    def init_trainer(self, train_dataset, val_dataset):
        
        log_dir = self.args.log_dir
        
        training_args = TrainingArguments(
            output_dir=self.args.checkpoint_dir, # output directory
            num_train_epochs=3,                  # total number of training epochs
            per_device_train_batch_size=2,       # batch size per device during training
            per_device_eval_batch_size=16,       # batch size for evaluation
            warmup_steps=500,                    # number of warmup steps for learning rate scheduler
            weight_decay=0.01,                   # strength of weight decay
            logging_dir=log_dir,                 # directory for storing logs
            logging_steps=100,
            save_steps=2000,
            learning_rate=2e-5,
            evaluation_strategy="steps",
            save_strategy="steps",
            # load_best_model_at_end=True,
            # metric_for_best_model="accuracy",
            report_to="tensorboard"
        )

        # model = AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased")

        self.trainer = Trainer(
            model = self.model,                    # the instantiated 🤗 Transformers model to be trained
            args = training_args,                  # training arguments, defined above
            train_dataset = train_dataset,         # training dataset
            eval_dataset = val_dataset,             # evaluation dataset
            compute_metrics = compute_metrics
        )
    
    
    def train(self):

        self.trainer.train()
        self.trainer.save_model()
    
    
    def evaluate(self, test_texts, test_labels, test_label_classes, test_TP, test_enc, test_dataset, eval_log):
        output_labels = self.trainer.predict(test_dataset).predictions
        print(test_labels[0])
        print(test_label_classes[0])
        print(len(output_labels))
        
        eval_fn = eval_utils.Evaluate(config['known_file'], config['wiki_file'])

        
        if self.args.eval_classes:
            eval_fn.main(output_labels, test_texts, test_labels, test_label_classes, test_TP, eval_log)
        else:
            eval_fn.evaluate(output_labels, test_labels)
            


def main(args):
    
    print("Torch cuda: ", torch.cuda.is_available())
    
    
    if not args.do_train:
        model_checkpoint = args.checkpoint_dir
    else:
        model_checkpoint = args.model_name
        
    
    if 'bert-base' in args.model_name:
        mask_token = '[MASK]'
    elif 'roberta' in args.model_name:
        mask_token = '<mask>'
    elif 'electra' in args.model_name:
        mask_token = '[MASK]'
        
        
    predictor = Predictor(args)
    predictor.load_model(args.model_name, model_checkpoint)
    
    stopwords = ['a', 'an', 'the', 'and']
    data = MCQA('masked_data', f'{args.task}_mcqa',
                   clipped=not args.sentence_only, 
                   max_sent=args.max_sent, 
                   total_sent=args.total_sent,
                   stopwords=stopwords, 
                   mask_token=mask_token, 
                   task=args.task)

    
    data.load(predictor.tokenizer, test_only=not args.do_train)
    
    if args.do_train:
        predictor.init_trainer(data.train.dataset, data.val.dataset)
        predictor.train()
        
    if args.do_eval:
        predictor.init_trainer(data.test.dataset, data.test.dataset)
        test_data = data.test
        predictor.evaluate(test_data.texts, test_data.labels, test_data.label_classes, test_data.TPs, test_data.texts_enc, test_data.dataset, args.eval_log)
        
    return


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    
    ### Training Config
    parser.add_argument('--model_name', type=str, 
                        default='BERT',
                        help='Pretrained model/tokenizer name')
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--task', type=str, 
                        default='desc',
                        help='desc (Description Identification); claim (Claim Identification)')

    ### Data Config
    parser.add_argument('--max_sent', type=int, default=5, 
                        help='Number of max sentences before MASK')
    parser.add_argument('--sentence_only', action='store_true', 
                        help='Only sentence or complete context as input')
    parser.add_argument('--total_sent', type=int, default=10, 
                        help='Number of total sentences')
    
    ### Checkpoint Config
    parser.add_argument('--checkpoint_dir', type=str,
                        help='checkpoint_dir')
    parser.add_argument('--log_dir', type=str, default='./logs', 
                        help='log_dir')
    
    ### Evaluation Config
    parser.add_argument('--eval_classes', action='store_true', 
                        help='Add argument to evaluate over all classes (factual/contextual)')
    parser.add_argument('--eval_log', type=str, default=None,
                        help='File to save evaluation results')

    
    args = parser.parse_args()
    
    
    if args.model_name=='BERT':
        args.model_name = 'bert-base-cased'
    elif args.model_name=='RoBERTa':
        args.model_name = 'roberta-base'
    elif args.model_name=='Electra':
        args.model_name = 'google/electra-base-discriminator'
    
    main(args)
