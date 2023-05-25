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

from transformers import Trainer, TrainingArguments
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

from dataloader import DescGen
import eval_utils

config = json.load(open('config.json', 'r'))


class Generator:
    
    def __init__(self, args):
        self.args = args
    
    
    def load_model(self, model_name, model_checkpoint):
        if "t5" in model_name:
            self.tokenizer = T5Tokenizer.from_pretrained(model_name, mask_token='[MASK]')
            self.model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)
        elif "bart" in model_name:
            self.tokenizer = BartTokenizer.from_pretrained(model_name)
            self.model = BartForConditionalGeneration.from_pretrained(model_checkpoint)
            self.model.resize_token_embeddings(len(self.tokenizer))
        elif "pegasus" in model_name:
            self.tokenizer = PegasusTokenizer.from_pretrained(model_name, mask_token='[MASK]')
            self.model = PegasusForConditionalGeneration.from_pretrained(model_checkpoint)
        else:
            print("Choose a valid model")
            
        self.model = self.model.cuda()
    
    
    def train(self, train_dataset, val_dataset):
        
        training_args = TrainingArguments(
            output_dir=self.args.checkpoint_dir, # output directory
            num_train_epochs=3,                  # total number of training epochs
            per_device_train_batch_size=1,       # batch size per device during training
            per_device_eval_batch_size=64,       # batch size for evaluation
            warmup_steps=500,                    # number of warmup steps for learning rate scheduler
            weight_decay=0.01,                   # strength of weight decay
            logging_dir=self.args.log_dir,       # directory for storing logs
            logging_steps=100,
            save_steps=5000
        )

        # model = AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased")

        trainer = Trainer(
            model = self.model,                    # the instantiated 🤗 Transformers model to be trained
            args = training_args,                  # training arguments, defined above
            train_dataset = train_dataset,         # training dataset
            eval_dataset = val_dataset             # evaluation dataset
        )
        
        trainer.train()
        trainer.save_model()
    
    
    
    def batch_generate(self, batch_inputs):
        output_sequences = self.model.generate(
            input_ids = batch_inputs,
            # attention_mask = torch.tensor(test_encodings['attention_mask']),
            do_sample = False, # disable sampling to test if batching affects output
        )

        batch_output = self.tokenizer.batch_decode(output_sequences, skip_special_tokens=True)

        return batch_output
    
    
    def generate(self, test_enc_input):
        batch_size = 10
        
        num_examples = len(test_enc_input)
        num_batches = math.ceil(num_examples/batch_size)

        output_labels = []
        for batch in tqdm(range(num_batches)):
            start_idx = batch*batch_size
            end_idx = min((batch+1)*batch_size, num_examples)

            batch_inputs = torch.tensor(test_enc_input[start_idx:end_idx]).to('cuda:0')
            batch_output = self.batch_generate(batch_inputs)
            output_labels += batch_output

        return output_labels
    
    
    def evaluate(self, test_texts, test_labels, test_TP, test_enc, log_file):
        output_labels = self.generate(test_enc['input_ids'])
        
        eval_fn = eval_utils.Evaluate(config['known_file'], config['wiki_file'])

        
        if self.args.eval_classes:
            eval_fn.main(output_labels, test_texts, test_labels, test_TP, log_file)
        else:
            eval_fn.evaluate(output_labels, test_labels)
            
        
        if self.args.eval_familiarity:
            eval_fns.evaluate_familiarity(output_labels, test_texts, test_labels, test_TP)
            
            

def main(args):
    
    print("Torch cuda: ", torch.cuda.is_available())
    
    
    if not args.do_train:
        model_checkpoint = args.checkpoint_dir
    else:
        model_checkpoint = args.model_name
        
        
    if 't5' in args.model_name:
        mask_token = '[MASK]'
    elif 'bart' in args.model_name:
        mask_token = '<mask>'
    elif 'pegasus' in args.model_name:
        mask_token = '[MASK]'
        
    generator = Generator(args)
    generator.load_model(args.model_name, model_checkpoint)
    
    stopwords = ['a', 'an', 'the', 'and']
    data = DescGen('masked_data', 
                   clipped=not args.sentence_only, 
                   max_sent=args.max_sent, 
                   stopwords=stopwords, mask_token=mask_token)
    
    data.load(generator.tokenizer, test_only=not args.do_train)
    
    if args.do_train:
        generator.train(data.train.dataset, data.val.dataset)
    if args.do_eval:
        test_data = data.test
        generator.evaluate(test_data.texts, test_data.labels, test_data.TPs, test_data.texts_enc, args.eval_log)
        
    return


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    
    ### Training Config
    parser.add_argument('--model_name', type=str, default='T5-small', 
                        help='Pretrained model/tokenizer name: t5-small or bart-base')
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_eval', action='store_true')

    ### Data Config
    parser.add_argument('--max_len', type=int, default=200, 
                        help='Maximum context length')
    parser.add_argument('--max_sent', type=int, default=5, 
                        help='Number of max sentences')
    parser.add_argument('--sentence_only', action='store_true', 
                        help='Only sentence or complete context as input')
    
    ### Checkpoint Config
    parser.add_argument('--checkpoint_dir', type=str, required=True, 
                        help='checkpoint_dir')
    parser.add_argument('--log_dir', type=str, default='./logs', 
                        help='log_dir')
    
    ### Evaluation Config
    parser.add_argument('--eval_classes', action='store_true', 
                        help='Add argument to evaluate over all classes (factual/contextual)')
    parser.add_argument('--eval_familiarity', action='store_true', 
                        help='Add argument to evaluate over familiarity/unfamiliarity subsets')
    parser.add_argument('--eval_log', type=str, default=None,
                        help='File to save evaluation results')

    
    args = parser.parse_args()
    
    if args.model_name=='T5-small':
        args.model_name = 't5-small'
    elif args.model_name=='T5-base':
        args.model_name = 't5-base'
    elif args.model_name=='BART':
        args.model_name = 'sshleifer/distilbart-cnn-6-6'
    
    main(args)