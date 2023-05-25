import nltk
import collections
from tqdm.auto import tqdm
import torch
import random

import utils
from dataloader import BaseProcessor
     
classes = ['accurate', 'incongruous', 'nonfactual', 'both']


class MCQAProcessor(BaseProcessor):
    def __init__(self, articles_name, options_name, clipped, max_sent, total_sent, mask_token='[MASK]'):
        self.save_dir = '../e2e_saved/'
        self.source = 'cnn'
        
        self.articles_name = articles_name
        self.options_name = options_name
        
        self.clipped = clipped
        self.max_sent = max_sent
        self.mask_token = mask_token
        self.total_sent = total_sent
        
        
    def remove_desc(self, text, desc):
        if(text.find(desc)):
            text = text.replace(desc, ' [UNK] ')
            
        return text
        
        
    def load_data(self, articles_path, options_path):
        articles_data = utils.read_jsonl_file(articles_path)
        options_data = utils.read_jsonl_file(options_path)
        
        texts, labels, label_classes, TPs, indices = [], [], [], [], []
        
        for instance in options_data:
            article_instance = articles_data[instance['idx']]
            
            assert article_instance['pers'] == instance['pers']
            # assert article_instance['label'] == instance['accurate']
            if article_instance['label'] != instance['accurate']:
                continue
            
            text = article_instance['masked_text']
            
            if self.clipped:
                text = super().clip_masked_text(text, self.max_sent, self.total_sent)
                if not text:
                    continue
                    
            text = self.remove_desc(text, instance['nonfactual'])
            if self.mask_token!='[MASK]':
                text = text.replace('[MASK]', self.mask_token)
                
            label = instance['accurate']
            options = [instance[c] for c in classes]
            x = list(enumerate(options))
            random.shuffle(x)
            order, options = zip(*x)
            
            true_label = options.index(label)
            
            texts.append([text + ' [SPC] '+ option for option in options])
            labels.append(true_label)
            label_classes.append(order)
            TPs.append(instance['pers'])
            indices.append(instance['idx'])
            
            
        return texts, labels, label_classes, TPs, indices
    
        
    def create_data(self, split):
        articles_path = f'{self.save_dir}/{self.articles_name}_{self.source}_{split}.jsonl'
        options_path = f'{self.save_dir}/{self.options_name}_{self.source}_{split}.jsonl'
        texts, labels, label_classes, TPs, indices = self.load_data(articles_path, options_path)
        
        
        # _, discard_indices = super().preprocess_masked_text([t[0] for t in texts], clipped=False, max_sent=0) ###Input text was clipped pre-appending options
        discard_indices = [idx for idx, text in enumerate(texts) \
                             if text[0]==None or self.mask_token not in text[0]]
        
        pre_filtering = len(texts)
        
        texts, labels, label_classes, TPs, indices = list(zip(*(x for idx, x in enumerate(zip(texts, labels, label_classes, TPs, indices)) if idx not in discard_indices)))
        
        print(f"Discarding {len(discard_indices)} {split} instances")
        
        assert pre_filtering == len(texts)+len(discard_indices)
        assert len(texts) == len(labels) == len(label_classes) == len(TPs) == len(indices)
        
        return texts, labels, label_classes, TPs, indices
    
    
    
class ClaimProcessor(BaseProcessor):
    def __init__(self, options_name, clipped, max_sent, total_sent, mask_token='[MASK]'):
        self.save_dir = '../e2e_saved/'
        self.source = 'cnn'
        
        self.options_name = options_name
        
        self.clipped = clipped
        self.max_sent = max_sent
        self.mask_token = mask_token
        self.total_sent = total_sent
        
        
    def load_data(self, options_path):
        options_data = utils.read_jsonl_file(options_path)
        
        texts, labels, label_classes, TPs, indices = [], [], [], [], []
        
        for idx, instance in enumerate(options_data):
            
            text = instance['masked_text']
            if not text:
                continue
            
            if self.clipped:
                text = super().clip_masked_text(text, self.max_sent, self.total_sent)
                if not text:
                    continue
                    
            if self.mask_token!='[MASK]':
                text = text.replace('[MASK]', self.mask_token)
                
            label = ', '.join(instance['accurate'])
            options = [', '.join(instance[c]) for c in classes]
            x = list(enumerate(options))
            random.shuffle(x)
            order, options = zip(*x)
            
            true_label = options.index(label)
            
            texts.append([text + ' [SEP] '+ option for option in options])
            labels.append(true_label)
            label_classes.append(order)
            TPs.append(instance['pers'])
            indices.append(idx)
            
            
        return texts, labels, label_classes, TPs, indices
    
        
    def create_data(self, split):
        options_path = f'{self.save_dir}/{self.options_name}_{self.source}_{split}.jsonl'
        texts, labels, label_classes, TPs, indices = self.load_data(options_path)
        
        
        # _, discard_indices = super().preprocess_masked_text([t[0] for t in texts], clipped=False, max_sent=0) ###Input text was clipped pre-appending options
        discard_indices = [idx for idx, text in enumerate(texts) \
                             if text[0]==None or self.mask_token not in text[0]]
        
        pre_filtering = len(texts)
        
        texts, labels, label_classes, TPs, indices = list(zip(*(x for idx, x in enumerate(zip(texts, labels, label_classes, TPs, indices)) if idx not in discard_indices)))
        
        print(f"Discarding {len(discard_indices)} {split} instances")
        
        assert pre_filtering == len(texts)+len(discard_indices)
        assert len(texts) == len(labels) == len(label_classes) == len(TPs) == len(indices)
        
        return texts, labels, label_classes, TPs, indices
        
        


class OptionsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class MCQALoader:
    
    def __init__(self, data, tokenizer):
        self.tokenizer = tokenizer
        self.mask_token_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
        self.load_data(data)
        
        
    def encode(self, texts):
        return self.tokenizer(texts, truncation=True, padding=True)
    
    
    def encode_texts(self, texts):
    
        # Flatten everything
        texts = sum(texts, [])
        num_options = 4

        # Tokenize
        tokenized_examples = self.encode(texts)

        # Un-flatten
        return {k: [v[i:i+num_options] for i in range(0, len(v), num_options)] for k, v in tokenized_examples.items()}
    
    
    def filter_data(self):
        ### Remove data without [MASK] token
        subset = [idx for idx, instance in enumerate(self.texts_enc['input_ids']) if self.mask_token_id in instance[0]]
        self.texts, self.labels, self.label_classes, self.TPs, self.indices = zip(*[(self.texts[idx], self.labels[idx], self.label_classes[idx], self.TPs[idx], self.indices[idx]) for idx in subset])
        
    
    def load_enc_data(self):
        self.texts_enc = self.encode_texts(self.texts)
        
        ## Remove inputs without [MASK] token
        self.filter_data()
        
        ## Update text and label encodings
        self.texts_enc = self.encode_texts(self.texts)
                
        self.dataset = OptionsDataset(self.texts_enc, self.labels)
        
        
    def load_data(self, data):
        self.texts, self.labels, self.label_classes, self.TPs, self.indices = data
        self.load_enc_data()
    
        
class MCQA:
    def __init__(self, articles_name=None, options_name=None, clipped=False, max_sent=5, total_sent=5, stopwords=[], mask_token='[MASK]', task='desc'):
        if task=='desc':
            self.data_proc = MCQAProcessor(articles_name, options_name, clipped, max_sent, total_sent, mask_token)
        elif task=='claim':
            self.data_proc = ClaimProcessor(options_name, clipped, max_sent, total_sent, mask_token)
        
    
    def load(self, tokenizer, test_only=False):
        self.test = MCQALoader(self.data_proc.create_data('test'), tokenizer)
        
        if test_only:
            return
        
        self.val = MCQALoader(self.data_proc.create_data('valid'), tokenizer)
        self.train = MCQALoader(self.data_proc.create_data('train'), tokenizer)