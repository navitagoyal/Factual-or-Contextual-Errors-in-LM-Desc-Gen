import nltk
import collections
from tqdm.auto import tqdm
import torch
import utils


class BaseProcessor:
    def __init__(self):
        pass
    
    
    def load_masked_data(self, save_path):
        data = utils.read_jsonl_file(save_path)
        # flattened_data = {'text': [], 'label': [], 'pers': [], 'pers':[]}
        flattened_data = {key:[] for key in data[0].keys()}

        for instance in data:
            for key, value in instance.items():
                flattened_data[key].append(value)
                
        texts = flattened_data['masked_text']
        labels = flattened_data['label']
        TPs = flattened_data['pers']
        indices = flattened_data['idx']
                
        return texts, labels, TPs, indices
    
    
    def clip_masked_text(self, text, max_sent_before, max_sent_after=None):
        sentences = nltk.sent_tokenize(text)
        first_mention = None
        for sent_num, sent in enumerate(sentences):
            if '[MASK]' in sent:
                first_mention = sent_num
                break
                
        if first_mention == None:
            return None
        
        if first_mention > max_sent_before:
            sentences = sentences[first_mention-max_sent_before:]
            
        if max_sent_after:
            sentences = sentences[:max_sent_after]
            
        text = " ".join(sentences)
        return text
    
    
    def preprocess_masked_text(self, texts, clipped, max_sent):
        if clipped:
            updated_texts = []
            for text in tqdm(texts):
                updated_texts.append(self.clip_masked_text(text, max_sent))
        else:
            updated_texts = texts
            
        discard_indices = [idx for idx, text in enumerate(updated_texts) \
                             if text==None or '[MASK]' not in text]
        
        return updated_texts, discard_indices
            


class DescGenProcessor(BaseProcessor):
    def __init__(self, save_name, clipped, max_sent, mask_token='[MASK]'):
        self.save_dir = '../e2e_saved/'
        self.source = 'cnn'
        
        self.save_name = save_name
        self.clipped = clipped
        self.max_sent = max_sent
        
        self.mask_token = mask_token
    
    
    def create_data(self, split):
        save_path = f'{self.save_dir}/{self.save_name}_{self.source}_{split}.jsonl'
        texts, labels, TPs, indices = super().load_masked_data(save_path)
        
        updated_texts, discard_indices = super().preprocess_masked_text(texts, 
                                                                        self.clipped, 
                                                                        self.max_sent)
        
        pre_filtering = len(texts)
        
        
        texts, labels, TPs, indices = list(zip(*(x for idx, x in enumerate(zip(updated_texts, labels, TPs, indices)) if idx not in discard_indices)))
        
        print(f"Discarding {len(discard_indices)} {split} instances")
        
        assert pre_filtering == len(texts)+len(discard_indices)
        
        if self.mask_token!='[MASK]':
            texts = [t.replace('[MASK]', self.mask_token) for t in texts]
        
        return texts, labels, TPs, indices
        
        
#     def main(self):
#         valid_data = self.load_data('valid')
#         test_data = self.load_data('test')
#         train_data = self.load_data('train')
        
#         return 


class GenerationDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class DescGenLoader:
    
    def __init__(self, data, tokenizer):
        self.tokenizer = tokenizer
        self.mask_token_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
        self.load_data(data)
        
        
    def encode(self, texts):
        return self.tokenizer(texts, truncation=True, padding=True) #, max_length=200)
        
        
    # def get_data(self):
    #     return self.texts, self.TPs, self.labels, self.label_classes, self.indices
    
    
    def filter_data(self):
        ### Remove data without [MASK] token
        subset = [idx for idx, instance in enumerate(self.texts_enc['input_ids']) if self.mask_token_id in instance]
        self.texts, self.labels, self.TPs = zip(*[(self.texts[idx], self.labels[idx], self.TPs[idx]) for idx in subset])
        
    
    def load_enc_data(self):
        self.texts_enc = self.encode(self.texts)
        
        ## Remove inputs without [MASK] token
        self.filter_data()
        
        ## Update text and label encodings
        self.texts_enc = self.encode(self.texts)
        self.labels_enc = self.encode(self.labels).input_ids
        
        self.dataset = GenerationDataset(self.texts_enc, self.labels_enc)
        
        
    def load_data(self, data):
        self.texts, self.labels, self.TPs, self.indices = data
        # self.data_proc.create_data(split)
        self.load_enc_data()
        

class DescGen:
    
    def __init__(self, save_name, clipped=False, max_sent=5, stopwords=[], mask_token='[MASK]'):
        self.data_proc = DescGenProcessor(save_name, clipped, max_sent, mask_token)
        
    
    def load(self, tokenizer, test_only=False):
        self.test = DescGenLoader(self.data_proc.create_data('test'), tokenizer)
        
        if test_only:
            return
        
        self.val = DescGenLoader(self.data_proc.create_data('valid'), tokenizer)
        self.train = DescGenLoader(self.data_proc.create_data('train'), tokenizer)
        
