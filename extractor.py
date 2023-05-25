import json
import collections
import pickle as pkl
import argparse
from tqdm.auto import tqdm

import utils
import load_articles
import ner_utils
import parser_utils, desc_utils

dep_parser = parser_utils.Parser()

class ExtractorUtils:
    def __init__(self):
        pass
    
    
    def load_pers(self, pers_tags=None):
        if not pers_tags:
            tagger = ner_utils.NerTagger()
            ner_tags = tagger.tag(sentence)
            pers_tags = ner_utils.extract_person_ner_tags(ner_tags)
            
        self.pers_lastnames = [tag[-1] for tag in pers_tags]
        _, self.pers_entities_key = ner_utils.get_alias(pers_tags) #for desc mapping to aliases
    
    
    def mask_sentence(self, text, desc, name):
        # text = " ".join(word_tokenize(text))
        # text = " - ".join(text.split('-'))
        
        if desc not in text:
            # raise Exception(f"Description {desc} not found in {text}")
            print(f"\x1b[31m\"Description {desc} not found\"\x1b[0m")
            
            
    def extract_pers(self, name):
        if name in self.pers_entities_key:
            return self.pers_entities_key[name]
        
        
        ### If name not in alias list then find names that have name as a subname
        ### If there is more than one name with the same subname, then skip
        candidate_pers = [fullname for _, fullname in self.pers_entities_key.items() if name in fullname]

        if len(list(set(candidate_pers)))==1:
            pers = self.pers_entities_key[candidate_pers[0]]
        else:
            # print(f"\x1b[31m\"Name {name} not found in {', '.join(self.pers_entities_key)}\"\x1b[0m")
            return None
        
        return pers
            
            
    def extract_desc(self, token, text):
        desc, name = dep_parser.parse_description(token, flag='root')
        name = f'{name}{str(token.text)}'
        name = desc_utils.clean_extracted_desc(name)

        desc = desc_utils.clean_desc(desc)
        desc_wo_name = desc_utils.extract_desc(desc, name)

        if not desc_wo_name:
            return [None] * 3
        else:
            desc_wo_name = desc_utils.clean_desc(desc_wo_name)

        if desc not in text:
            # print(f"\x1b[31m\"Description {desc} not found\"\x1b[0m")
            return [None] * 3
        
        return desc, desc_wo_name, name
            
    
    def main(self, text, pers_tags=None):
        self.text = text
        self.load_pers(pers_tags)
        self.parse = dep_parser.dep_parsing(text)
        
        pers_labels = collections.defaultdict(lambda: {'span': [], 'desc': [], 'alias': []})
        
        for token in self.parse:
            if str(token.text) not in self.pers_lastnames:
                continue
                
            desc, desc_wo_name, name = self.extract_desc(token, text)
            if not desc:
                continue

            pers = self.extract_pers(name)
            
            if not pers:
                continue
            
            if desc not in pers_labels[pers]['span']:
                pers_labels[pers]['span'].append(desc)
                pers_labels[pers]['alias'].append(name)
                
            pers_labels[pers]['desc'].append(desc_wo_name)
            
                
        return pers_labels
    
    
class Extractor(ExtractorUtils):
    def __init__(self, articles, ner_tags):
        self.articles = articles
        self.ner_tags = ner_tags
    
    
    def extract(self, idx):
        text = self.articles[idx]
        pers_tags = ner_utils.extract_person_ner_tags(self.ner_tags[idx])
        desc_data = super().main(text, pers_tags)
        
        jsonl_data = self.convert2jsonl(idx, dict(desc_data))

        return jsonl_data
    
    
    def convert2jsonl(self, idx, desc_data):
        jsonl_data = []
        for pers, data in desc_data.items():
            jsonl_data.append(
                {
                    'idx': idx, 
                    'pers': pers, 
                    'spans': data['span'], 
                    'descriptions': data['desc'], 
                    'aliases': data['alias']
                }
            )
            
        return jsonl_data
    
    
    def main(self, save_name):        
        with open(save_name, 'w') as save_file:
            for idx in tqdm(range(len(self.articles))):
                out_data = self.extract(idx)
                
                for line in out_data:
                    json.dump(line, save_file)
                    save_file.write('\n')
            
        # pkl.dump(desc_data, open(save_name, 'wb'))
    
    
if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='cnn', 
                        help='cnn/dm')
    parser.add_argument('--split', type=str, default='valid', 
                        help='train/test/valid split')
    parser.add_argument('--save_name', type=str, required=True, 
                        help='directory path and file name for saving description data')
    
    save_dir = '../e2e_saved/'
    
    args = parser.parse_args()
    
    articles = load_articles.main(args.source, args.split)
    ner_tags = ner_utils.load_ner_tags(args.source, args.split)
    
    articles, ner_tags = utils.remove_duplicates(articles, ner_tags)
    
    extractor = Extractor(articles, ner_tags)
    
    save_path = f'{save_dir}/{args.save_name}_{args.source}_{args.split}.jsonl'
    extractor.main(save_path)