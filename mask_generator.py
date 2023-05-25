import json
import argparse
import regex as re
from tqdm.auto import tqdm

import utils
import load_articles
import ner_utils


def check_aliases(span, aliases):
    ### Check if span is in alias or a substring of the alias
    if span in aliases:
        return 1
    
    for alias in aliases:
        if len(list(set(span.split(" ")) & set(alias.split(" ")))) == len(list(set(span.split(" ")))):
            return 1
        
    
    return 0


def remove_consecutive_mask_token(text, aliases):
    ### collate multiple MASK into single
    text = text.split('[MASK]')
    flag = text[0]=='' ### Does the text start with a MASK token
    
    text = "[MASK]".join([i.strip() for i in text if i.strip()!=''])
    # text = re.sub(r'\[MASK\]\[MASK\]', r'\[MASK\]', text)
    if flag==1:
        text = f'[MASK] {text}'

    chunked = text.split('[MASK]')

    updated_text = chunked[0]

    idx = 1
    prev_desc = ''
    curr_desc = ''

    while idx<len(chunked):
        curr_desc = f'{prev_desc} {chunked[idx]}'
        curr_desc = curr_desc.strip()

        curr = check_aliases(curr_desc, aliases)
        prev = check_aliases(prev_desc, aliases)
        
        if curr == 0 and (prev_desc=='' or prev==1):
            updated_text = f'{updated_text} [MASK] {prev_desc} [MASK]'
            prev_desc = chunked[idx]
        elif curr == 0:
            updated_text = f'{updated_text} {prev_desc}'
            prev_desc = chunked[idx]
        else:
            prev_desc = curr_desc
            
        idx += 1

    updated_text = f'{updated_text} {prev_desc}'

    return updated_text.strip()


class MaskDescEntity:
    
    ### Place the same alias for all occurences of a given span
    def remove_span(self, text, span, alias):
        #### span: full span including the name of the person
        #### alias: the name of person appearing in the description
        
        if span not in text:
            # print(f"\x1b[31m\"Description {span} not found in {text}\"\x1b[0m")
            return text
        
        text = text.split(span)
        updated_text = text[0]

        for t in text[1:]:
            updated_text += f'{alias} {t}'
        
        return updated_text
    
    
    ### Remove all spans by placing the same alias for all occurences of a given span (unique)
    def remove_all_spans(self, text, spans, aliases):
        #### Arrange spans in descending order first so that the subspans are removed later
        spans, aliases = list(zip(*sorted(list(zip(spans, aliases)), key = lambda x: len(x[0]), reverse=True)))
        
        ### Remove spans one at a time
        for span, alias in zip(spans, aliases):
            text = self.remove_span(text, span, alias)
            
        return text
                
    
    def place_mask(self, text, alias):
        text = text.split(alias)
        
        updated_text = text[0]

        for t in text[1:]:
            updated_text += f'[MASK] {alias} [MASK] {t}'
                
        return updated_text
            
    
    def place_all_masks(self, text, aliases):
        for alias in aliases:
            text = self.place_mask(text, alias)
            
        return text
    
    
    def clean_masked_text(self, text, aliases):
        text = remove_consecutive_mask_token(text, aliases)
        
        ### Remove consecutive white spaces
        text = re.sub(r'\s\s+', ' ', text)
        return text
            
    
    def get_label(self, descriptions):
        ### Predict first mention only
        return descriptions[0]
        
        ### Return largest mention
        # return sorted(descriptions, key=len, reverse=True)[0]
    
    
    def main(self, text, spans, descriptions, aliases_w_desc, aliases):
        text = self.remove_all_spans(text, spans, aliases_w_desc)
        text = self.place_all_masks(text, aliases)
        text = self.place_all_masks(text, aliases_w_desc)
        text = self.clean_masked_text(text, aliases)
        label = self.get_label(descriptions)
        
        return text, label
    
    
class MaskDesc(MaskDescEntity):
    def __init__(self, articles, ner_tags):
        self.articles = articles
        self.ner_tags = ner_tags
    
    
    def load_pers(self, pers_tags=None):
        if not pers_tags:
            tagger = ner_utils.NerTagger()
            ner_tags = tagger.tag(sentence)
            pers_tags = ner_utils.extract_person_ner_tags(ner_tags)
            
        ## alias to fullname and fullname to alias mapping    
        _, pers_entities_alias2fname = ner_utils.get_alias(pers_tags) #for desc mapping to aliases
        pers_entities_fname2alias = ner_utils.get_reverse_alias(pers_entities_alias2fname)
        self.pers_entities_fname2alias = {key: sorted(value, key=len, reverse=True) for key, value in pers_entities_fname2alias.items()}
      
    
    def mask_single(self, data):
        idx = data['idx']
        text = self.articles[idx]
        pers_tags = ner_utils.extract_person_ner_tags(self.ner_tags[idx])
        self.load_pers(pers_tags)
        
        pers = data['pers']
        spans, descriptions, aliases_w_desc = data['spans'], data['descriptions'], data['aliases']
        aliases = self.pers_entities_fname2alias[pers]
        
        masked_text, label = super().main(text, spans, descriptions, aliases_w_desc, aliases)
        
        return self.convert2jsonl(idx, masked_text, label, pers)
    
    
    def convert2jsonl(self, idx, masked_text, label, pers):
        return {
                    'idx': idx, 
                    'pers': pers, 
                    'masked_text': masked_text, 
                    'label': label
        }    
    
    
    def main(self, desc_data, save_name):
        with open(save_name, 'w') as save_file:
            for idx in tqdm(range(len(desc_data))):
                out_data = self.mask_single(desc_data[idx])
                json.dump(out_data, save_file)
                save_file.write('\n')
    
if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='cnn', 
                        help='cnn/dm')
    parser.add_argument('--split', type=str, default='valid', 
                        help='train/test/valid split')
    parser.add_argument('--input_name', type=str, required=True, 
                        help='directory path and file name for loading description data')
    parser.add_argument('--save_name', type=str, required=True, 
                        help='directory path and file name for saving masked data')
    
    save_dir = '../e2e_saved/'
    
    args = parser.parse_args()
    
    articles = load_articles.main(args.source, args.split)
    ner_tags = ner_utils.load_ner_tags(args.source, args.split)
    
    articles, ner_tags = utils.remove_duplicates(articles, ner_tags)
    
    mask_generator = MaskDesc(articles, ner_tags)
    
    desc_data = utils.read_jsonl_file(f'{save_dir}/{args.input_name}_{args.source}_{args.split}.jsonl')
    
    save_path = f'{save_dir}/{args.save_name}_{args.source}_{args.split}.jsonl'
    mask_generator.main(desc_data, save_path)
    
        