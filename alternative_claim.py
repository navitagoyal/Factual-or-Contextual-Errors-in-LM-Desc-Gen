import nltk
import json
import random
import argparse
import collections
import regex as re
import numpy as np
import pickle as pkl
from tqdm import tqdm

from textblob import TextBlob

import utils
import ner_utils
import load_articles
from mask_generator import MaskDescEntity


def read_data(source, split):
    pomo_data = '../Data/PoMo/dataset/'+split+'.pm'    
    
    with open(pomo_data, 'r') as f:
        data = f.readlines()
        
    data = [i.split('\t') for i in data]
    
    if source:
        data = list(filter(lambda x: x[-1].startswith(source), data))
    else: 
        data = list(filter(lambda x: x[-1].startswith('cnn') or x[-1].startswith('dm'), data)) ## Filter data from CNN and Daily Mail
        
    return data


def get_article(filename):
    filename = load_articles.get_name_from_id(filename)
    article = load_articles.get_article(f'{data_path}{filename}')        
    article = load_articles.clean(article)
    article_wo_spaces = load_articles.remove_tokenized_spaces(article)
    return article_wo_spaces


def mask_desc(article, desc):
    return re.sub(desc, " ", article)



def get_attr_frequency(data):
    attr_frequency = collections.defaultdict(int)
    for story_data in data:
        wiki_id = story_data[4]
        story_id = story_data[-1].strip()

        wiki_entry = wiki_data[wiki_id]

        for prop in wiki_entry['properties']:
            if story_id in prop['used']:
                attr_frequency[prop['property'][0]] += 1
                
    return attr_frequency



def get_wiki_data():
    wiki = []
    for wiki_split in ['train', 'test', 'valid']:
        wiki_path = '../Data/PoMo/dataset/'+wiki_split+'.wiki' 
        with open(wiki_path, 'r') as f:
            wiki += f.readlines()

    wiki_data = collections.defaultdict()
    keys = ['name', 'aliases', 'descriptions', 'properties']
    for entry in wiki:
        entity_id = entry.split('\t')[0]
        entity_data = entry.split('\t')[1:]

        curr_data = {}
        for key, value in zip(keys, entity_data):
            curr_data[key] = value

        wiki_data[entity_id] = curr_data

    for idx, entry in wiki_data.items():
        entry['properties'] = json.loads(entry['properties'])
        
    return wiki_data


def get_wiki_aliases(wiki_data):

    wiki_name_to_id = {value['name']:key for key, value in wiki_data.items()}

    wiki_name_to_id = {}
    for idx, entry in wiki_data.items():
        wiki_name_to_id[entry['name']] = idx
        for alias in entry['aliases'].split(','):
            wiki_name_to_id[alias] = idx


    return wiki_name_to_id


def sample_relevant(example):
    
    ### True description
    true = TextBlob(example['desc'])
    ngram1 = [" ".join(x) for x in true.ngrams(n=1)]
    ngram2 = [" ".join(x) for x in true.ngrams(n=2)]
    
    ### Candidate claims that are relevant to the true description
    wiki_entry = wiki_data[example['wiki_id']]
    candidates = list(filter(lambda x: example['story_id'] in x['used'], wiki_entry['properties']))
    candidates = [x['property'] for x in candidates]
    
    ### Select the property with highest n-gram overlap
    match_max, match_idx = 0, 0
    for idx, prop in enumerate(candidates):
        target = TextBlob(prop[1])
        target_ngram1 = [" ".join(x) for x in target.ngrams(n=1)]
        target_ngram2 = [" ".join(x) for x in target.ngrams(n=2)]
        
        if len(ngram2)==0:
            overlap = len(set(ngram1) & set(target_ngram1))/(len(set(ngram1).union(set(target_ngram1))))
        else:
            overlap1 = len(set(ngram1) & set(target_ngram1))/(len(set(ngram1).union(set(target_ngram1))))
            overlap2 = len(set(ngram2) & set(target_ngram2))/(len(set(ngram2).union(set(target_ngram2))))
            overlap = overlap1 + overlap2
            
        if match_max < overlap:
            match_max = overlap
            match_idx = idx
            
    return candidates[match_idx]
    
    
def sample_irrelevant(example, attr_frequency):
    
    wiki_entry = wiki_data[example['wiki_id']]
    candidates = list(filter(lambda x: example['story_id'] not in x['used'], wiki_entry['properties']))
    candidates = [x['property'] for x in candidates]
    
    if len(candidates)==0:
        return None
        
    ### sample from irr_prop randomly
    # irr_prop = random.sample(candidates, 1)[0]
    
    ### weighted sampling by the frequency of the attribute
    prob_distr = [attr_frequency[prop[0]] for prop in candidates]
    coice_idx = np.random.choice(range(len(candidates)), 1, prob_distr)[0]
    irr_prop = candidates[coice_idx]  
    
    return irr_prop


def sample_other_person(example, attr_frequency, ner_tag):
    full_names, _ = ner_utils.get_alias(ner_utils.extract_person_ner_tags(ner_tag))
    
    ### Other people in article without any overlap with the target person
    other_pers = [pers for pers in full_names \
                  if len(set(nltk.word_tokenize(example['pers'])) & set(nltk.word_tokenize(pers)))==0]
    
    ### Subset of other people in article for which we can find a wiki entry
    other_pers = [pers for pers in other_pers if pers in wiki_name_to_id.keys()]
    other_pers_dist = [np.sum([len(prop['used']) for prop in wiki_data[wiki_name_to_id[pers]]['properties']]) for pers in other_pers]
    
    if len(other_pers)==0:
        return None, None
    
    # sample_pers = random.sample(other_pers, 1)[0]
    sample_pers_idx = np.random.choice(range(len(other_pers)), 1, other_pers_dist)[0]
    sample_pers = other_pers[sample_pers_idx]
    
    sample_pers_wiki = wiki_data[wiki_name_to_id[sample_pers]]
    
    candidates = [x['property'] for x in sample_pers_wiki['properties']]
    prob_distr = [attr_frequency[prop[0]] for prop in candidates]
    
    coice_idx = np.random.choice(range(len(candidates)), 1, prob_distr)[0]
    sample_pers_prop = candidates[coice_idx]
    
    return sample_pers, sample_pers_prop
    
    
    
def sample_random(example, attr_frequency, relevant_property):
    candidates = [x for x in all_properties if len(set(x[1].split(" ")) & set(relevant_property[1].split(" ")))==0]
    if len(candidates)==0:
        return None
        
    prob_distr = [attr_frequency[prop[0]] for prop in candidates]
    
    # random_property = random.sample(random_properties, 1, candidate_dist)[0]
    coice_idx = np.random.choice(range(len(candidates)), 1, prob_distr)[0]
    random_prop = candidates[coice_idx]
    
    return random_prop
    

def create_example(idx, story_data, attr_frequency, ner_tag):
    
    example = {
        # 'text': story_data[0],
        'pers': story_data[1],
        'desc': story_data[2],
        'wiki_id': story_data[4],
        'story_id': story_data[-1].strip()
    }
    
    # example['text'] = get_article(example['story_id'])
    # example['pers'] = example['pers'].replace('.', '')
    
    # input_text = mask_postmod(get_article(story_id), appos)
    # example['wiki'] = wiki_data[example['wiki_id']]
    
    ### Extract options from wiki data
    relevant_property = sample_relevant(example)                                    ##accurate
    
    irrelevant_property = sample_irrelevant(example, attr_frequency)                ##incongruous, but factual
    if not irrelevant_property:
        return None
    
    other_pers, other_person_property = sample_other_person(example, attr_frequency, ner_tag)   ##nonfactual, but contextual
    if not other_person_property:
        return None
    
    
    ### Sample random property with non-overlapping value
    random_property = sample_random(example, attr_frequency, relevant_property)     ##both nonfactual and incongruous
    if not random_property:
        return None
    
    
    example['accurate'] = relevant_property
    example['incongruous'] = irrelevant_property
    example['nonfactual'] = other_person_property
    example['both'] = random_property
    example['other_pers'] = other_pers
    example['article_num'] = idx
    
    return example


def update_json(save_file, claim, text):
    claim['masked_text'] = text
    json.dump(claim, save_file)
    save_file.write('\n')


def create_masked_data(claim, ner_tags):
    text = get_article(claim['story_id'])
    ner_tag = ner_tags[claim['article_num']]
    
    fnames, alias_to_fname = ner_utils.get_alias(ner_utils.extract_person_ner_tags(ner_tag))
    fname_to_alias = ner_utils.get_reverse_alias(alias_to_fname)
    if claim['pers'] not in fname_to_alias:
        return None
        # raise Exception(f"Name {claim['pers']} not found in {fname_to_alias}")

    text = mask_utils.place_all_masks(text, fname_to_alias[claim['pers']])
    text = mask_desc(text, load_articles.remove_tokenized_spaces(claim['desc']))
    return text


def add_index(input_path, save_path, ner_tags):
    masked_data = utils.read_jsonl_file(input_path)
    claim_data = utils.read_jsonl_file(save_path)
    
    global mask_utils
    mask_utils = MaskDescEntity()
    
    with open(save_path, 'w') as save_file:
        claim_pt = 0
        masked_pt = 0
        
        # for idx, instance in enumerate(masked_data):
        while True:
            claim = claim_data[claim_pt]
            if masked_pt==len(masked_data):
                text = create_masked_data(claim, ner_tags)
                update_json(save_file, claim, text)
                claim_pt+=1
            else:
                masked = masked_data[masked_pt]
                if (masked['pers']==claim['pers']) and (masked['idx']==claim['article_num']):
                    text = masked['masked_text']
                    text = mask_desc(text, load_articles.remove_tokenized_spaces(claim['desc']))
                    update_json(save_file, claim, text)

                    claim_pt+=1
                    masked_pt+=1

                elif masked['idx']>claim['article_num']:
                    text = create_masked_data(claim, ner_tags)
                    claim_pt+=1
                    if not text:
                        continue
                    update_json(save_file, claim, text)

                else:
                    masked_pt+=1
                
            if claim_pt==len(claim_data):
                print("End of claim data. Masked:", masked_pt, len(masked_data))
                break
        

def main(args, save_dir):
    
    save_path = f'{save_dir}/{args.save_name}_{args.source}_{args.split}.jsonl'
    
    data = read_data(args.source, args.split)
    ner_tags = ner_utils.load_ner_tags(args.source, args.split)
    attr_frequency = get_attr_frequency(data)
        
    with open(save_path, 'w') as save_file:
        for idx, instance in tqdm(enumerate(data), total=len(data)):
            out_data = create_example(idx, instance, attr_frequency, ner_tags[idx])
            if not out_data:
                continue
            json.dump(out_data, save_file)
            save_file.write('\n')
    
    input_path = f'{save_dir}/{args.input_name}_{args.source}_{args.split}.jsonl'
    add_index(input_path, save_path, ner_tags)


if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='cnn', 
                        help='cnn/dm')
    parser.add_argument('--split', type=str, default='valid', 
                        help='train/test/valid split')
    parser.add_argument('--input_name', type=str, required=True, 
                        help='directory path and file name for loading masked data')
    parser.add_argument('--save_name', type=str, required=True, 
                        help='directory path and file name for saving claim MCQA data')
    
    
    save_dir = '../e2e_saved'
    args = parser.parse_args()
    
    global wiki_data
    global wiki_name_to_id
    global all_properties
    global data_path
    
    data_path = f'../Data/{args.source}_stories_tokenized/'
    
    wiki_data = get_wiki_data()
    wiki_name_to_id = get_wiki_aliases(wiki_data)
    all_properties = [prop['property'] for idx, entry in wiki_data.items() for prop in entry['properties']]
    
    main(args, save_dir)
    
    
    