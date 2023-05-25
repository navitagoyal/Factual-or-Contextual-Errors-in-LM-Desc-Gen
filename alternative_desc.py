import json
import random
import argparse
import pandas as pd
from tqdm import tqdm
from nltk import word_tokenize

import utils


def overlap(desc1, desc2):
    common_tokens = list(set(word_tokenize(desc1)) & set(word_tokenize(desc2)))
    # print(len(common_tokens))
    common_tokens = [token for token in common_tokens if token not in ['a', 'an', 'the']]
    return len(common_tokens)!=0


def overlap_multiple(desc1, desc):
    for desc2 in desc:
        if len(list(set(desc1) & set(desc2)))!=0:
            return True
    return False
    # return len(list(set(nltk.word_tokenize(desc1)) & set(nltk.word_tokenize(desc2))))!=0


def sample_same_person(D_df):
    for index, row in tqdm(D_df.iterrows(), total=len(D_df)):
        ## Choose a smaller random sample to begin just to speed up computation
        # sample_space = D_df.loc[random.sample(list(D_df.index), 500)]

        curr_article, curr_person = row['article_num'], row['person']

        sample_space = D_df[D_df.apply(lambda x: x['person']==curr_person, axis=1)]
        sample_space = sample_space[sample_space.apply(lambda x: x['article_num']!=curr_article, axis=1)]
        sample_space = sample_space.reset_index()
        
        ## Consider a smaller subset of sample to find non-overlapping desc
        sample = random.sample(list(sample_space.index),
                                   min(len(sample_space), 100))
        sample_space = sample_space.iloc[sample]

        ### Non-overlapping description
        sample_space = sample_space[sample_space.apply(lambda x: overlap(x['description'], row['description'])==False, axis=1)]

        if len(sample_space)>0:
            sample = random.sample(list(sample_space.index), 1)
            D_df.loc[index, 'same_person'] = sample_space.loc[sample[0], 'description']
            
    D_df = D_df[D_df['same_person']==D_df['same_person']]
    return D_df


def sample_same_article(D_df):
    for index, row in D_df.iterrows():
        curr_article, curr_person = row['article_num'], row['person']
        sample_space = D_df[D_df.apply(lambda x: (x['article_num']==curr_article) & (x['person']!=curr_person), axis=1)]
        if len(sample_space)>0:
            sample_space = sample_space[sample_space.apply(lambda x: overlap(x['description'], row['description'])==False, axis=1)]
            sample_space = sample_space[sample_space.apply(lambda x: overlap(x['description'], row['same_person'])==False, axis=1)]

        if len(sample_space)>0:
            sample = random.sample(list(sample_space.index), 1)
            D_df.loc[index, 'same_article'] = sample_space.loc[sample[0], 'description']
            
    
    D_df = D_df[D_df['same_article']==D_df['same_article']]
    return D_df


def sample_random(D_df):
    D_df = D_df[D_df['description']==D_df['description']]
    D_df = D_df[D_df['same_person']==D_df['same_person']]
    D_df = D_df[D_df['same_article']==D_df['same_article']]

    D_df['description_prime'] = [[token for token in word_tokenize(x) if token not in ['a', 'an', 'the']] for x in D_df['description']]
    D_df['same_person_prime'] = [[token for token in word_tokenize(x) if token not in ['a', 'an', 'the']] for x in D_df['same_person']]
    D_df['same_article_prime'] = [[token for token in word_tokenize(x) if token not in ['a', 'an', 'the']] for x in D_df['same_article']]
    
    for index, row in tqdm(D_df.iterrows(), total=len(D_df)):
        curr_article, curr_person = row['article_num'], row['person']
        sample_space = D_df[D_df.apply(lambda x: (x['article_num']!=curr_article) & (x['person']!=curr_person), axis=1)]

        if len(sample_space)>0:
            sample_space = sample_space[sample_space.apply\
                                        (lambda x: overlap_multiple(x['description_prime'], \
                                                           [row['description_prime'], row['same_person_prime'], \
                                                            row['same_article_prime']])==False, axis=1)]
        if len(sample_space)>0:
            sample = random.sample(list(sample_space.index), 1)
            D_df.loc[index, 'random'] = sample_space.loc[sample[0], 'description']
            
            
    D_df = D_df.drop(columns=['description_prime', 'same_person_prime', 'same_article_prime'])
    D_df = D_df[D_df['random']==D_df['random']]

    return D_df


def csv_to_jsonl(D_df, save_name):
    with open(save_name, 'w') as save_file:

        for idx, row in D_df.iterrows():
            out_data = {'idx': idx, #row['idx'] if index is dumped while saving dataframe
                    'article_num': row['article_num'], 
                    'pers': row['person'],
                    'accurate': row['description'], 
                    'nonfactual': row['same_article'], 
                    'incongruous': row['same_person'], 
                    'both': row['random']
                   }

            json.dump(out_data, save_file)
            save_file.write('\n')


def main(args, save_dir):
    
    input_name = f'{save_dir}/{args.input_name}_{args.source}_{args.split}.jsonl'
    save_name = f'{save_dir}/{args.save_name}_{args.source}_{args.split}.jsonl'
    
    masked_data = utils.read_jsonl_file(input_name)
    
    D_df = pd.DataFrame(columns=['article_num', 'person', 'description'])
    for instance in masked_data:
        D_df.loc[len(D_df.index)] = [instance['idx'], instance['pers'], instance['label']]
        
    D_df = sample_same_person(D_df)
    D_df = sample_same_article(D_df)
    D_df = sample_random(D_df)
    
    # D_df.to_csv(f'../saved/desc_controlled/alternate_labels_controlled_articles_cnn_{split}.csv', index_label = 'idx')
    
    csv_to_jsonl(D_df, save_name)
        
    


if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='cnn', 
                        help='cnn/dm')
    parser.add_argument('--split', type=str, default='valid', 
                        help='train/test/valid split')
    parser.add_argument('--input_name', type=str, required=True, 
                        help='directory path and file name for loading masked data')
    parser.add_argument('--save_name', type=str, required=True, 
                        help='directory path and file name for saving description MCQA data')
    
    save_dir = '../e2e_saved'
    args = parser.parse_args()
    
    main(args, save_dir)