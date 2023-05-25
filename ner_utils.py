from nltk.tokenize import word_tokenize
import pickle as pkl
import collections


class NerTagger:
    def __init__(self):
        from nltk.tag import StanfordNERTagger
        
        self.ner = StanfordNERTagger(
            '../tools/stanford-ner-2020-11-17/classifiers/english.all.3class.distsim.crf.ser.gz',
            '../tools/stanford-ner-2020-11-17/stanford-ner.jar',
            encoding='utf-8')
        
    def tag(self, sentence):
        return self.ner.tag(word_tokenize(sentence))
    
    def tag_all(self, source, split):
        ## Run NER tag on source, split articles and save
        pass
        
        
def load_ner_tags(source, split):
    data_path = '../saved/ner_tags'
    filename = f'ner_tags_articles_{source}_{split}'
    ner_tags = pkl.load(open(f'{data_path}/{filename}.pkl', 'rb'))
    
    return ner_tags


def get_continuous_chunks(tagged_sent):
    continuous_chunk = []
    current_chunk = []
    prev_tag = "O"

    for token, tag in tagged_sent:
        if tag != "O":
            if tag == prev_tag:
                current_chunk.append((token, tag))
            else:
                prev_tag = tag
                if current_chunk:
                    continuous_chunk.append(current_chunk)
                current_chunk = [(token, tag)]
            
        else:
            prev_tag = "O"
            if current_chunk: # if the current chunk is not empty
                continuous_chunk.append(current_chunk)
                current_chunk = []
                
                
    # Flush the final current_chunk into the continuous_chunk, if any.
    if current_chunk:
        continuous_chunk.append(current_chunk)
    return continuous_chunk


def extract_person_ner_tags(ner_tags):
    chunked_tags = get_continuous_chunks(ner_tags)
    pers_tags = [i for i in chunked_tags if i[0][1]=='PERSON']
    pers_tags = [[token[0] for token in tag] for tag in pers_tags]
    return pers_tags


discard_entities = ['U. S.', 'U. N.']


def get_alias(pers_entities):
    
    pers_entities = [" ".join(tag) for tag in pers_entities]
    pers_entities = list(set(pers_entities))
    
    pers_entities = [i for i in pers_entities if i not in discard_entities]
    # Sorted in decreasing order so full names are added first
    pers_entities = sorted(pers_entities, key=len, reverse=True)  
    
    pers_entities_full  = []
    pers_entities_key = {}
    
    for name in pers_entities:
        flag = 0 # person's name is not included in full_name
        chosen_full_name = None
        for full_name in pers_entities_full:
            if name in full_name.split(" "):
                chosen_full_name = full_name
                flag += 1
                
        ### If more than one person has the same subname (/lastname) then do not construct an alias
        if flag == 1:
            pers_entities_key[name] = chosen_full_name
            
        ### If the name is not part of the fullname yet, add it
        if flag ==  0:
            pers_entities_full.append(name)
            
    for name in pers_entities_full:
        pers_entities_key[name] = name
    
    return pers_entities_full, pers_entities_key


def get_reverse_alias(pers_entities_key):
    pers_entities_reverse = collections.defaultdict(list)
    for alias, fullname in pers_entities_key.items():
        pers_entities_reverse[fullname].append(alias)
        
    return pers_entities_reverse