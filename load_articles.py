import pickle as pkl
import regex as re

pattern = re.compile("(?<=/).*(?=:)")

def get_filenames(source=None, split=''):
    if split == '':
        raise Exception("Specify a data split")
        
    pomo_data = '../Data/PoMo/dataset/'+split+'.pm'
    
    ### Use source as 'cnn' or 'dm'. Leave None to combine from both
    
    with open(pomo_data, 'r') as f:
        data = f.readlines()
        
    data = [i.split('\t') for i in data]
    
    if source:
        data_f = list(filter(lambda x: x[-1].startswith(source), data))
    else: 
        ## Filter data from CNN and Daily Mail
        data_f = list(filter(lambda x: x[-1].startswith('cnn') or x[-1].startswith('dm'), data))
        
    # filenames = [pattern.search(f[-1]).group(0) for f in data_f]
    filenames = [get_name_from_id(f[-1]) for f in data_f]
    
    return filenames


def get_name_from_id(file_id):
    filename = pattern.search(file_id).group(0)
    return filename


def get_article(filepath):
    text = open(filepath, 'r').read()
    text = text.split("@highlight")[0]
    return text


def clean(text):
    ### Remove the location and source from begining of the article
    if "-RRB-" in text[:50]:
        text = "".join(text.split('-RRB-')[1:])
        ## Remove any leading --
        # text = re.sub(r'^?\s--?\s', '', text)
        text = text.lstrip().lstrip('--').strip()
    
    text = text.replace("-LRB- ", "")
    text = text.replace("-RRB- ", "")
    
    text = text.replace("-LSB- ", "")
    text = text.replace("-RSB- ", "")
    
    ### Concatenate multi-line article
    text = " ".join(text.split("\n\n"))
    
    return text.strip()


def remove_tokenized_spaces(text):
    ### Remove consecutive white spaces
    text = re.sub(r'\s\s+', ' ', text)
    
    ### Remove spaces before punctuation to aid referring expression extraction
    text = re.sub(r'\s([?.!,"](?:\s|$))', r'\1', text)
    ### Remove space before 's; may not be necessary
    text = re.sub(r"\b\s+'\b", r"'", text)
    
    ### Remove space before opening (`) and closing (') quotes
    text = re.sub(r"(\`)\s", r"\1", text)
    text = re.sub(r"\s(\')", r"\1", text)
    
    ### Remove space before and after slash ('/')
    text = re.sub(r"[/]\s", "/", text)
    text = re.sub(r"\s[/]", "/", text)
    
    ### Remove space before puntuation (again!)
    text = re.sub(r'\s([.;])', r'\1', text)
    
    return text


def main(source, split):
    # global data_path 
    data_path = f'../Data/{source}_stories_tokenized/'
    filenames = get_filenames(source, split)
    
    articles = [get_article(f'{data_path}{filename}') for filename in filenames]            
    articles = [clean(article) for article in articles]
    articles_wo_spaces = [remove_tokenized_spaces(article) for article in articles]
    
    return articles_wo_spaces
    