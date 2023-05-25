import json

def remove_duplicates(articles, ner_tags):
    ### Remove duplicate entries

    nondup_idx = []
    for idx, a in enumerate(articles):
        if a not in articles[:idx]:
            nondup_idx.append(idx)

            
    articles = [a for idx, a in enumerate(articles) if idx in nondup_idx]
    ner_tags = [a for idx, a in enumerate(ner_tags) if idx in nondup_idx]

    
    return articles, ner_tags


def read_jsonl_file(filename):
    json_data = []

    with open(filename, 'r') as json_file:
        json_list = list(json_file)

    for json_str in json_list:
        json_data.append(json.loads(json_str))
        
    return json_data