import string
import regex as re
import load_articles

discard_list = ['a', 'an', 'the', 
                'your', 'his', 'her'
                'himself', 'herself', 'themselves', 
                'mr.', 'miss', 'ms.', 'mrs.']



def remove_tokenized_spaces(text):
    text = load_articles.remove_tokenized_spaces(text)
    text = text.replace(" - ", "-")
    text = text.replace(' .', '.')
    return text


def clean_desc(desc):
    desc = desc.split(" ")
    desc = [i for i in desc if i!=""]
    desc = " ".join(desc)
    desc = desc.strip()
    if desc.endswith(" and"):
        desc = desc[:-len(" and")]
    if desc.endswith(" and."):
        desc = desc[:-len(" and.")]
    desc = desc.rstrip(", ")

    desc = remove_tokenized_spaces(desc)

    return desc



def clean_extracted_desc(text):
    ### clean description after removing person name
    text = text.strip()
    text = '' if all(j in string.punctuation for j in text) else text
    text = text.strip(',')
    text = re.sub(r"^\'s", r'', text)
    text = remove_tokenized_spaces(text)

    return text.strip()


def extract_desc(desc, name):   
    if name not in desc:
        # print(f"\x1b[31m\"Name {name} not found in {desc}\"\x1b[0m")
        return None

    desc_wo_name = desc.replace(name, '')
    desc_wo_name = clean_extracted_desc(desc_wo_name)

    if desc_wo_name=='':
        return None

    if desc_wo_name.lower() in discard_list:
        return None

    return desc_wo_name


