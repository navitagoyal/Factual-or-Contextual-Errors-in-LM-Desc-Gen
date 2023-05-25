import spacy
from spacy import displacy

##### https://towardsdatascience.com/natural-language-processing-dependency-parsing-cf094bbbe3f7

class Parser:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        
        
    def parse_sub_node(self, node, flag=None):
        ### For the named entity node, skip the conjuctions, but include conjuctions of descriptions
        if flag=='root' and str(node.dep_)=='conj':
            return None, None

        ### Retain the "root" tag for the entity node across commas and conjunctions
        sub_flag = None    
        if flag=='root' and str(node.dep_) in ['punct', 'cc']:
            sub_flag = 'root'

        desc, name = self.parse_description(node, sub_flag)
        if flag=='root' and str(node.ent_type_) in ['PERSON']: # Including ORG as PERSON and ORG is often misidentified
            name += f'{str(node.text)} '

        return desc, name


    def parse_node(self, node, curr_desc, curr_name, flag=None):
        child_desc, child_name = self.parse_sub_node(node, flag)
        if child_desc:
            curr_desc += child_desc
        if child_name:
            curr_name += child_name

        return curr_desc, curr_name


    def parse_description(self, node, flag=None):
        desc = ""
        name = ""

        for c in node.lefts:
            desc, name = self.parse_node(c, desc, name, flag)

        desc += f'{str(node.text)} '

        for c in node.rights:
            desc, _ = self.parse_node(c, desc, name, flag)

        return desc, name


    def dep_parsing(self, text):
        # nlp function returns an object with individual token information, 
        # linguistic features and relationships
        return self.nlp(text)
    
    
    def visualize(self, doc):
        print ("{:<15} | {:<8} | {:<8} | {:<15} | {:<20}".format('Token','Relation', 'POS', 'Head', 'Children'))
        print ("-" * 70)
        for token in doc:
            # Print the token, dependency nature, head and all dependents of the token
            print ("{:<15} | {:<8} | {:<8} | {:<15} | {:<20}"
                 .format(str(token.text), str(token.dep_), str(token.ent_type_), str(token.head.text), str([child for child in token.children])))


        # Use displayCy to visualize the dependency 
        displacy.render(doc, style='dep', jupyter=True, options={'distance': 120})