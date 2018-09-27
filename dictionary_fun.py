import pandas
import nltk
#nltk.download()
#Run nltk.download() if you encounter nltk issues.

def map_bodies(data, body_col):
    """
    data: DataFrame
    body_col : string
    
    data represents a DataFrame containing Body IDs and actual text bodies.
    
    Ex:    Body ID                                        articleBody
    0           0  A small meteorite crashed into a wooded area i...
    1           4  Last week we hinted at what was to come as Ebo...
    2           5  (NEWSER) – Wonder how long a Quarter Pounder w...
    3           6  Posting photos of a gun-toting child online, I...
    4           7  At least 25 suspected Boko Haram insurgents we...
    5           8  There is so much fake stuff on the Internet in...
    6           9  (CNN) -- A meteorite crashed down in Managua, ...



    body_col is the name of the column containing article text bodies
    
    Returns: dictionary such that {Body ID : Body Text}
    """
    dictionary = dict()
    
    for x in range(len(data[body_col])):
        dictionary.update({data.iloc[x,0] : data.iloc[x,1]})
    
    return dictionary

def tokenize_dict(dictionary):
    """
    dictionary : dictionary
    
    Takes in a dictionary containing mappings from Body ID to Body.
    Returns a dictionary containing mappings from Body ID to Tokenized Bodies.
    """
    new_dict = dict()
    for x in dictionary:
        tokens = nltk.word_tokenize(dictionary.get(x))
        new_dict.update({x:tokens})
    return new_dict

def tag_tokens(dictionary):
    """
    Takes in a dictionary containing mappings from Body ID to tokenized bodies.
    Returns a dictionary containing mappings from Body ID to tagged tokenized bodies.
    """
    new_dict = dict()
    for x in dictionary:
        tagged = nltk.nltk.post_tag(dictionary.get(x))
        new_dict.update({x:tagged})
    return new_dict