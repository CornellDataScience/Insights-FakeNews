import pickle
import bcolz
import queue
import spacy
from scipy import spatial
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn import feature_extraction
import numpy as np
from nltk import tokenize
import nltk
import re
"""
FUNCTIONS FOR PREPROCESSING AND FEATURE ENGINEERING - pending reorganization
"""


"""
Helpers borrowed from baseline implementation in feature_engineering.py
"""


"setup after spacy is installed: python -m spacy download en"
nlp = spacy.load("en")

_wnl = nltk.WordNetLemmatizer()
analyzer = SentimentIntensityAnalyzer()
glove_path = "word_embeddings"

"""
stolen from SO: 
stackoverflow.com/questions/25534214

in: string representing POS tag
out: bool
"""


def is_noun(tag):
    return tag in ['NN', 'NNS', 'NNP', 'NNPS']


def is_verb(tag):
    return tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']


def is_adverb(tag):
    return tag in ['RB', 'RBR', 'RBS']


def is_adjective(tag):
    return tag in ['JJ', 'JJR', 'JJS']


"""
takes word and lemmatizes/makes lowercase
in: string
out: string
"""


def normalize_word(w):
    return _wnl.lemmatize(w).lower()


"""
lemmatized, lowercased tokens
in: string
out: string list
"""


def get_tokenized_lemmas(s):
    lemmas = [normalize_word(t) for t in nltk.word_tokenize(s)]
    return [i for i in lemmas if len(i) > 1]


"""
lowercased tokens - for use w/ word embeddings
in: string
out: string list
"""


def get_tokens(s):
    return [t.lower() for t in nltk.word_tokenize(s)]


"""
Cleans a string: trimming, removing non-alphanumeric
in: string
out: string
"""


def clean(s, lower=True):
    if lower:
        return re.sub('[\W_]+', ' ', s, flags=re.UNICODE).lower()
    else:
        return re.sub('[\W_]+', ' ', s, flags=re.UNICODE)


"""
Removes stopwords from a list of tokens
in: string list
out: string list
"""


def remove_stopwords(l):
    return [w for w in l if w not in feature_extraction.text.ENGLISH_STOP_WORDS]


"""
Function for getting lemmatized tokens in one step
in: string
out: string list
"""


def get_clean_tokens(body):
    clean_body = clean(body)
    tokens = get_tokenized_lemmas(clean_body)
    clean_tokens = remove_stopwords(tokens)
    return clean_tokens


"""
in: row number, dataframe (train_stances)
out: string of headline
"""


def get_headline(n, df):
    return (df.iloc[n])["Headline"]


"""
in: body ID, dataframe (train_bodies)
out: string of body text
example: get_body(6, train_bodies)
"""


def get_body(n, df):
    return df.loc[lambda x: x["Body ID"] == n, "articleBody"].item()


"""
sentiment for single sentence
in: string
out: sentiment dict - SEE BELOW
"""


def get_sentiment(sentence):
    return analyzer.polarity_scores(sentence)


"""
average of sentiment scores for each sentence
{"pos", "neg", "neu", "compound"}

in: string w/ multiple sentences
out: dictionary object with sentiment scores for each category
"""


def sentiment_multi(paragraph):
    categories = ["pos", "neg", "neu", "compound"]
    sentence_list = tokenize.sent_tokenize(paragraph)
    paragraphSentiments = {"pos": 0, "neg": 0, "neu": 0, "compound": 0}
    for sentence in sentence_list:
        vs = get_sentiment(sentence)
        for cat in categories:
            paragraphSentiments[cat] += vs[cat]
    for cat in categories:
        paragraphSentiments[cat] /= len(sentence_list)
    return paragraphSentiments


"""
union of strings in A and B
in: 2 string lists
out: string list
"""


def shared_vocab(a, b):
    return list(set(a).union(set(b)))


"""
cosine similarity of two bags of words

in: 2 string lists - cannot be empty
out: float
"""


def bow_cos_similarity(a, b):
    vocab = shared_vocab(a, b)
    a_bow, b_bow = set(a), set(b)
    if len(a) == 0 or len(b) == 0:
        return -1
    a_vec = [(1 if i in a_bow else 0) for i in vocab]
    b_vec = [(1 if i in b_bow else 0) for i in vocab]
    return 1 - spatial.distance.cosine(a_vec, b_vec)


"""
just a cosine similarity wrapper
pls no zero vectors
"""


def cosine_similarity(a, b):
    return 1 - spatial.distance.cosine(a, b)


'''
extracts the subject, verb, and object of a sentence, 
including news headlines and body text
in: string (sentence)
out: triple of (subj, verb, obj)
'''


def extract_SVO(sent):
    subj = ""
    verb = ""
    obj = ""
    subjFound = False
    verbFound = False
    objFound = False

    tokenized_sent = nlp(sent)

    for token in tokenized_sent:
        if (token.dep_ == "nsubj" and subjFound == False):
            subj = token.text
            subjFound = True

        if (token.pos_ == "VERB" and verbFound == False):
            verb = token.text
            verbFound = True
        elif (token.head.pos_ == "VERB" and verbFound == False):
            verb = token.head.text
            verbFound = True

        if (token.dep_ == "dobj" or token.dep_ == "pobj" and objFound == False):
            obj = token.text
            objFound = True

    return (subj, verb, obj)


"""
helper function for IDF's
in: 2d list of strings
out: dict of string -> idf score (float)
"""

'''root distance feature link to paper:
http://aclweb.org/anthology/N/N16/N16-1138.pdf

returns the average root distance among all three tokens in a trigram.
If the trigram had a negative score of 0, simply return 1.0
'''


def find_avg_root_dist(sent):
    tokenized_sent = nlp(sent)
    num_toks = len(tokenized_sent)
    root = find_root(tokenized_sent)

    trigram_tok_lst = list(
        zip(tokenized_sent, tokenized_sent[1:], tokenized_sent[2:]))

    max_neg, trigram = find_most_neg_trigram(trigram_tok_lst)

    if (max_neg == 0):
        return (1.0, trigram)

    dist = 0.0

    for token in trigram:
        dist_to_tok = calc_root_dist(root, token, num_toks)
        dist = dist + dist_to_tok

    avg_dist = dist / len(trigram)

    return (avg_dist, trigram)


'''
calculates the root distance. In other words,
this method performs a BFS from the root to the
token node and returns the distance divided by the number 
of tokens in order to keep all the values between 0 and 1
'''


def calc_root_dist(root, token, num_toks):
    if (root == None):
        return 1.0

    dist = 0.0
    q = queue.Queue(maxsize=0)
    visited = set()

    q.put(root)
    visited.add(root)

    while (not(q.empty())):
        curr_tok = q.get()

        dist = dist + 1.0

        for tok in curr_tok.children:
            if (not(tok in visited)):

                if (tok == token):
                    return dist / num_toks

                q.put(tok)
                visited.add(tok)

    return dist / num_toks


'''
retrieves the root of a tokenized sentence
'''


def find_root(tokenized_sent):
    for tok in tokenized_sent:
        if (tok.dep_ == "ROOT"):
            return tok
    return None


''' 
given a trigram list, 
find the trigram that results in the most negativity
returns the max_negative value, along with the trigram itself
'''


def find_most_neg_trigram(trigram_lst):
    max_neg = 0.0
    most_neg_trigram = trigram_lst[0]

    for trigram in trigram_lst:
        phrase = ""

        for tok in trigram:
            phrase = phrase + " " + tok.text

        polarity_scores = analyzer.polarity_scores(phrase)
        neg_val = polarity_scores["neg"]

        if (neg_val > max_neg):
            max_neg = neg_val
            most_neg_trigram = trigram

    return max_neg, most_neg_trigram


def build_idf_tokens(corpus, pos_tags=None):
    num_docs = len(corpus)
    vocabulary = Counter()
    for wordlist in list(corpus):
        if pos_tags is None:
            filtered_tokens = wordlist
        else:
            pos = nltk.pos_tag(wordlist)
            filtered_tokens = [x[0] for x in pos if (x[1] in pos_tags)]
        for i in set(filtered_tokens):
            vocabulary[i] += 1
    idf = {}
    for i in vocabulary:
        idf[i] = np.log(num_docs/(vocabulary[i]))
    return idf


"""
builds IDF from the headlines and bodies included in a stances df
NOTE: THIS IS VERY SLOW
example: idf = preprocessing.build_idf(stances_tr, ["NN","NNP", "NNPS", "NNS"])

in: body df, stances df (containing list of bodies to extract)
out: dict of string -> idf score (float) with one special entry (_avg -> average idf score)
"""


def build_idf(bodies, stances, pos_tags=None):
    corpus = [get_body(x, bodies) for x in set(
        stances['Body ID'])] + list(stances['Headline'])
    tokenized_corpus = [get_clean_tokens(x) for x in corpus]
    idf = build_idf_tokens(tokenized_corpus, pos_tags)
    idf["_avg"] = float(sum(idf.values())) / len(idf)
    return idf


"""
get average tf-idf score of tokens in sentence
if no idf, then use tf score
must provide avg_idf if using idf!

in: list of tokens, tf dictionary, idf dictionary, average of idf
out: float
"""


def score_sentence(sentence, tf, idf=None):
    acc = 0
    if idf == None:
        for token in sentence:
            acc += tf[token]
    else:
        avg_idf = idf["_avg"]
        for token in sentence:
            if token in idf:
                acc += tf[token]*idf[token]
            else:
                acc += tf[token]*avg_idf
    return acc/len(sentence)


"""
extract metadata from sentence/body of text
in: string
out: dict
"""


def process_sentence(s):
    clean_s = clean(s)
    tokens = get_tokenized_lemmas(clean_s)
    clean_tokens = remove_stopwords(tokens)
    svo = extract_SVO(" ".join(tokens))

    bigram = list(nltk.bigrams(clean_tokens))
    bigram_str = [x[0]+' '+x[1] for x in bigram]

    pos = nltk.pos_tag(clean_tokens)

    # list of words that belong to that part of speech
    nouns = [x[0] for x in pos if is_noun(x[1])]
    verbs = [x[0] for x in pos if is_verb(x[1])]
    adjectives = [x[0] for x in pos if is_adjective(x[1])]
    adverbs = [x[0] for x in pos if is_adverb(x[1])]

    vader_sentiment = get_sentiment(s)

    return {
        "tokens": clean_tokens,
        "bigrams": bigram_str,
        "nouns": nouns,
        "verbs": verbs,
        "adjectives": adjectives,
        "adverbs": adverbs,
        "sentiment": vader_sentiment,
        "svo": list(svo)
    }


"""
extracts metadata from article body - READ COMMENTS FOR MORE INFO!
in: string, idf dictionary (string->float) [optional]
out: mixed dictionary

example usage:
process_body(get_body(6, train_bodies))
"""


def process_body(body, idf=None):
    sentences = list(nltk.tokenize.sent_tokenize(body))
    # sentences are tokenized
    clean_sentences = [get_clean_tokens(s) for s in sentences]
    clean_sentences = [s for s in clean_sentences if len(s) > 3]
    clean_tokens = [
        token for sentence in clean_sentences for token in sentence]
    body_length = len(clean_tokens)

    # look at first sentence of article
    first_sentence = sentences[0]
    first_sentence_data = process_sentence(first_sentence)

    # extracting bigrams
    bigram = list(nltk.bigrams(clean_tokens))
    bigram_str = [x[0]+' '+x[1] for x in bigram]

    pos = nltk.pos_tag(clean_tokens)

    # count of each tag type (dict), counter for word (tf)
    tags_count = Counter([x[1] for x in pos])
    word_count = Counter([x[0] for x in pos])

    # list of words that belong to that part of speech
    nouns = [x[0] for x in pos if is_noun(x[1])]
    verbs = [x[0] for x in pos if is_verb(x[1])]
    adjectives = [x[0] for x in pos if is_adjective(x[1])]
    adverbs = [x[0] for x in pos if is_adverb(x[1])]

    doc_len = len(clean_tokens)

    n_counter = Counter(nouns)
    v_counter = Counter(verbs)
    b_counter = Counter(bigram)
    token_counter = Counter(clean_tokens)

    # common words are highest scoring IDF (or TF if IDF not available)
    # significant sentence - sentence with highest average token IDF score
    # if no IDF use TF (which is not that good)
    if idf == None:
        common_nouns = [x[0] for x in n_counter.most_common(5)]
        common_verbs = [x[0] for x in v_counter.most_common(5)]
        common_tokens = [x[0] for x in token_counter.most_common(5)]
        # this is really shitty
        sentence_importance = [(s, score_sentence(s, word_count))
                               for s in clean_sentences]
        most_significant_sentence, sentence_score = list(
            sorted(sentence_importance, key=lambda x: x[1]))[-1]
        most_significant_sentence_data = process_sentence(
            ' '.join(most_significant_sentence))

    else:
        avg_idf = idf["_avg"]
        n_tfidf, v_tfidf, t_tfidf = {}, {}, {}
        for n in n_counter:
            n_tfidf[n] = (n_counter[n]/doc_len) * \
                (idf[n] if n in idf else avg_idf)
        for v in v_counter:
            v_tfidf[v] = (v_counter[v]/doc_len) * \
                (idf[v] if v in idf else avg_idf)
        for t in token_counter:
            t_tfidf[t] = (token_counter[t]/doc_len) * \
                (idf[t] if t in idf else avg_idf)
        common_nouns = sorted(n_tfidf, key=n_tfidf.get, reverse=True)[:5]
        common_verbs = sorted(v_tfidf, key=v_tfidf.get, reverse=True)[:5]
        common_tokens = sorted(t_tfidf, key=t_tfidf.get, reverse=True)[:5]

        sentence_importance = [
            (s, score_sentence(s, word_count, idf)) for s in clean_sentences]
        most_significant_sentence, sentence_score = list(
            sorted(sentence_importance, key=lambda x: x[1]))[-1]
        most_significant_sentence_data = process_sentence(
            ' '.join(most_significant_sentence))

    # no idf for bigrams increase "common" count to 10
    common_bigrams = [x[0] for x in b_counter.most_common(10)]

    n_adj = len(adjectives)
    n_adv = len(adverbs)
    # breakdown of porportion of regular/comparative/superlative adverbs as a tuple (in that order)
    if n_adj != 0:
        adj_types = (tags_count['JJ']/n_adj,
                     tags_count['JJR']/n_adj, tags_count['JJS']/n_adj)
    else:
        adj_types = (0, 0, 0)
    if n_adv != 0:
        adv_types = (tags_count['RB']/n_adv,
                     tags_count['RBR']/n_adv, tags_count['RBS']/n_adv)
    else:
        adv_types = (0, 0, 0)

    vader_sentiment = sentiment_multi(body)

    return {
        "tokens": clean_tokens,
        "bigrams": bigram_str,
        "nouns": nouns,
        "verbs": verbs,
        "adjectives": adjectives,
        "adverbs": adverbs,
        "sentiment": vader_sentiment,
        "first_sentence": first_sentence_data,
        "significant_sentence": most_significant_sentence_data,
        "adj_types": adj_types,
        "adv_types": adv_types,
        "vocabulary": set(clean_tokens),
        "common_tokens": common_tokens,
        "common_nouns": common_nouns,
        "common_verbs": common_verbs,
        "common_bigrams": common_bigrams,
    }


"""
in: df of bodies, idf dict (string->float) [optional]
out: dict with k=bodyid and v=dict of bodyinfo as per process_body

NOTE: THIS IS VERY SLOW
"""


def process_bodies(df, idf=None):
    body_info = {}
    ids = list(df["Body ID"])
    for i in range(len(ids)):
        if i % 100 == 0 and i != 0:
            print("processed "+str(i))
        body_info[ids[i]] = process_body(get_body(ids[i], df), idf)
    print("done! processed " + str(len(ids)))
    return body_info


"""
in: df of bodies, df of stances, fraction of bodies you want to be in training set
out: 2 df's of stances, first is training second is test; 2 lists of

this ensures no overlap of bodies between train and test sets, exactly like actual testing

example usage: stances_tr, stances_val = preprocessing.train_test_split(train_bodies, train_stances)
"""


def train_test_split(bodies, stances, split=0.8):
    idx = np.random.permutation(np.arange(len(bodies)))
    bodies = bodies.values[idx]
    train = int(len(bodies)*0.8)
    bodies_tr = set([i[0] for i in bodies[:train]])
    bodies_val = set([i[0] for i in bodies[train:]])
    stances_tr = stances.loc[stances["Body ID"].isin(bodies_tr), :]
    stances_val = stances.loc[stances["Body ID"].isin(bodies_val), :]
    return stances_tr, stances_val


"""
return a dictionary of features
see usage example:

data - [headline:str, body_id:str/int]
body_dict - dictionary of processed bodies
idf - dictionary of idf scores

NOTE: THIS IS VERY SLOW
"""


def get_feats(data, body_dict, idf=None):
    headline, body_id = data[0], int(data[1])
    headline_data = process_sentence(headline)

    shared_common_nouns = len(set(headline_data['nouns']).intersection(
        set(body_dict[body_id]['common_nouns'])))
    shared_common_verbs = len(set(headline_data['verbs']).intersection(
        set(body_dict[body_id]['common_verbs'])))
    shared_common_tokens = len(set(headline_data['tokens']).intersection(
        set(body_dict[body_id]['tokens'])))
    shared_bigrams = len(set(headline_data['bigrams']).intersection(
        set(body_dict[body_id]['bigrams'])))

    shared_nouns_first = len(set(headline_data['nouns']).intersection(
        set(body_dict[body_id]['first_sentence']['nouns'])))
    shared_verbs_first = len(set(headline_data['verbs']).intersection(
        set(body_dict[body_id]['first_sentence']['verbs'])))
    shared_bigrams_first = len(set(headline_data['bigrams']).intersection(
        set(body_dict[body_id]['first_sentence']['bigrams'])))
    shared_tokens_first = len(set(headline_data['tokens']).intersection(
        set(body_dict[body_id]['first_sentence']['tokens'])))

    shared_nouns_sig = len(set(headline_data['nouns']).intersection(
        set(body_dict[body_id]['significant_sentence']['nouns'])))
    shared_verbs_sig = len(set(headline_data['verbs']).intersection(
        set(body_dict[body_id]['significant_sentence']['verbs'])))
    shared_bigrams_sig = len(set(headline_data['bigrams']).intersection(
        set(body_dict[body_id]['significant_sentence']['bigrams'])))
    shared_tokens_sig = len(set(headline_data['tokens']).intersection(
        set(body_dict[body_id]['significant_sentence']['tokens'])))

    # #adv and adj for stance
    # shared_adjectives_sig = len(set(headline_data['adjectives']).intersection(
    #     set(body_dict[body_id]['significant_sentence']['adjectives'])))
    # shared_adverbs_sig = len(set(headline_data['adverbs']).intersection(
    #     set(body_dict[body_id]['significant_sentence']['adverbs'])))
    # shared_adjectives_fst = len(set(headline_data['adjectives']).intersection(
    #     set(body_dict[body_id]['first_sentence']['adjectives'])))
    # shared_adverbs_fst = len(set(headline_data['adverbs']).intersection(
    #     set(body_dict[body_id]['first_sentence']['adverbs'])))

    #difference in sentiment
    sentiment_diff = {
        "pos": headline_data['sentiment']['pos']-body_dict[body_id]['sentiment']['pos'],
        "neg": headline_data['sentiment']['neg']-body_dict[body_id]['sentiment']['neg'],
        "neu": headline_data['sentiment']['neu']-body_dict[body_id]['sentiment']['neu'],
        "compound": headline_data['sentiment']['compound']-body_dict[body_id]['sentiment']['compound']
    }
    sentiment_diff_first = {
        "pos": headline_data['sentiment']['pos']-body_dict[body_id]['first_sentence']['sentiment']['pos'],
        "neg": headline_data['sentiment']['neg']-body_dict[body_id]['first_sentence']['sentiment']['neg'],
        "neu": headline_data['sentiment']['neu']-body_dict[body_id]['first_sentence']['sentiment']['neu'],
        "compound": headline_data['sentiment']['compound']-body_dict[body_id]['first_sentence']['sentiment']['compound']
    }
    sentiment_diff_sig = {
        "pos": headline_data['sentiment']['pos']-body_dict[body_id]['significant_sentence']['sentiment']['pos'],
        "neg": headline_data['sentiment']['neg']-body_dict[body_id]['significant_sentence']['sentiment']['neg'],
        "neu": headline_data['sentiment']['neu']-body_dict[body_id]['significant_sentence']['sentiment']['neu'],
        "compound": headline_data['sentiment']['compound']-body_dict[body_id]['significant_sentence']['sentiment']['compound']
    }

    headline_svo = headline_data['svo']
    body_fst_svo = body_dict[body_id]['first_sentence']['svo']
    body_sig_svo = body_dict[body_id]['significant_sentence']['svo']

    # cosine similarity - no verbs because relatively few per sentence
    cos_nouns_first = bow_cos_similarity(
        headline_data['nouns'], body_dict[body_id]['first_sentence']['nouns'])
    cos_bigrams_first = bow_cos_similarity(
        headline_data['bigrams'], body_dict[body_id]['first_sentence']['bigrams'])
    cos_tokens_first = bow_cos_similarity(
        headline_data['tokens'], body_dict[body_id]['first_sentence']['tokens'])

    cos_nouns_sig = bow_cos_similarity(
        headline_data['nouns'], body_dict[body_id]['significant_sentence']['nouns'])
    cos_bigrams_sig = bow_cos_similarity(
        headline_data['bigrams'], body_dict[body_id]['significant_sentence']['bigrams'])
    cos_tokens_sig = bow_cos_similarity(
        headline_data['tokens'], body_dict[body_id]['significant_sentence']['tokens'])

    return {
        'shared_nouns': shared_common_nouns,
        'shared_verbs': shared_common_verbs,
        'shared_bigrams': shared_bigrams,
        'shared_tokens': shared_common_tokens,

        'shared_nouns_fst': shared_nouns_first,
        'shared_verbs_fst': shared_verbs_first,
        'shared_bigrams_fst': shared_bigrams_first,
        'shared_tokens_fst': shared_tokens_first,

        'shared_nouns_sig': shared_nouns_sig,
        'shared_verbs_sig': shared_verbs_sig,
        'shared_bigrams_sig': shared_bigrams_sig,
        'shared_tokens_sig': shared_tokens_sig,

        'cos_nouns_sig': cos_nouns_sig,
        'cos_bigrams_sig': cos_bigrams_sig,
        'cos_tokens_sig': cos_tokens_sig,

        'cos_nouns_fst': cos_nouns_first,
        'cos_bigrams_fst': cos_bigrams_first,
        'cos_tokens_fst': cos_tokens_first,

        'svo_s_fst': int(headline_svo[0] == body_fst_svo[0]),
        'svo_v_fst': int(headline_svo[1] == body_fst_svo[1]),
        'svo_o_fst': int(headline_svo[2] == body_fst_svo[2]),

        'svo_s_sig': int(headline_svo[0] == body_sig_svo[0]),
        'svo_v_sig': int(headline_svo[1] == body_sig_svo[1]),
        'svo_o_sig': int(headline_svo[2] == body_sig_svo[2]),

        'sentiment_pos': sentiment_diff['pos'],
        'sentiment_neg': sentiment_diff['neg'],
        'sentiment_neu': sentiment_diff['neu'],
        'sentiment_compound': sentiment_diff_first['compound'],

        'sentiment_pos_fst': sentiment_diff_first['pos'],
        'sentiment_neg_fst': sentiment_diff_first['neg'],
        'sentiment_neu_fst': sentiment_diff_first['neu'],
        'sentiment_compound_fst': sentiment_diff_first['compound'],

        'sentiment_pos_sig': sentiment_diff_sig['pos'],
        'sentiment_neg_sig': sentiment_diff_sig['neg'],
        'sentiment_neu_sig': sentiment_diff_sig['neu'],
        'sentiment_compound_sig': sentiment_diff_sig['compound']
    }


"""
modified from: https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76

input should just be the filename with no extensions or directory
dumps some compressed files into the disk for easier loading/lookup of word embeddings later

usage example: preprocessing.extract_word_embeddings("glove.6B.50d")
"""


def extract_word_embeddings(filename):
    words = []
    idx = 0
    word2idx = {}
    dims = None
    vectors = bcolz.carray(
        np.zeros(1), rootdir=f'{glove_path}/{filename}.dat', mode='w')

    with open(f'{glove_path}/{filename}.txt', 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            words.append(word)
            word2idx[word] = idx
            idx += 1
            vect = np.array(line[1:]).astype(np.float)
            if idx == 1:
                dims = len(vect)
            vectors.append(vect)
    vectors = bcolz.carray(vectors[1:].reshape(
        (idx, dims)), rootdir=f'{glove_path}/{filename}.dat', mode='w')
    vectors.flush()
    pickle.dump(words, open(f'{glove_path}/{filename}_words.pkl', 'wb'))
    pickle.dump(word2idx, open(f'{glove_path}/{filename}_idx.pkl', 'wb'))


"""
taken from: https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76

in: filename of preprocessed glove embeddings (ex: glove.6B.50d), must run extract_word_embeddings first
out: dict that u can use to look up vectors for words

loading example:
glove = preprocessing.get_glove_dict("glove.6B.50d")

lookup example:
glove["the"]
"""


def get_glove_dict(name):
    vectors = bcolz.open(f'{glove_path}/{name}.dat')[:]
    words = pickle.load(open(f'{glove_path}/{name}_words.pkl', 'rb'))
    word2idx = pickle.load(open(f'{glove_path}/{name}_idx.pkl', 'rb'))

    glove = {w: vectors[word2idx[w]] for w in words}
    return glove
