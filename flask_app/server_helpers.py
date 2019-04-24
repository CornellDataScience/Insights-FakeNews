import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import numpy as np
import pandas as pd
import spacy
import networkx as nx
from scipy.spatial import distance
import matplotlib.pyplot as plt
import nltk
from collections import Counter, defaultdict
import itertools
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sentic import SenticPhrase
import en_coref_md
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

nlp = spacy.load('en_core_web_sm')
coref = en_coref_md.load()

negating_words = set([
    "n't", "not", "no", 
    "never", "nobody", "non", "nope"])
doubting_words = set([
    'fake','fraud', 'hoax', 
    'false', 'deny', 'denies', 
    'despite', 'doubt', 
    'bogus', 'debunk', 'prank', 
    'retract', 'scam', "withdrawn",
    "misinformation"])
hedging_words = set([
    'allege', 'allegedly','apparently',
    'appear','claim','could',
    'evidently','largely','likely',
    'mainly','may', 'maybe', 'might',
    'mostly','perhaps','presumably',
    'probably','purport', 'purportedly',
    'reported', 'reportedly',
    'rumor', 'rumour', 'rumored', 'rumoured',
    'says','seem','somewhat',
    'unconfirmed'])
sus_words = doubting_words.union(hedging_words)

vader = SentimentIntensityAnalyzer()

def get_sentiment(sentence):
    sent =  vader.polarity_scores(sentence)
    return [sent["pos"],sent["neg"],sent["neu"],sent["compound"]]

def get_avg_sentiment(lst):
    sents = np.array([get_sentiment(s) for s in lst])
    return list(np.mean(sents, axis = 0))

def get_diff_sentiment(a,b):
    return list(np.array(a) - np.array(b))

def preprocess(text):
    text = text.replace("' ",' ')
    text = text.replace("'\n",'\n')
    text = text.replace(" '",' ')
    text = text.replace('"',' ')
    text = text.replace('“',' ')
    text = text.replace('”', ' ')
    text = text.replace(":", ". ")
    text = text.replace(";", ". ")
    text = text.replace("...", " ")
    return text

def make_graph(token):
    valid_children = [c for c in list(token.lefts)+list(token.rights) if c.dep_ != "SPACE"]
    return {
        "name": token.lemma_.lower() + str(token.i),
        "token": token.lemma_.lower(),
        "pos": token.pos_,
        "dep": token.dep_,
        "idx": token.i,
        "children": [make_graph(c) for c in valid_children]
    }

def get_display_graph(headline, body_sents):
    headline_root = [t for t in headline if t.dep_== "ROOT"][0]
    body_roots = [[t for t in sent if t.dep_== "ROOT"][0] for sent in body_sents]
    headline_graph = make_graph(headline_root)
    body_graphs = [make_graph(r) for r in body_roots]
    return {"headline":headline_graph, "body":body_graphs}


def cosine_similarity(x,y):
    if all([a == 0 for a in x]) or all([a == 0 for a in y]):
        return 0
    return 1 - np.nan_to_num(distance.cosine(x,y))

def get_topics(doc):
    """
    get topics of a sentence
    input: spacy doc
    output: dictionary with nouns as the key, and the set of noun chunks that contain the noun as the value
    special entry _vocab has the set of all tokens in the dict
    """
    subjs = {}
    for chunk in doc.noun_chunks:
        if len(chunk.root.text) > 2 and chunk.root.pos_ not in ["NUM", "SYM","PUNCT"]:
            txt = chunk.root.lemma_.lower()
            if txt not in subjs:
                subjs[txt] = set([txt])
            subjs[txt].add(chunk.text.lower())
    subjects_= []
    for word in subjs:
        for phrase in subjs[word]:
            subjects_ += phrase.split(" ")
    subjs["_vocab"] = set(subjects_)
    return subjs

def get_svos(sent):
    """
    input: Spacy processed sentence
    output: dict of subj, dict of v, dict of obj (each word is lemmatized and lowercased)
    each entry in dict has key of lemmatized token, value is actual token (to do traversals with later if needed)
    """
    s = {}
    v = {}
    o = {}
    for token in sent:
        if token.dep_ == 'ROOT':
            v[token.lemma_.lower()] = token
        elif token.dep_ in ["nsubj", "nsubjpass", "csubj","csubjpass", "agent","compound"]:
            s[token.lemma_.lower()] = token
        elif token.dep_ in ["dobj", "dative", "attr", "oprd", "pobj"]:
            o[token.lemma_.lower()] = token
    # https://github.com/clir/clearnlp-guidelines/blob/master/md/specifications/dependency_labels.md
    return (s,v,o)

def build_graph(doc):
    """
    build a NetworkX graph of the dependency tree
    input: spacy Doc
    output: networkx graph
    """
    edges = set()
    for token in doc:
        if token.pos_ not in ['SPACE']:
            for child in token.children:
                if child.pos_ not in ['SPACE']:
                    edges.add((token.lemma_.lower(),child.lemma_.lower()))
    graph = nx.DiGraph(list(edges))
    return graph

def get_edges(doc):
    """
    return list of edges
    """
    edges = []
    for token in doc:
        if token.pos_ not in ['SPACE', 'PUNCT', 'SYM']:
            for child in token.children:
                if child.pos_ not in ['SPACE', 'PUNCT', 'SYM']:
                    edges.append((
                        {"token":token.lemma_.lower(), "dep":token.dep_ , "pos":token.pos_},
                        {"token":child.lemma_.lower(), "dep":child.dep_ , "pos":child.pos_}
                    ))
    return edges

def get_summary(doc, subjects, n = 5):
    """
    get summary of n sentences in document
    first meaningful sentence will always be returned
    """
    subjects_ = subjects
    def score_sentence(sent):
        # not very robust right now
        score = 0
        word_count = 0
        for token in sent:
            word_count += 1
            t = token.lemma_.lower()
            if t in subjects_:
                score += 1
            elif t in negating_words or t in doubting_words or t in hedging_words:
                score += 1.5
        return score/word_count if word_count > 4 else 0
    sentences = [s for s in doc.sents]
    scored_sentences = [[idx, sent, score_sentence(sent)] for idx, sent in enumerate(sentences)]
    scored_sentences = [s for s in scored_sentences if s[2] > 0 and s[0] > 0] #filter out non-scoring sentences
    scored_sentences.sort(key = lambda x: x[2], reverse = True)
    top = scored_sentences[:n]
    top.sort(key = lambda x: x[0])
    scored_sentences.sort(key = lambda x: x[0])
    result = None
    if len(scored_sentences) == 0:
        result = [sentences[0]]
    else:
        result = [scored_sentences[0][1]] + [s[1] for s in top]
    return result

def get_shortest_path_to_negating(graph, subjects):
    """
    get the shortest path from each subject to any negating or doubting/hedging word
    returns: dictionary with subject as key, and 2-element list of path lengths [negating, doubting]
    - if a subject does not exist in graph or have a path to any negating word, then the value will be [None, None]
    """
    results = {}
    for s in subjects:
        results[s] = [None, None, None]
        if graph.has_node(s):
            for word in negating_words:
                if word in graph:
                    try:
                        path = nx.shortest_path(graph, source = s, target = word)
                        if results[s][0] == None or len(path) < results[s][0]:
                            results[s][0] = len(path)
                    except:
                        continue
            for word in hedging_words:
                if word in graph:
                    try:
                        path = nx.shortest_path(graph, source = s, target = word)
                        if results[s][1] == None or len(path) < results[s][1]:
                            results[s][1] = len(path)
                    except:
                        continue
            for word in doubting_words:
                if word in graph:
                    try:
                        path = nx.shortest_path(graph, source = s, target = word)
                        if results[s][2] == None or len(path) < results[s][2]:
                            results[s][2] = len(path)
                    except:
                        continue
    return results

def root_distance(graph, root):
    """
    as implemented in the Emergent paper - return the shortest distance between the given root and any 
    doubting or hedging words in the graph, or None if no such path exists
    """
    if root == None:
        return None
    min_dist = None
    for word in sus_words:
        if word in graph:
            try:
                path = nx.shortest_path(graph, source = root, target = word)
                if min_dist == None or len(path) < min_dist:
                    min_dist = len(path)
            except:
                continue
    return min_dist

def get_neg_ancestors(doc):
    """
    get the ancestors of every negating word
    input: spacy Doc
    returns: tuple  - set of words that were in the ancestor list of negating words, 
    set of words that were in ancestor list of refuting words, # negating words, # refuting words
    """
    results = [set(), set(), set(), 0, 0, 0]
    for token in doc:
        if token.lemma_.lower() in negating_words:
            results[0] = results[0].union(
                set([ancestor.lemma_.lower() for ancestor in token.ancestors if len(ancestor) > 2]).union(
                    set([child.lemma_.lower() for child in token.head.children if child.text != token.text and len(child) > 2])
                )
            )
            results[3] += 1
        elif token.lemma_.lower() in doubting_words:
            results[1] = results[1].union(
                set([ancestor.lemma_.lower() for ancestor in token.ancestors if len(ancestor) > 2]).union(
                    set([child.lemma_.lower() for child in token.head.children if child.text != token.text and len(child) > 2])
                )
            )
            results[4] += 1
        elif token.lemma_.lower() in hedging_words:
            results[2] = results[1].union(
                set([ancestor.lemma_.lower() for ancestor in token.ancestors if len(ancestor) > 2]).union(
                    set([child.lemma_.lower() for child in token.head.children if child.text != token.text and len(child) > 2])
                )
            )
            results[5] += 1
    return tuple(results)

sp = SenticPhrase("Hello, World!")

def get_sentics(sent):
    """
        input: Spacy processed sentence
        output: a tuple containing the polarity score and a list of sentic values 
            (pleasantness, attention, sensitiviy, aptitude )
    """
    info = sp.info(sent)
          
    # Sometimes sentic doesn't returns any sentics values, seems to be only when purely neutral. 
    # Some sort of tag to make sure this is true could help with classiciation! (if all 0's not enough)
    sentics = {"pleasantness":0, "attention":0, "sensitivity":0, "aptitude":0}
    sentics.update(info["sentics"])
    return [info['polarity'], sentics['aptitude'], sentics['attention'], sentics['sensitivity'], sentics['pleasantness']]

def process_sentence(sentence):
    svo = get_svos(sentence)

    # list of words that belong to that part of speech
    nouns = []
    verbs = []
    adjectives = []
    adverbs = []
    tokens = []
    for token in sentence:
        if not token.is_stop and token.pos_ not in ['PUNCT', 'NUM', 'SYM','SPACE','PART']:
            if token.pos_ == "NOUN":
                nouns.append(token.lemma_.lower())
            elif token.pos_ == "VERB":
                verbs.append(token.lemma_.lower())
            elif token.pos_ == "ADJ":
                adjectives.append(token.lemma_.lower())
            elif token.pos_ == "ADV":
                adverbs.append(token.lemma_.lower())
            tokens.append(token.lemma_.lower())   
    
    bigram = list(nltk.bigrams(tokens))
    bigram_str = [x[0]+' '+x[1] for x in bigram]

    return {
        "raw": sentence.text,
        "tokens": tokens,
        "bigrams": bigram_str,
        "nouns": nouns,
        "verbs": verbs,
        "adjectives": adjectives,
        "adverbs": adverbs,
        "svo": [list(item) for item in svo]
    }

def score_sentence_idf(sent, idf):
    # not very robust right now
    score = 0
    word_count = 0
    for token in sent:
        word_count += 1
        t = token.lemma_.lower()
        if t in idf:
            score += idf[t]
    return score/word_count if word_count > 4 else 0

def process_body(body, idf):
    sentences = [s for s in body.sents]
    if len(sentences) == 0:
        sentences = [body]

    # first sentence of article
    first_sentence_data = process_sentence(sentences[0])

    nouns = []
    verbs = []
    adjectives = []
    adverbs = []
    tokens = []
    for token in body:
        if not token.is_stop and token.pos_ not in ['PUNCT', 'NUM', 'SYM','SPACE','PART']:
            if token.pos_ == "NOUN":
                nouns.append(token.lemma_.lower())
            elif token.pos_ == "VERB":
                verbs.append(token.lemma_.lower())
            elif token.pos_ == "ADJ":
                adjectives.append(token.lemma_.lower())
            elif token.pos_ == "ADV":
                adverbs.append(token.lemma_.lower())
            tokens.append(token.lemma_.lower())   
    
    bigram = list(nltk.bigrams(tokens))
    bigram_str = [x[0]+' '+x[1] for x in bigram]

    doc_len = len(tokens)
    n_counter = Counter(nouns)
    v_counter = Counter(verbs)
    b_counter = Counter(bigram)
    t_counter = Counter(tokens)

    avg_idf = idf["_avg"]
    n_tfidf, v_tfidf, t_tfidf = {}, {}, {}
    for n in n_counter:
        n_tfidf[n] = (n_counter[n]/doc_len) *             (idf[n] if n in idf else avg_idf)
    for v in v_counter:
        v_tfidf[v] = (v_counter[v]/doc_len) *             (idf[v] if v in idf else avg_idf)
    for t in t_counter:
        t_tfidf[t] = (t_counter[t]/doc_len) *             (idf[t] if t in idf else avg_idf)
    
    common_nouns = sorted(n_tfidf, key=n_tfidf.get, reverse=True)[:5]
    common_verbs = sorted(v_tfidf, key=v_tfidf.get, reverse=True)[:5]
    common_tokens = sorted(t_tfidf, key=t_tfidf.get, reverse=True)[:5]

    # no idf for bigrams increase "common" count to 10
    common_bigrams = [x[0] for x in b_counter.most_common(10)]
    
    scored_sentences = [[idx, sent, score_sentence_idf(sent, idf)] for idx, sent in enumerate(sentences)]
    scored_sentences = [s for s in scored_sentences] #filter out non-scoring sentences
    scored_sentences.sort(key = lambda x: x[2], reverse = True)
    most_significant_sentence_data = process_sentence(scored_sentences[0][1])

    return {
        "tokens": tokens,
        "bigrams": bigram_str,
        "nouns": nouns,
        "verbs": verbs,
        "first_sentence": first_sentence_data,
        "significant_sentence": most_significant_sentence_data,
        "vocabulary": list(set(tokens)),
        "common_tokens": common_tokens,
        "common_nouns": common_nouns,
        "common_verbs": common_verbs,
        "common_bigrams": common_bigrams,
    }

def bow_cos_similarity(a, b):
    vocab = list(set(a).union(set(b)))
    a_bow, b_bow = set(a), set(b)
    if len(a) == 0 or len(b) == 0:
        return -1
    a_vec = [(1 if i in a_bow else 0) for i in vocab]
    b_vec = [(1 if i in b_bow else 0) for i in vocab]
    return 1 - distance.cosine(a_vec, b_vec)

def get_features_rel(headline_data, body_data):
    features = []
    for item in body_data:
        h, b = headline_data, item
        fts = get_feats_rel(h, b)
        features.append(fts)
    return features

def get_feats_rel(headline_data, body_data):
    shared_common_nouns = len(set(headline_data['nouns']).intersection(
        set(body_data['common_nouns'])))
    shared_common_verbs = len(set(headline_data['verbs']).intersection(
        set(body_data['common_verbs'])))
    shared_common_tokens = len(set(headline_data['tokens']).intersection(
        set(body_data['common_tokens'])))
    shared_bigrams = len(set(headline_data['bigrams']).intersection(
        set(body_data['common_bigrams'])))

    shared_nouns_first = len(set(headline_data['nouns']).intersection(
        set(body_data['first_sentence']['nouns'])))
    shared_verbs_first = len(set(headline_data['verbs']).intersection(
        set(body_data['first_sentence']['verbs'])))
    shared_bigrams_first = len(set(headline_data['bigrams']).intersection(
        set(body_data['first_sentence']['bigrams'])))
    shared_tokens_first = len(set(headline_data['tokens']).intersection(
        set(body_data['first_sentence']['tokens'])))

    shared_nouns_sig = len(set(headline_data['nouns']).intersection(
        set(body_data['significant_sentence']['nouns'])))
    shared_verbs_sig = len(set(headline_data['verbs']).intersection(
        set(body_data['significant_sentence']['verbs'])))
    shared_bigrams_sig = len(set(headline_data['bigrams']).intersection(
        set(body_data['significant_sentence']['bigrams'])))
    shared_tokens_sig = len(set(headline_data['tokens']).intersection(
        set(body_data['significant_sentence']['tokens'])))

    headline_svo = headline_data['svo']
    body_fst_svo = body_data['first_sentence']['svo']
    body_sig_svo = body_data['significant_sentence']['svo']

    # cosine similarity - no verbs because relatively few per sentence
    cos_nouns_first = bow_cos_similarity(
        headline_data['nouns'], body_data['first_sentence']['nouns'])
    cos_bigrams_first = bow_cos_similarity(
        headline_data['bigrams'], body_data['first_sentence']['bigrams'])
    cos_tokens_first = bow_cos_similarity(
        headline_data['tokens'], body_data['first_sentence']['tokens'])

    cos_nouns_sig = bow_cos_similarity(
        headline_data['nouns'], body_data['significant_sentence']['nouns'])
    cos_bigrams_sig = bow_cos_similarity(
        headline_data['bigrams'], body_data['significant_sentence']['bigrams'])
    cos_tokens_sig = bow_cos_similarity(
        headline_data['tokens'], body_data['significant_sentence']['tokens'])
    
    svo_cos_sim_fst = bow_cos_similarity(
        body_fst_svo[0]+body_fst_svo[1]+body_fst_svo[2], 
        headline_svo[0]+headline_svo[1]+headline_svo[2])

    svo_cos_sim_sig = bow_cos_similarity(
        body_sig_svo[0]+body_sig_svo[1]+body_sig_svo[2], 
        headline_svo[0]+headline_svo[1]+headline_svo[2])
    
    svo_s_fst = len(set(body_fst_svo[0]).intersection(set(headline_svo[0]))) 
    svo_v_fst = len(set(body_fst_svo[1]).intersection(set(headline_svo[1])))
    svo_o_fst = len(set(body_fst_svo[2]).intersection(set(headline_svo[2])))
    svo_s_sig = len(set(body_sig_svo[0]).intersection(set(headline_svo[0])))
    svo_v_sig = len(set(body_sig_svo[1]).intersection(set(headline_svo[1])))
    svo_o_sig = len(set(body_sig_svo[2]).intersection(set(headline_svo[2])))
    
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

        'svo_cos_sim_fst' : svo_cos_sim_fst,
        'svo_cos_sim_sig' : svo_cos_sim_sig,
        
        'svo_s_fst': svo_s_fst,
        'svo_v_fst': svo_v_fst,
        'svo_o_fst': svo_o_fst,

        'svo_s_sig': svo_s_sig,
        'svo_v_sig': svo_v_sig,
        'svo_o_sig': svo_o_sig,
    }

def get_sentence_vec(s):
    vecs = [token.vector for token in s]
    return np.nan_to_num(np.product(vecs, axis = 0))

def get_features_stance(headline_data, body_data, n_sent = 5):
    features = []
    summary_graphs = []
    for item in body_data:
        headline, headline_graph, headline_subjs, headline_svo, headline_root_dist, headline_neg_ancestors, headline_edges = headline_data
        body, body_graph = item
        
        h_important_words = set(headline_subjs["_vocab"]).union(set(headline_svo[0])).union(set(headline_svo[1])).union(set(headline_svo[2]))
        
        #sometimes the coref deletes bodies that are one sentence
        if len(body) == 0:
            body = nlp(preprocess(get_body(b)))
            body_graph = build_graph(body)

        #return the shortest path to negating word for each subject in headline_subjs, if one exists
        neg_h = get_shortest_path_to_negating(headline_graph, h_important_words)
        neg_b = get_shortest_path_to_negating(body_graph, h_important_words)

        #body summary
        summary = get_summary(body, h_important_words, n_sent)
        first_summ_sentence = summary[0]
        summary_edges = [get_edges(s) for s in summary]
        summary_graph = get_display_graph(headline, summary)
        
        summary_svos = [get_svos(s) for s in summary]
        summary_root_dist = [root_distance(body_graph, list(s[1].keys())[0]) for s in summary_svos]
        summary_neg_ancestors = [get_neg_ancestors(s) for s in summary]
        summary_neg_counts = [s[3:] for s in summary_neg_ancestors]
        
        summary_neg_ancestors_superset = [set(), set(), set()]
        for a in summary_neg_ancestors:
            summary_neg_ancestors_superset[0] = summary_neg_ancestors_superset[0].union(a[0])
            summary_neg_ancestors_superset[1] = summary_neg_ancestors_superset[1].union(a[1])
            summary_neg_ancestors_superset[2] = summary_neg_ancestors_superset[2].union(a[2])
            
        #ancestors
        h_anc = [[1 if w in headline_neg_ancestors[0] else -1 for w in h_important_words],
                [1 if w in headline_neg_ancestors[1] else -1 for w in h_important_words],
                [1 if w in headline_neg_ancestors[2] else -1 for w in h_important_words]]
        b_anc = [[1 if w in summary_neg_ancestors_superset[0] else -1 for w in h_important_words],
                [1 if w in summary_neg_ancestors_superset[1] else -1 for w in h_important_words],
                [1 if w in summary_neg_ancestors_superset[2] else -1 for w in h_important_words]]    
        neg_anc_sim = cosine_similarity(h_anc[0], b_anc[0])
        doubt_anc_sim = cosine_similarity(h_anc[1], b_anc[1])
        hedge_anc_sim = cosine_similarity(h_anc[2], b_anc[2])
        neg_anc_overlap = len(headline_neg_ancestors[0].union(summary_neg_ancestors_superset[0]))
        doubt_anc_overlap = len(headline_neg_ancestors[1].union(summary_neg_ancestors_superset[1]))
        hedge_anc_overlap = len(headline_neg_ancestors[2].union(summary_neg_ancestors_superset[2]))
        
        #svo
        body_s, body_v, body_o = {}, {}, {}
        headline_s, headline_v, headline_o = headline_svo
        for svo in summary_svos:
            body_s.update(svo[0])
            body_v.update(svo[1])
            body_o.update(svo[2])
        body_s_vec = list(np.sum([body_s[s].vector for s in body_s], axis = 0)) if len(body_s) > 0 else np.zeros(384)
        body_v_vec = list(np.sum([body_v[s].vector for s in body_v], axis = 0)) if len(body_v) > 0 else np.zeros(384)
        body_o_vec = list(np.sum([body_o[s].vector for s in body_o], axis = 0)) if len(body_o) > 0 else np.zeros(384)
    
        headline_s_vec = list(np.sum([headline_s[s].vector for s in headline_s], axis = 0)) if len(headline_s) > 0 else np.zeros(384)
        headline_v_vec = list(np.sum([headline_v[s].vector for s in headline_v], axis = 0)) if len(headline_v) > 0 else np.zeros(384)
        headline_o_vec = list(np.sum([headline_o[s].vector for s in headline_o], axis = 0)) if len(headline_o) > 0 else np.zeros(384)
        
        cos_sim_s = cosine_similarity(body_s_vec, headline_s_vec)
        cos_sim_v = cosine_similarity(body_v_vec, headline_v_vec)
        cos_sim_o = cosine_similarity(body_o_vec, headline_o_vec)
        
        #negating paths
        headline_paths = [neg_b[x] for x in neg_b]
        headline_neg_paths = [1 if x[0] != None else -1 for x in headline_paths]
        headline_doubt_paths = [1 if x[1] != None else -1 for x in headline_paths]
        headline_hedge_paths = [1 if x[2] != None else -1 for x in headline_paths]
        body_paths = [neg_h[x] for x in neg_h]
        body_neg_paths = [1 if x[0] != None else -1 for x in body_paths]
        body_doubt_paths = [1 if x[1] != None else -1 for x in body_paths]
        body_hedge_paths = [1 if x[2] != None else -1 for x in body_paths]
        
        neg_path_cos_sim = cosine_similarity(headline_neg_paths, body_neg_paths)
        hedge_path_cos_sim = cosine_similarity(headline_hedge_paths, body_hedge_paths)
        doubt_path_cos_sim = cosine_similarity(headline_doubt_paths, body_doubt_paths)
        
        #root distance
        summary_root_dists = [x if x != None else 15 for x in summary_root_dist]
        avg_summary_root_dist = sum(summary_root_dists)/len(summary_root_dists)
        root_dist_feats = [headline_root_dist, avg_summary_root_dist]
        root_dist_feats = [x/15 if x != None else 1 for x in root_dist_feats]
        root_dist_feats = root_dist_feats + [int(headline_root_dist == None), len([x for x in summary_root_dist if x != None])]
    
        #sentiment
        headline_sent = get_sentiment(headline.text)
        body_sents = [get_sentiment(s.text) for s in summary]
        avg_body_sent = list(np.mean(body_sents, axis = 0))
        diff_avg_sents = list(np.array(headline_sent) - avg_body_sent)
        diff_sents = list(np.sum([get_diff_sentiment(headline_sent, s) for s in body_sents], axis = 0))
        sent_cos_sim = cosine_similarity(headline_sent, avg_body_sent)

        headline_sentics = get_sentics(headline.text)
        body_sentics = [get_sentics(s.text) for s in summary]
        avg_body_sentics = list(np.mean(body_sentics, axis = 0))
        diff_avg_sentics = list(np.array(headline_sentics) - avg_body_sentics)
        diff_sentics = list(np.sum([get_diff_sentiment(headline_sentics, s) for s in body_sentics], axis = 0))
        sentics_cos_sim = cosine_similarity(headline_sentics, avg_body_sentics)
        
        #bow
        headline_vocab = set([tok.lemma_.lower() for tok in headline])
        fst_summ_vocab = set([tok.lemma_.lower() for tok in first_summ_sentence])
        total_vocab = list(headline_vocab.union(fst_summ_vocab))
        headline_embedding = [1 if tok in headline_vocab else -1 for tok in total_vocab]
        fst_summ_embedding = [1 if tok in fst_summ_vocab else -1 for tok in total_vocab]
        bow_cos_sim = cosine_similarity(headline_embedding, fst_summ_embedding)
        
        #word vecs
        cos_sims = [cosine_similarity(get_sentence_vec(s), headline.vector) for s in summary]
        fst_cos_sim = cos_sims[0]
        avg_cos_sim = sum(cos_sims)/len(cos_sims)
        
        #neg_hedge_doubt distributions
        hd_dist = list(headline_neg_ancestors[3:])
        body_dist = list(np.sum(summary_neg_counts, axis = 0))
        dist_sim = cosine_similarity(hd_dist, body_dist)
        
        #build final features list
        fts = (
            [fst_cos_sim, avg_cos_sim, bow_cos_sim, 
                neg_path_cos_sim, hedge_path_cos_sim, doubt_path_cos_sim,
                neg_anc_sim, hedge_anc_sim, doubt_anc_sim,
                neg_anc_overlap, hedge_anc_overlap, doubt_anc_overlap,
                cos_sim_s, cos_sim_v, cos_sim_o,
                dist_sim, sent_cos_sim, sentics_cos_sim] + 
            diff_avg_sents + diff_sents + diff_avg_sentics + diff_sentics + 
            root_dist_feats + hd_dist + body_dist +
            headline_sent + avg_body_sent + headline_sentics + avg_body_sentics
        )
        features.append(fts)
        summary_graphs.append(summary_graph)
    return features, summary_graphs, headline_subjs
