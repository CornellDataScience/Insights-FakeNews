import spacy
import json

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

nlp = spacy.load('en_core_web_sm')
headline = """
Julian Assange must face Swedish justice first - MPs and peers
"""
body = """
More than 70 MPs and peers have signed a letter urging the home secretary to ensure Julian Assange faces authorities in Sweden if they want his extradition.

The Wikileaks founder, who is now in UK custody, was arrested on Thursday after years in Ecuador's London embassy.

Sweden is considering whether to reopen an investigation into rape and sexual assault allegations against him.

And the US is seeking his extradition in relation to one of the largest ever leaks of government secrets, in 2010.

The whistle-blowing website Wikileaks has published thousands of classified documents covering everything from the film industry to national security and war.
"""

headline, body = nlp(headline), nlp(body)
headline_root = [t for t in headline if t.dep_== "ROOT"][0]
body_sents = [s for s in body.sents]
body_roots = [[t for t in sent if t.dep_== "ROOT"][0] for sent in body_sents]
headline_graph = make_graph(headline_root)
body_graphs = [make_graph(r) for r in body_roots]
with open("test_deps_tree.json","w") as f:
    json.dump({"headline":headline_graph, "body":body_graphs}, f)



