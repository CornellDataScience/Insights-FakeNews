from flask import Flask, render_template, request, jsonify
from server_helpers import *
import json
from joblib import dump, load

app = Flask(__name__)

# helper to to most of the heavy lifting- takes in 1 headline and a list of bodies (as strings of text)
def classify_helper(headline, bodies):
    idf = {}
    with open('idf.json',"r") as idf_file:
        idf = json.load(idf_file)
    nlp_h = nlp(preprocess(headline))
    headline_processed = process_sentence(nlp_h)
    headline_edges = get_edges(nlp_h)
    headline_graph = nx.DiGraph(list(set([(e[0]['token'], e[1]['token']) for e in headline_edges])))
    headline_subj = get_topics(nlp_h)
    headline_svo = get_svos(nlp_h)
    headline_root_dist = root_distance(headline_graph, list(headline_svo[1].keys())[0])
    headline_neg_ancestors = get_neg_ancestors(nlp_h)
    headline_info_rel = headline_processed # ready to dump
    body_info_rel = [] # ready to dump
    headline_info = (nlp_h, headline_graph, headline_subj, headline_svo, headline_root_dist, headline_neg_ancestors, headline_edges) # not ready
    body_info = [] # not ready
    for body in bodies:
        nlp_a = coref(preprocess(body))
        nlp_b = nlp(nlp_a._.coref_resolved.lower())
        body_processed = process_body(nlp_b, idf)
        body_graph = build_graph(nlp_b)
        body_info_rel.append(body_processed)
        body_info.append((nlp_b, body_graph))

    features_rel = get_features_rel(headline_info_rel, body_info_rel)
    features_stance, summary_graphs = get_features_stance(headline_info, body_info)

    rel_model = load('../saved_models/relevance_detection_trained.joblib')
    stance_model = load('../saved_models/stance_detection_trained.joblib')

    testing_data = [[],[]]
    for i in range(len(features_rel)):
        testing_data[0].append(list(features_rel[i].values()))
        testing_data[1].append(features_stance[i])

    rel_predicted = rel_model.predict(testing_data[0])
    stance_predicted = stance_model.predict(np.nan_to_num(np.array(testing_data[1]).astype("float32")))
    predicted = [stance_predicted[i] if rel_predicted[i]!="unrelated" else rel_predicted[i] for i in range(len(rel_predicted))]
    result = {
        "headline": headline,
        "bodies": bodies,
        "graphs": {
            "headline": headline_edges,
            "bodies": summary_graphs
        },
        "relevance_features": {
            "headline": headline_processed,
            "bodies": body_info_rel
        },
        "predictions": predicted
    }
    return result

@app.route('/', methods = ['GET'])
def index():
    return render_template('index.html', data=[])

@app.route('/classify_multiple', methods = ['GET'])
def classify_multiple():
    headline = request.form.get('headline')
    stances = pd.read_csv("train_stances.csv")
    bodies = pd.read_csv("train_bodies.csv")
    def get_body(n):
        bodies.loc[lambda x: x["Body ID"] == n, "articleBody"].item()
    relevant_stances = stances[stances["Headline"]==headline]
    relevant_bodies = [get_body(i[1]) for i in relevant_stances.values()]
    results = classify_helper(headline, relevant_bodies)
    return render_template('index.html', data=results)

@app.route('/classify', methods = ['GET'])
def classify():
    headline = request.form.get('headline')
    body = request.form.get('body')
    results = classify_helper(headline, [body])
    return render_template('index.html', data=results)

@app.route('/classify_test', methods = ['GET'])
def classify_test():
    headline = "Julian Assange must face Swedish justice first - MPs and peers"
    body = """More than 70 MPs and peers have signed a letter urging the home secretary to ensure Julian Assange faces authorities in Sweden if they want his extradition.
    The Wikileaks founder, who is now in UK custody, was arrested on Thursday after years in Ecuador's London embassy.
    Sweden is considering whether to reopen an investigation into rape and sexual assault allegations against him.
    And the US is seeking his extradition in relation to one of the largest ever leaks of government secrets, in 2010.
    The whistle-blowing website Wikileaks has published thousands of classified documents covering everything from the film industry to national security and war."""
    results = classify_helper(headline, [body, body])
    return render_template('index.html', data=results)
