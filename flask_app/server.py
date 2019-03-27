from flask import Flask, render_template, request, jsonify
from server_helpers import *
import json
from joblib import dump, load

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify_multiple', methods = ['POST'])
def classify_multiple():
    headline = request.form.get('headline')
    stances = pd.read_csv("train_stances.csv")
    bodies = pd.read_csv("train_bodies.csv")
    def get_body(n):
        bodies.loc[lambda x: x["Body ID"] == n, "articleBody"].item()
    relevant_stances = stances[stances["Headline"]==headline]
    relevant_bodies = [get_body(i[1]) for i in relevant_stances.values()]
    results = classify_helper(headline, bodies)
    return jsonify(results)

@app.route('/classify', methods = ['POST'])
def classify():
    headline = request.form.get('headline')
    body = request.form.get('body')
    results = classifiy_helper(headline, [body])
    return jsonify(results)

@app.route('/classify_test', methods = ['POST'])
def classify():
    headline = ""
    body = ""
    results = classifiy_helper(headline, [body])
    return jsonify(results)

# helper to to most of the heavy lifting- takes in 1 headline and a list of bodies (as strings of text)
def classify_helper(headline, bodies):
    idf = {}
    with open('idf.json',"r") as idf_file:
        idf = json.load(idf_file)
    nlp_h = nlp(preprocess(h))
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
        body_info.append(nlp_b, body_graph)

    features_rel = get_features_rel(headline_info_rel, body_info_rel)
    features_stance, summary_graphs = get_features_stance(headline_info, body_info)

    rel_model = load('saved_models/relevance_detection_trained.joblib')
    stance_model = load('saved_models/stance_detection_trained.joblib')

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
        }
    }
