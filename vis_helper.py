import torch
import pandas as pd
import nltk
import numpy as np
import preprocessing, feature_engineering, helpers
import importlib
import re
import nltk
from nltk.corpus import stopwords
import helpers
import json

class Vis_Helper():

    def __init__(self):
        self.preprocess = preprocessing.Preprocessing()

    """
        Creates a dictionary that maps the input text to the cell value. 
        Input includes both headline and body. 
        Last 40 elements of output are the body. 
        Remaining elements of the output are the headline. 
    """
    def get_values(self, text, tokens, cell):
        j = 0 # index in tokens for duplicate token values
        body = [{} for i in range(len(text))]
            
        for i in range(len(text)):
            test = self.preprocess.clean(text[i])
            test = self.preprocess.get_tokenized_lemmas(test)
            test = self.preprocess.remove_stopwords(test, True)
            if(len(test)==0): 
                body[i] = {text[i]:str(0)}
                #print(text_body[i], 0)
            else:
                #token_index = np.where(tokens[j:]==test[0])
                index = list(tokens[j:]).index(test[0])
                body[i] = {text[i]:str(cell[index])} # Try index vs. j 
                j+=1
        return body


    """
        For every cell in the ouput, creates a dict in the format of get_values and returns them all as a list. 
        text_body/text_headline: the raw (unprocessed) text of the body or headline
        output: the modified output of the model of shape [num_words, hidden_dim]
    """
    def get_all_cell_activations(self, text_body, text_headline, output):
        act_cells = []
        all_cells = np.array(np.swapaxes(output, 0, 1))
        i = 0

        for cell in all_cells:
            # Get the full headline text
            #body = stances_val.iloc[0]["Body ID"]
            #text_body = preprocess.get_body(body,train_bodies)
            #text_headline = stances_val.iloc[0]["Headline"]

            # Get the tokens that are actually fed into the network
            tokens_body = self.preprocess.get_clean_tokens(text_body, False)[:40]
            tokens_headline = self.preprocess.get_clean_tokens(text_headline, False)[:20]
            tokens = np.concatenate((tokens_headline, tokens_body))

            list_body = text_body.split(" ")
            list_headline = text_headline.split(" ")
            text = np.concatenate((list_headline[:20], list_body[:40]))

            v = self.get_values(text, tokens, cell)
            values_body = v[-40:] # Gets the body text
            values_headline = v[:-40] # Gets the headline text
            values_json = {"body":values_body, "headline":values_headline,"cell_number":str(i)}
            
            i+=1
            act_cells.append(values_json)

        return act_cells

    
    """
        For every cell in the ouput, creates a dict in the format of get_values and returns them all as a list. 
        Assumes that only working with headline or body, ie for Siamese
        text: either headline or body str. 
        output: the modified output of the model of shape [num_words, hidden_dim]
    """
    def get_all_activations_separate(self, text, output, is_headline):
        act_cells = []
        all_cells = np.array(np.swapaxes(output, 0, 1))
        i = 0
        n=0

        if(is_headline): n = 20
        else: n = 40

        for cell in all_cells:
            # Get the full headline text
            #body = stances_val.iloc[0]["Body ID"]
            #text_body = preprocess.get_body(body,train_bodies)
            #text_headline = stances_val.iloc[0]["Headline"]

            # Get the tokens that are actually fed into the network
            tokens = self.preprocess.get_clean_tokens(text, False)[:n]
            

            list_text = text.split(" ")
            text = np.array(list_text)[:n]
            v = self.get_values(text, tokens, cell)

            values_json = {"values":v,"cell_number":str(i)}
            
            i+=1
            act_cells.append(values_json)

        return act_cells

    def get_important_cells(self, output, hidden_dim):
        output_cov= np.corrcoef(output, rowvar=False)

        # Cell with highest magnitude: high response overall to the whole sequence
        output2_norm =[np.linalg.norm(row) for row in output.T]
        output3_norm = np.argsort(output2_norm)
        output_norm = np.array(output.T)[output3_norm]
        output_highMag = output_norm[:5] # cells with highest mangitude

        # Cell with smallest magnitude: low response
        output_lowMag = output_norm[-5:] # cells with 5 lowest magnitude

        # Cell with with greatest variance: 
        output2_var = [np.var(row) for row in output.T]
        output3_var = np.argsort(output2_var)
        output_var = np.array(output.T)[output3_var]
        output_highVar = output_var[:5]

        # Cell with least variance
        output_lowVar = output_var[-5:]

        # Get all the cells that highly correlate
        # Can the number of cells with high correlation tell us to decrease the hidden dim? Could be redundant info/overfitting...
        np.fill_diagonal(output_cov,0)
        output2_cov = np.argsort(np.triu(np.abs(output_cov)).flatten()) # take 1 triangle to avoid duplicates
        cov_pairs = []
        for i in range(1,6):
            #print(output2_cov[i])
            #print("output shape", output_cov.shape)
            #print("maxes",np.argmax(output_cov), np.max(output_cov))
            row = output2_cov[-i] % hidden_dim
            col = output2_cov[-i] // hidden_dim
            #print("row",row)
            #print("col",col)
            #print([row,col],output_cov[row,col])
            cov_pairs.append([row,col])

        return {
            "bot5_mag":[str(i) for i in output3_norm[:5]],
            "top5_mag":[str(i) for i in output3_norm[-5:]],
            "top5_var":[str(i) for i in output3_var[:5]],
            "bot5_var":[str(i) for i in output3_var[-5:]],
            "top5_corr":[[str(i[0]), str(i[1])] for i in cov_pairs]
        }



    def write_json(self, filepath, data):
        with open(filepath, "w") as outfile:
            json.dump(data, outfile)

