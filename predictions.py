#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import cv2
import pytesseract
from glob import glob
import spacy
import re 
import string
from spacy import displacy
import warnings

warnings.filterwarnings('ignore')

#Load NER Model

model_ner = spacy.load("./output/model-best/")


###Functions and Classes

def cleanText(txt):
    
    txt = str(txt)
    #txt = txt.lower()
    whitespace = string.whitespace
    punctuation = "!#$%&\'()*+:;<=>?[\\]^`{|}~"
    tablewhitespace = str.maketrans('','',whitespace)
    tablepunc = str.maketrans('', '', punctuation)
    txt = txt.translate(tablewhitespace)
    txt = txt.translate(tablepunc)
    
    return str(txt)

class GroupGen():
    
    def __init__(self):
        self.id = 0
        self.text = ''
        
    def getGroup(self, text):
        if self.text == text:
            return self.id
        else:
            self.id +=1 
            self.text = text
            return self.id
        

###Parser
def parser(text, label):
    
    if label == 'PHONE':
        text = text.lower()
        text = re.sub(r'\D', '', text)
        
    elif label == 'EMAIL':
        text = text.lower()
        special_chr = '@_.\-'
        text = re.sub(r'[^A-Za-z0-9{} ]'.format(special_chr),'', text)
        
    elif label == 'WEB':
        text = text.lower()
        special_chr = ':/.%#\-'
        text = re.sub(r'[^A-Za-z0-9{} ]'.format(special_chr),'', text)
        
    elif label == ('NAME', 'DES'):
        text = text.lower()
        text = re.sub(r'[^A-Za-z ]', '', text)
        text = text.title()
        
    elif label == 'ORG':
        text = text.lower()
        text = re.sub(r'[^A-Za-z0-9 ]', '', text)
        text = text.title()
        
    return text

grp_gen = GroupGen()


def prediction(image):

	#Extract data using Pytesseract
	data = pytesseract.image_to_data(image)

	data = list(map(lambda x: x.split('\t'), data.split('\n')))
	df = pd.DataFrame(data[1:], columns = data[0])
	df.dropna(inplace =True)
	df['text'] = df['text'].apply(cleanText)

	#Convert Data into Content
	df_clean = df.query("text != '' ")
	data = " ".join([w for w in df_clean['text']])

	# Prediction from NER model
	doc = model_ner(data)

	doc_json = doc.to_json()
	doc_text = doc_json['text']
	doc_tokens = doc_json['tokens']
	df_doc = pd.DataFrame(doc_tokens)
	df_doc['text'] = df_doc[['start', 'end']].apply(lambda x: doc_text[x[0]:x[1]], axis = 1)

	df_doc_label = pd.DataFrame(doc_json['ents'])[['start', 'label']]
	df_doc = pd.merge(df_doc, df_doc_label, how = 'left', on = 'start')
	df_doc.fillna('O', inplace = True)

	#Merging tesseract and NER data
	df_clean['end'] = df_clean['text'].apply(lambda x: len(x) + 1).cumsum() - 1
	df_clean['start'] = df_clean[['text', 'end']].apply(lambda x: x[1] - len(x[0]), axis = 1)
	df_clean = pd.merge(df_clean, df_doc[['start', 'label']], how = 'inner', on = 'start')


	###Bounding Box

	bb_df = df_clean.query(" label != 'O' ")
	bb_df['label'] = bb_df['label'].apply(lambda x: x[2:])


		
	bb_df['group'] = bb_df['label'].apply(grp_gen.getGroup)
	#right and bottom of bounding box

	bb_df[['left', 'top', 'width', 'height']] = bb_df[['left', 'top', 'width', 'height']].astype(int)
	bb_df['right'] = bb_df['left'] + bb_df['width']
	bb_df['bottom'] = bb_df['top'] + bb_df['height']
	col_group = ['left', 'top', 'right', 'bottom', 'label', 'text', 'group']
	bb_df = bb_df[col_group].groupby(by = 'group').agg({
	    
	    'left': min,
	    'right': max,
	    'top': min,
	    'bottom': max,
	    'label': np.unique,
	    'text': lambda x: ' '.join(x)
	})

	bb_df['label'] = bb_df['label'].apply(lambda x: x[0][:])


	img_bb = image.copy()

	for l, r, t, b, label, text in bb_df.values:
	    
	    cv2.rectangle(img_bb, (l, t), (r, b), (0, 255, 0), 2)
	    cv2.putText(img_bb, label, (l,t),cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255))





	###Entities
	info_array = df_doc[['text', 'label']].values
	entities = dict(NAME = [], ORG = [], DES = [], PHONE = [], EMAIL = [], WEB = [])
	previous = 'O'
	for text, label in info_array:
	    bio_tag = label[:1]
	    label_tag = label[2:]
	    #parse the token
	    text = parser(text, label_tag)
	    if bio_tag in ('B', 'I'):
	    	
	    	if previous != label_tag:
	    		entities[label_tag].append(text)
	    	
	    	else:
		    	if bio_tag == 'B':
		        	entities[label_tag].append(text)
		    	else:
		        	if label_tag in ('NAME', 'ORG', 'DES'):
		            		entities[label_tag][-1] = entities[label_tag][-1] + ' ' + text
		            
		        	else:
		            		entities[label_tag][-1] = entities[label_tag][-1] + text
	    
	    previous = label_tag
    

	return img_bb, entities




