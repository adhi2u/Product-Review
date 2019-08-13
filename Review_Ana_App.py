import numpy as np
import pandas as pd
import string
import warnings
warnings.filterwarnings('ignore')
from sklearn.feature_extraction.text import CountVectorizer
import re
from pattern.en import tag
from nltk.corpus import wordnet as wn
# global Amazon_quality
CONTRACTION_MAP = {
"ain't": "is not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"dont": "do not",  
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"I'd": "I would",
"I'd've": "I would have",
"I'll": "I will",
"I'll've": "I will have",
"I'm": "I am",
"I've": "I have",
"i'd": "i would",
"i'd've": "i would have",
"i'll": "i will",
"i'll've": "i will have",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have"
}
import nltk
stopword_list = nltk.corpus.stopwords.words('english')
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
stopword_list.remove("don't")
stopword_list.remove("not")
stopword_list.remove("shouldn't")
stopword_list.remove("didn't")
stopword_list.remove("aren't")
stopword_list.remove("is")
stopword_list.remove("are")
stopword_list.remove("was")
stopword_list.remove("but")
stopword_list.remove("were")
stopword_list.remove("wasn't")
stopword_list.remove("on")
stopword_list.remove("no")
stopword_list.remove("with")
stopword_list.remove("couldn't")

def tokenize_text(text):
    tokens = nltk.word_tokenize(text) 
    tokens = [token.strip() for token in tokens]
    return tokens

def expand_contractions(text, contraction_mapping):
    
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())                       
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
        
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text

def remove_special_characters(text):
    tokens = tokenize_text(text)
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filtered_tokens = filter(None, [pattern.sub('', token) for token in tokens])
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

def remove_stopwords(text):
    tokens = tokenize_text(text)
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text

def normalize_corpus(corpus, tokenize=False):
    
    normalized_corpus = []    
    for text in corpus:
        text = expand_contractions(text, CONTRACTION_MAP)
        text =text.lower()
#        text = lemmatize_text(text)
        text = remove_special_characters(text)
        text = remove_stopwords(text)
        normalized_corpus.append(text)
        if tokenize:
            text = tokenize_text(text)
            normalized_corpus.append(text)
            
    return normalized_corpus

def remove_empty_docs(corpus, labels):
   filtered_corpus = []
   filtered_labels = []
   for doc, label in zip(corpus, labels):
       if doc.strip():
          filtered_corpus.append(doc)
          filtered_labels.append(label)
   return filtered_corpus, filtered_labels

def getResults(corpus1):
        corpus = pd.Series(corpus1)
        norm_corpus = normalize_corpus(corpus)
        norm_str_corpus = "".join(map(str, norm_corpus))

        Amazon_new = pd.DataFrame(data=norm_corpus, columns=['Text'])
        Amazon_quality = pd.read_excel("Smart-review v1.xlsx", encoding='latin1')

        Amazon_quality=pd.concat([Amazon_quality[col].astype(str).str.lower() for col in Amazon_quality.columns], axis=1)

        search_pos_qual_values = Amazon_quality["Good_Quality"].dropna().tolist()
        search_pos_qual_values = Amazon_quality["Good_Quality"].dropna().tolist()
        search_neg_qual_values = Amazon_quality["Incompetent_Quality"].dropna().tolist()
        search_pos_dlvry_values = Amazon_quality["Good_Delivery"].dropna().tolist()
        search_neg_dlvry_values = Amazon_quality["Incompetent_Delivery"].dropna().tolist()
        search_pos_pkge_values = Amazon_quality["Good_Package"].dropna().tolist()
        search_neg_pkge_values = Amazon_quality["Incompetent_Package"].dropna().tolist()
        search_pos_srvce_values = Amazon_quality["Good_Service"].dropna().tolist()
        search_neg_srvce_values = Amazon_quality["Incompetent_Service"].dropna().tolist()

        POS_QUAL_CNT=NEG_QUAL_CNT=POS_DLVRY_CN=NEG_DLVRY_CNT=POS_PKGE_CNT=NEG_PKGE_CNT=0
        POS_SRVCE_CNT=NEG_SRVCE_CNT=0

        POS_QUAL_CNT=Amazon_new["Text"].str.count('|'.join(search_pos_qual_values),re.I)
        NEG_QUAL_CNT=Amazon_new["Text"].str.count('|'.join(search_neg_qual_values),re.I)

        POS_DLVRY_CNT=Amazon_new["Text"].str.count('|'.join(search_pos_dlvry_values),re.I)
        NEG_DLVRY_CNT=Amazon_new["Text"].str.count('|'.join(search_neg_dlvry_values),re.I)

        POS_PKGE_CNT=Amazon_new["Text"].str.count('|'.join(search_pos_pkge_values),re.I)
        NEG_PKGE_CNT=Amazon_new["Text"].str.count('|'.join(search_neg_pkge_values),re.I)

        POS_SRVCE_CNT=Amazon_new["Text"].str.count('|'.join(search_pos_srvce_values),re.I)
        NEG_SRVCE_CNT=Amazon_new["Text"].str.count('|'.join(search_neg_srvce_values),re.I)

        if POS_QUAL_CNT.loc[0] >=2 and NEG_QUAL_CNT.loc[0] ==0:
                QUALITY_RATING = 5
        elif POS_QUAL_CNT.loc[0] ==1 and NEG_QUAL_CNT.loc[0] ==0:
                QUALITY_RATING = 4
        elif NEG_QUAL_CNT.loc[0] >=1 and POS_QUAL_CNT.loc[0]==0:
                QUALITY_RATING = 1
        elif NEG_QUAL_CNT.loc[0] > POS_QUAL_CNT.loc[0] :
                QUALITY_RATING = 2
        elif (POS_QUAL_CNT.loc[0] > NEG_QUAL_CNT.loc[0]) or (POS_QUAL_CNT.loc[0] ==1 and NEG_QUAL_CNT.loc[0] ==1):
                QUALITY_RATING = 3
        elif norm_str_corpus in ('awesome','excellent','good','very good','fantastic','super','superb'):
                QUALITY_RATING = 4
        else:
                QUALITY_RATING =0

        if POS_DLVRY_CNT.loc[0] >=2 and NEG_DLVRY_CNT.loc[0] ==0:
                DELIVERY_RATING = 5
        elif POS_DLVRY_CNT.loc[0] ==1 and NEG_DLVRY_CNT.loc[0] ==0:
                DELIVERY_RATING = 4
        elif NEG_DLVRY_CNT.loc[0] >=1 and POS_DLVRY_CNT.loc[0] ==0:
                DELIVERY_RATING = 1
        elif NEG_DLVRY_CNT.loc[0] > POS_DLVRY_CNT.loc[0] :
                DELIVERY_RATING = 2
        elif (POS_DLVRY_CNT.loc[0] > NEG_DLVRY_CNT.loc[0]) or (POS_DLVRY_CNT.loc[0] ==1 and NEG_DLVRY_CNT.loc[0] ==1):
                DELIVERY_RATING = 3
        elif norm_str_corpus in ('awesome','excellent','good','very good','fantastic','super','superb'):
                DELIVERY_RATING = 4
        else:
                DELIVERY_RATING =0

        if POS_PKGE_CNT.loc[0] >=2 and NEG_PKGE_CNT.loc[0] ==0:
                PACKAGE_RATING = 5
        elif POS_PKGE_CNT.loc[0] ==1 and NEG_PKGE_CNT.loc[0] ==0:
                PACKAGE_RATING = 4
        elif NEG_PKGE_CNT.loc[0] >=1 and POS_PKGE_CNT.loc[0] ==0:
                PACKAGE_RATING = 1
        elif NEG_PKGE_CNT.loc[0] > POS_PKGE_CNT.loc[0] :
                PACKAGE_RATING = 2
        elif (POS_PKGE_CNT.loc[0] > NEG_PKGE_CNT.loc[0]) or (POS_PKGE_CNT.loc[0] ==1 and NEG_PKGE_CNT.loc[0] ==1):
                PACKAGE_RATING = 3
        elif norm_str_corpus in ('awesome','excellent','good','very good','fantastic','super','superb'):
                PACKAGE_RATING = 4
        else:
                PACKAGE_RATING =0

        if POS_SRVCE_CNT.loc[0] >=2 and NEG_SRVCE_CNT.loc[0] ==0:
                SERVICE_RATING = 5
        elif POS_SRVCE_CNT.loc[0] ==1 and NEG_SRVCE_CNT.loc[0] ==0:
                SERVICE_RATING = 4
        elif NEG_SRVCE_CNT.loc[0] >=1 and POS_SRVCE_CNT.loc[0] ==0:
                SERVICE_RATING = 1
        elif NEG_SRVCE_CNT.loc[0] > POS_SRVCE_CNT.loc[0] :
                SERVICE_RATING = 2
        elif (POS_SRVCE_CNT.loc[0] > NEG_SRVCE_CNT.loc[0]) or (POS_SRVCE_CNT.loc[0] ==1 and NEG_SRVCE_CNT.loc[0] ==1):
                SERVICE_RATING = 3
        elif norm_str_corpus in ('awesome','excellent','good','very good','fantastic','super','superb'):
                SERVICE_RATING = 4
        else:
                SERVICE_RATING =0

        return QUALITY_RATING, DELIVERY_RATING, PACKAGE_RATING, SERVICE_RATING
