# this file contains code use to generate training dataset with FEVER
import os,json,sys,math
import nltk
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
from collections import Counter
import spacy
import random
import allennlp
from allennlp.predictors.predictor import Predictor
from scipy import spatial
import unicodedata
import numpy as np

class InvertedIndex:
    def __init__(self, vocab, doc_term_freqs):
        self.vocab = vocab
        self.doc_len = [0] * len(doc_term_freqs)
        self.doc_term_freqs = [[] for i in range(len(vocab))]
        self.doc_ids = [[] for i in range(len(vocab))]
        self.doc_freqs = [0] * len(vocab)
        self.total_num_docs = 0
        self.max_doc_len = 0
        for docid, term_freqs in enumerate(doc_term_freqs):
            doc_len = sum(term_freqs.values())
            self.max_doc_len = max(doc_len, self.max_doc_len)
            self.doc_len[docid] = doc_len
            self.total_num_docs += 1
            for term, freq in term_freqs.items():
                term_id = vocab[term]
                self.doc_ids[term_id].append(docid)
                self.doc_term_freqs[term_id].append(freq)
                self.doc_freqs[term_id] += 1

    def num_terms(self):
        return len(self.doc_ids)

    def num_docs(self):
        return self.total_num_docs

    def docids(self, term):
        term_id = self.vocab[term]
        return self.doc_ids[term_id]

    def freqs(self, term):
        term_id = self.vocab[term]
        return self.doc_term_freqs[term_id]

    def f_t(self, term):
        term_id = self.vocab[term]
        return self.doc_freqs[term_id]

    def space_in_bytes(self):
    # this function assumes each integer is stored using 8 bytes
        space_usage = 0
        for doc_list in self.doc_ids:
            space_usage += len(doc_list) * 8
        for freq_list in self.doc_term_freqs:
            space_usage += len(freq_list) * 8
        return space_usage

def get_index(wiki_documents):
    processed_titles =[]
    wiki_titles = []
    for wiki_title in wiki_documents:
        processed_title = get_raw_word(wiki_title)
        processed_titles.append(processed_title)
        wiki_titles.append(wiki_title)
    processed_docs = []
    vocab = {}
    doc_term_freqs = []
    lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
    for raw_doc in processed_titles:
        norm_doc = []
        tokens = nltk.tokenize.word_tokenize(raw_doc)
        for token in tokens:
            term = lemmatizer.lemmatize(token).lower()
            norm_doc.append(term)        
            if term not in vocab.keys():
                vocab[term]= len(vocab)
        processed_docs.append(norm_doc)
    for norm_doc in processed_docs:
        temp = Counter(norm_doc)
        doc_term_freqs.append(temp)
    index = InvertedIndex(vocab, doc_term_freqs)
    return index,wiki_titles

def query_tfidf(query, index, k=5):
    scores = Counter()
    termScore =0
    N= index.num_docs()
    for term in query:
        position=0
        if term not in index.vocab:
            continue
        docids = index.docids(term)
        for docid in docids:
            fdtList= index.freqs(term)
            fdt = fdtList[position]
            ft= index.f_t(term)
            termScore = math.log(1+fdt)*math.log(N/ft)
            if docid not in scores.keys():
                scores[docid] = termScore
            else:
                scores[docid] += termScore
            position +=1
    for docid in scores.keys():
        scores[docid] =(1/math.sqrt(index.doc_len[docid]))*scores[docid]
    return scores.most_common(k)

def file_name(file_dir):   
    L=[]   
    for root, dirs, files in os.walk(file_dir):  
        for file in files:  
            if os.path.splitext(file)[1] == '.txt':  
                L.append(str(file))  
    return L

def wiki_title_preprocess():
    lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
    names = file_name('wiki-pages-text')
    documents={}
    for name in names:
        file = open('wiki-pages-text/'+name, 'r')
        for line_number,line in enumerate(file):
            tokens=line.split(' ')
            title=tokens[0]
            if title not in documents:
                documents[title]={}
            if tokens[1].isdigit():
                sentence_number=int(tokens[1])
                sentence=get_raw_sentence(tokens[2:])
                documents[title][sentence_number]=sentence
        file.close()
    return documents

def get_wiki_document(wiki,document):
    return wiki[document]

def get_raw_sentence(sentence):
    raw_sentence=''
    for word in sentence:
        word=unicodedata.normalize('NFC',word)
        if word=='_':
            raw_sentence+=' '
        elif word=='-LRB-':
            raw_sentence+='( '
        elif word=='-RRB-':
            raw_sentence+=') '
        elif word=='-LCB-':
            raw_sentence+='{ '
        elif word=='-RCB-':
            raw_sentence+='} '
        elif word=='-LSB-':
            raw_sentence+='[ '
        elif word=='-RSB-':
            raw_sentence+='] '
        elif word=='\n':
            continue
        else:
            raw_sentence+=(word+' ')
    return raw_sentence

def get_raw_word(word):
    raw_word=word.replace('_',' ')
    raw_word=raw_word.replace('-LRB-','(')
    raw_word=raw_word.replace('-RRB-',')')
    raw_word=raw_word.replace('-LCB-','{')
    raw_word=raw_word.replace('-RCB-','}')
    raw_word=raw_word.replace('-LSB-','[')
    raw_word=raw_word.replace('-RSB-',']')
    raw_word=unicodedata.normalize('NFC',raw_word)
    return raw_word

def get_entities_by_spacy(query):
    entities=set()
    doc = nlp(query)
    for entity in doc.ents:
        entities.add(entity.text)
    return entities

def get_entities_by_allen_nlp(query,ner):
    results = ner.predict(sentence=query)
    entities=set()
    i=0
    while i <len(results["words"]):
        word=results['words'][i]
        tag=results['tags'][i]
        new_word=word
        while i+1<len(results["words"]):
            next_word=results['words'][i+1]
            next_tag=results['tags'][i+1]
            if sametag(tag,next_tag) and tag!='O':
                i+=1
                new_word+=' '+next_word
            else:
                break
        if i==len(results["words"])-1:
            if tag!='O' and word not in entities:
                entities.add(word.lower())
        i+=1
        if new_word not in entities and tag!='O':
            entities.add(new_word.lower())
    return entities

def sametag(tag1,tag2):
    if len(tag1)>1 and len(tag2)>1:
        if tag1[1:]==tag2[1:]:
            return True
    return False

def get_relevant_document_entity(query,ner,alias_wiki):
    entities=get_entities_by_allen_nlp(query,ner)
    documents=set()
    for entity in entities:
        if entity in alias_wiki:
            entity_documents=alias_wiki[entity]
            for document in entity_documents:
                if document not in documents:
                    documents.add(document)
    return documents

def get_alias_dictionaries(wiki):
    alias_wiki={}
    for processed_title in wiki:
        raw_title=get_raw_word(processed_title)
        if raw_title in alias_wiki:
            alias_wiki[raw_title].append(processed_title)
        else:
            alias_wiki[raw_title]=[processed_title]
        title_lower=raw_title.lower()
        if title_lower in alias_wiki:
            alias_wiki[title_lower].append(processed_title)
        else:
            alias_wiki[title_lower]=[processed_title]

        if '(' in raw_title and ')' in raw_title:
            position_l=raw_title.index('(')
            title_1=raw_title[0:position_l].rstrip()
            title_1_lower=title_1.lower()
            if title_1 in alias_wiki:
                alias_wiki[title_1].append(processed_title)
            else:
                alias_wiki[title_1]=[processed_title]
            if title_1_lower in alias_wiki:
                alias_wiki[title_1_lower].append(processed_title)
            else:
                alias_wiki[title_1_lower]=[processed_title]  
    return alias_wiki

def get_relevant_document_tf_idf(query,index,wiki_titles):
    lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
    tokens = nltk.tokenize.word_tokenize(query)
    processed_query=[]
    for token in tokens:
        term = lemmatizer.lemmatize(token).lower()
        processed_query.append(term)
    ids=query_tfidf(processed_query,index)
    documents=set()
    for id in ids:
        documents.add(wiki_titles[id[0]])
    return documents

def get_retrieval_documents(claim,wiki,alias_wiki,wiki_titles,index,ner):
    claim=unicodedata.normalize('NFC',claim)
    documents_entity=get_relevant_document_entity(claim,ner,alias_wiki)
    documents_tf_idf=get_relevant_document_tf_idf(claim,index,wiki_titles)
    documents=documents_entity | documents_tf_idf
    return documents

def allen_nlp_ner():
    ner = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/ner-model-2018.12.18.tar.gz")
    return ner

def save_wiki(wiki):
    file=open('wiki.txt','w')
    for entry in wiki:
        data=entry+'\t'+wiki[entry][0]+'\t'+str(wiki[entry][1])
        file.write(data+'\n')
    file.close()

def save_alias_wiki(alias_wiki):
    file=open('alias_wiki.txt','w')
    for entry in alias_wiki:
        data=entry
        for one_alias in alias_wiki[entry]:
            data+='\t'+one_alias
        file.write(data+'\n')
    file.close()

def load_wiki():
    file=open('wiki.txt','r')
    wiki={}
    for line in file:
        tokens=line.split('\t')
        wiki[tokens[0]]=(tokens[1],int(tokens[2]))
    file.close()
    return wiki

def load_alias_wiki():
    file=open('alias_wiki.txt','r')
    alias_wiki={}
    for line in file:
        tokens=line.split('\t')
        alias_wiki[tokens[0]]=[]
        for token in tokens[1:]:
            alias_wiki[tokens[0]].append(token.replace('\n',''))
    file.close()
    return alias_wiki

def devide_file();
    f1=open('nli.txt','r')
    x,y,z=[],[],[]
    for line in f1:
        items=line.split('\t')
        label=items[2].replace('\n','')
        if label=='SUPPORTS':
            x.append(line.replace('\n',''))
        elif label=='REFUTES':
            y.append(line.replace('\n',''))
        else:
            z.append(line.replace('\n',''))
    m=min(len(x),len(y),len(z))
    a=x[1000:m]
    b=y[1000:m]
    c=z[1000:m]
    d=a+b+c
    e=x[:1000]+y[:1000]+z[:1000]
    np.random.shuffle(d)
    np.random.shuffle(e)
    f2=open('nli_train.txt','w')
    for i in d:
        f2.write(i+'\n')
    f2.close()
    f3=open('nli_test.txt','w')
    for i in e:
        f3.write(i+'\n')
    f3.close()



def generate_nli_dataset(wiki,alias_wiki,ner):
    nlp=spacy.load('en_vectors_web_lg')
    parser=spacy.load('en_core_web_sm')
    lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
    file=open('train.json','r')
    training_nli=open('nli.txt','w')
    training_data=json.load(file)
    print('success')
    for i,data in enumerate(training_data):
        claim=training_data[data]['claim']
        claim_vector=nlp(claim).vector
        label=training_data[data]['label']
        correct_documents=set()
        for document in training_data[data]['evidence']:
            correct_documents.add((document[0],document[1]))
        if len(correct_documents)==0:
            documents_entity=get_relevant_document_entity(claim,ner,alias_wiki)
            if len(documents_entity)>0:
                similarities=Counter()
                for title in documents_entity:
                    if title in wiki:
                        for sentence_number in wiki[title]:
                            sentence=replace_all(wiki[title][sentence_number],get_raw_word(title))
                            sentence_vector=nlp(sentence).vector
                            similarity=1 - spatial.distance.cosine(sentence_vector, claim_vector)
                            similarities[(title,sentence_number)]=similarity
                best_sentences=similarities.most_common(1)
                for item in best_sentences:
                    best_sentence_title,best_sentence_number=item[0]
                    supporting_sentence=replace_all(wiki[best_sentence_title][best_sentence_number],get_raw_word(best_sentence_title))  
                    training_nli.write(claim+'\t'+supporting_sentence+'\t'+label+'\n')
        else:
            for document in correct_documents:
                title=document[0]
                number=document[1]
                if title in wiki and number in wiki[title]:
                    supporting_sentence=replace_all(wiki[title][number],get_raw_word(title))
                    training_nli.write(claim+'\t'+supporting_sentence+'\t'+label+'\n')
        if i%100==0:
            print(i)
    training_nli.close()
    file.close()

ner=allen_nlp_ner()
wiki=wiki_title_preprocess()
alias_wiki=get_alias_dictionaries(wiki)
print('wiki data processed')
import sys
print(sys.getsizeof(wiki)/1024/1024)
print(sys.getsizeof(alias_wiki)/1024/1024)
generate_nli_dataset(wiki,alias_wiki,ner)
devide_file()