# this file use FEVER dataset to fine-tune the learned DA model
import spacy
import numpy as np
import ujson as json
import keras
from keras.utils import to_categorical
from keras import layers, Model, models
from keras import backend as K

def read_fever(path):
    texts1=[]
    texts2=[]
    labels=[]
    file=open(path,'r')
    LABELS = {'SUPPORTS': 0, 'REFUTES': 1, 'NOT ENOUGH INFO': 2}
    for line in file:
        items=line.split('\t')
        texts1.append(items[1]) # supporting sentences
        texts2.append(items[0]) # hypothesis
        label=items[2].replace('\n','')
        labels.append(LABELS[label])
    return texts1, texts2, to_categorical(np.asarray(labels, dtype='int32'))

def create_dataset(nlp, texts, hypotheses, num_oov, max_length, norm_vectors = True):
    # some code in this function reuse code from https://github.com/explosion/spaCy/blob/master/examples/notebooks/Decompositional%20Attention.ipynb
    # which is an implementation of DA model
    sents = texts + hypotheses
    num_vectors = max(lex.rank for lex in nlp.vocab) + 2 
    oov = np.random.normal(size=(num_oov, nlp.vocab.vectors_length))
    oov = oov / oov.sum(axis=1, keepdims=True)
    vectors = np.zeros((num_vectors + num_oov, nlp.vocab.vectors_length), dtype='float32')
    vectors[num_vectors:, ] = oov
    for lex in nlp.vocab:
        if lex.has_vector and lex.vector_norm > 0:
            vectors[lex.rank + 1] = lex.vector / lex.vector_norm if norm_vectors == True else lex.vector
    sents_as_ids = []
    for sent in sents:
        doc = nlp(sent)
        word_ids = []
        for i, token in enumerate(doc):
            if token.has_vector and token.vector_norm == 0:
                continue
            if i > max_length:
                break
            if token.has_vector:
                word_ids.append(token.rank + 1)
            else:
                word_ids.append(token.rank % num_oov + num_vectors) 
        word_id_vec = np.zeros((max_length), dtype='int')
        clipped_len = min(max_length, len(word_ids))
        word_id_vec[:clipped_len] = word_ids[:clipped_len]
        sents_as_ids.append(word_id_vec)
    return vectors, np.array(sents_as_ids[:len(texts)]), np.array(sents_as_ids[len(texts):])

def create_embedding(vectors, max_length, projected_dim):
    # some code in this function reuse code from https://github.com/explosion/spaCy/blob/master/examples/notebooks/Decompositional%20Attention.ipynb
    # which is an implementation of DA model
    return models.Sequential([
        layers.Embedding(
            vectors.shape[0],
            vectors.shape[1],
            input_length=max_length,
            weights=[vectors],
            trainable=False),
        
        layers.TimeDistributed(
            layers.Dense(projected_dim,
                         activation=None,
                         use_bias=False))
    ])

def create_feedforward(num_units=200, activation='relu', dropout_rate=0.2):
    # some code in this function reuse code from https://github.com/explosion/spaCy/blob/master/examples/notebooks/Decompositional%20Attention.ipynb
    # which is an implementation of DA model
    return models.Sequential([
        layers.Dense(num_units, activation=activation),
        layers.Dropout(dropout_rate),
        layers.Dense(num_units, activation=activation),
        layers.Dropout(dropout_rate)
    ])

def normalizer(axis):
    # some code in this function reuse code from https://github.com/explosion/spaCy/blob/master/examples/notebooks/Decompositional%20Attention.ipynb
    # which is an implementation of DA model
    def _normalize(att_weights):
        exp_weights = K.exp(att_weights)
        sum_weights = K.sum(exp_weights, axis=axis, keepdims=True)
        return exp_weights/sum_weights
    return _normalize

def sum_word(x):
    # some code in this function reuse code from https://github.com/explosion/spaCy/blob/master/examples/notebooks/Decompositional%20Attention.ipynb
    # which is an implementation of DA model
    return K.sum(x, axis=1)

def build_model(vectors, max_length, num_hidden, num_classes, projected_dim, entail_dir='both'):
    # some code in this function reuse code from https://github.com/explosion/spaCy/blob/master/examples/notebooks/Decompositional%20Attention.ipynb
    # which is an implementation of DA model
    input1 = layers.Input(shape=(max_length,), dtype='int32', name='words1')
    input2 = layers.Input(shape=(max_length,), dtype='int32', name='words2')
    
    # embeddings (projected)
    embed = create_embedding(vectors, max_length, projected_dim)
   
    a = embed(input1)
    b = embed(input2)
    
    # step 1: attend
    F = create_feedforward(num_hidden)
    att_weights = layers.dot([F(a), F(b)], axes=-1)
    
    G = create_feedforward(num_hidden)
    
    if entail_dir == 'both':
        norm_weights_a = layers.Lambda(normalizer(1))(att_weights)
        norm_weights_b = layers.Lambda(normalizer(2))(att_weights)
        alpha = layers.dot([norm_weights_a, a], axes=1)
        beta  = layers.dot([norm_weights_b, b], axes=1)

        # step 2: compare
        comp1 = layers.concatenate([a, beta])
        comp2 = layers.concatenate([b, alpha])
        v1 = layers.TimeDistributed(G)(comp1)
        v2 = layers.TimeDistributed(G)(comp2)

        # step 3: aggregate
        v1_sum = layers.Lambda(sum_word)(v1)
        v2_sum = layers.Lambda(sum_word)(v2)
        concat = layers.concatenate([v1_sum, v2_sum])
    elif entail_dir == 'left':
        norm_weights_a = layers.Lambda(normalizer(1))(att_weights)
        alpha = layers.dot([norm_weights_a, a], axes=1)
        comp2 = layers.concatenate([b, alpha])
        v2 = layers.TimeDistributed(G)(comp2)
        v2_sum = layers.Lambda(sum_word)(v2)
        concat = v2_sum
    else:
        norm_weights_b = layers.Lambda(normalizer(2))(att_weights)
        beta  = layers.dot([norm_weights_b, b], axes=1)
        comp1 = layers.concatenate([a, beta])
        v1 = layers.TimeDistributed(G)(comp1)
        v1_sum = layers.Lambda(sum_word)(v1)
        concat = v1_sum
    
    H = create_feedforward(num_hidden)
    out = H(concat)
    out = layers.Dense(num_classes, activation='softmax')(out)
    
    model = Model([input1, input2], out)
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def start():
    # some code in this function reuse code from https://github.com/explosion/spaCy/blob/master/examples/notebooks/Decompositional%20Attention.ipynb
    # which is an implementation of DA model
    nlp = spacy.load('en_vectors_web_lg')
    texts,hypotheses,labels = read_fever('nli_train.txt')
    sem_vectors, text_vectors, hypothesis_vectors = create_dataset(nlp, texts, hypotheses, 100, 50, True)
    texts_test,hypotheses_test,labels_test = read_fever('nli_test.txt')
    _, text_vectors_test, hypothesis_vectors_test = create_dataset(nlp, texts_test, hypotheses_test, 100, 50, True)
    K.clear_session()
    m = keras.models.load_model('da.h5')
    m.summary()
    m.fit([text_vectors, hypothesis_vectors], labels, batch_size=1024, epochs=30,validation_data=([text_vectors_test, hypothesis_vectors_test], labels_test))
    return m
m=start()
m.save('da_fever.h5')