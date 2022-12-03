# from random import randint
# from scipy import sparse
# from multiprocessing import Pool
# from collections import Counter
import string
# import requests,time
import numpy
# import re
# from math import log
# import logging
# import timeout_decorator
# from bs4 import BeautifulSoup,SoupStrainer
# import pickle
# from underthesea import text_normalize
from synonyms import generateVariants
# from synonyms import generateVariantKeywords
from underthesea import pos_tag
# from entity_linking import extractEntVariants
from gensim.models import KeyedVectors
from underthesea import word_tokenize
# import json

w2v = KeyedVectors.load_word2vec_format('resources/wiki.vi.model.bin', binary=True)

stopwords = open('resources/stopwords_small_origin.txt', encoding='utf-8').read().split('\n')
stopwords = set([w.strip().replace(' ','_') for w in stopwords])

stopwords_extraction = open('resources/stopwords_small.txt', encoding='utf-8').read().split('\n')
stopwords_extraction = set([w.strip() for w in stopwords_extraction])

punct_set = set([c for c in string.punctuation]) | set(['“','”',"...","-","…","..","•",'“', '\"', '?', '!', ''])


overlap_threshold = 0.8
isRelevant_threshold = 0.65


def cos_sim(a, b):
    return numpy.dot(a, b) / (numpy.linalg.norm(a) * numpy.linalg.norm(b))

def document_vector(doc):
    vec = [w2v.wv[i] for i in doc]
    return numpy.sum(vec,axis = 0)

def embedding_similarity(s1,s2):
    s1 = sorted(list(set(s1.lower().split())))
    s2 = sorted(list(set(s2.lower().split())))
    
    s1 = [word for word in s1 if word in w2v.wv.vocab]
    s2 = [word for word in s2 if word in w2v.wv.vocab]
    
    if len(s1) == 0 or len(s2) == 0:
         return 0
    
    return cos_sim(document_vector(s1),document_vector(s2)) 


def generateNgram(paper, ngram = 2, deli = '_', rmSet = {}):
    words = paper.split()
    if len(words) == 1:
        return ''
    
    ngrams = []
    for i in range(0,len(words) - ngram + 1):
        block = words[i:i + ngram]
        if not any(w in rmSet for w in block):
            ngrams.append(deli.join(block))

    # ngrams = [deli.join(words[i:i + ngram]) for i in range(0, len(words) - ngram + 1) if not any(w in rmSet for w in words[i:i + ngram])]
    
    return ngrams


def generatePassages(document,n):
    passages = []
    paragraphs = document[1].split('\n\n')
    for para in paragraphs:
        sentences = para.rsplit(' . ')
        
        if len(sentences) <= 3:
            passages.append((document[0], ' '.join(sentences)))
        else:
            for i in range(0,len(sentences) - n + 1):
                passages.append((document[0], ' '.join([sentences[i + j] for j in range(0,n) if '?' not in sentences[i + j]])))

    return passages


def passage_score(q_ngrams,passage):
    try:
        passage = passage.lower()

        p_unigram = set(generateNgram(passage,1,'_',punct_set | stopwords))
        
        uni_score = len(p_unigram & q_ngrams['unigram'])
        # uni_score = 0
        p_bigram  = set(generateNgram(passage,2,'_',punct_set | stopwords))
        p_trigram = set(generateNgram(passage,3,'_',punct_set | stopwords))
        p_fourgram= set(generateNgram(passage,4,'_',punct_set))

        bi_score = len(p_bigram & q_ngrams['bigram'])
        tri_score = len(p_trigram & q_ngrams['trigram'])
        four_score = len(p_fourgram & q_ngrams['fourgram'])

        #emd_sim = embedding_similarity(' '.join(p_unigram),' '.join(q_ngrams['unigram']))
        emd_sim = 0

        return uni_score + bi_score*2 + tri_score*3 + four_score*4 + emd_sim*3
    except:
        return 0

def passage_score_wrap(args):
    return passage_score(args[0],args[1])

# def chunks(l, n):
#     for i in range(0, len(l), n):
#         yield l[i:i + n]


# def get_entities(seq):
#     i = 0
#     chunks = []
#     seq = seq + ['O']  # add sentinel
#     types = [tag.split('-')[-1] for tag in seq]
#     while i < len(seq):
#         if seq[i].startswith('B'):
#             for j in range(i+1, len(seq)):
#                 if seq[j].startswith('I') and types[j] == types[i]:
#                     continue
#                 break
#             chunks.append((types[i], i, j))
#             i = j
#         else:
#             i += 1
#     return chunks


# def get_ner(text):
#     res = ner(text)
#     words = [r[0] for r in res]
#     tags = [t[3] for t in res]
    
#     chunks = get_entities(tags)
#     res = []
#     for chunk_type, chunk_start, chunk_end in chunks:
#         res.append(' '.join(words[chunk_start: chunk_end]))
#     return res


def keyword_extraction(question):
    # question = question.replace('_',' ').replace('\"', '').replace('"', '').replace('?', '').replace('!', '').replace(',', '').replace('“', '').replace('”', '').replace('-', '')
    # question = ' '.join([w for w in question.split(' ') if w not in stopwords_extraction])

    ### Tokenize question
    # regex = '\d*\s*tháng\s*\d*'
    # regex = 'năm \d*'
    # regex = 
    # question = text_normalize(question)
    
    tokens = []
    rule_based = question.split(' ')
    if 'sinh' in rule_based:
       del rule_based[rule_based.index('sinh')]  
       tokens.append('sinh')
       question = ' '.join(rule_based)
        
    if 'khởi' in rule_based and 'quay' in rule_based:
       del rule_based[rule_based.index('khởi')]  
       del rule_based[rule_based.index('quay')]  
       tokens.append('khởi quay')
       question = ' '.join(rule_based)
        
    if '“' in rule_based and '”' in rule_based:
        start = int(rule_based.index('“'))
        end = int(rule_based.index('”'))
        tokens.append(' '.join(rule_based[start+1:end]))
        del rule_based[start:end+1]
        question = ' '.join(rule_based)
    
    # if rule_based.count('\"') == 2:
    #     start = int(rule_based.index('\"'))
    #     del rule_based[start]
    #     end = int(rule_based.index('\"'))
    #     del rule_based[end]
    #     tokens.append(' '.join(rule_based[start:end]))
    #     del rule_based[start:end]
    #     question = ' '.join(rule_based)
    
    # if rule_based.count('"') == 2:
    #     start = int(rule_based.index('"'))
    #     del rule_based[start]
    #     end = int(rule_based.index('"'))
    #     del rule_based[end]
    #     tokens.append(' '.join(rule_based[start:end]))
    #     del rule_based[start:end]
    #     question = ' '.join(rule_based)
    
    tokens.extend([word.lower() for word in word_tokenize(question) if word.lower() not in stopwords_extraction and word not in punct_set])
    tokens = [token for token in tokens if token != '']
    tokens = list(set(tokens))
    tokens = sorted(tokens, reverse=True)
    # tokens = list(dict.fromkeys(tokens))
    
    # keywords = []    
    # for token in tokens:
    #     variants = extractEntVariants(token)
    #     keywords.append(variants)
    
    keywords = [[token] for token in tokens]
    
    print(f'Value in keywords: {keywords}')
    print(f'Value in keywords before augmenting: {tokens}')

    
    pos_tokens = [word[0].lower() for word in pos_tag(question) if word[0].lower() not in stopwords_extraction and word[0] not in punct_set and word[1] in ['N', 'Np'] and word[0].istitle()]
    
    return keywords, pos_tokens


def isRelevantDocument(text,keywords):
    text = text.lower()
    
    for words in keywords:
        if not any(e for e in words if e in text):
            return False

    return True


def isRelevantDocumentChau(text,keywords):
    text = text.lower().replace('_',' ')

    n = len(keywords)
    check = [True]*n
    for i, words in enumerate(keywords):
        if not any(e for e in words if e in text):
            check[i] = False
            
    return (sum(check)/n >= isRelevant_threshold)


def removeDuplicate(documents):
    mapUnigram  = {}
    for doc in documents:
        mapUnigram[doc[1]] = generateNgram(doc[1].lower(),1,'_',punct_set | stopwords)

    uniqueDocs = []
    for i in range(0,len(documents)):
        check = True
        for j in range(0,len(uniqueDocs)):
            check_doc  = mapUnigram[documents[i][1]]
            exists_doc = mapUnigram[uniqueDocs[j][1]]
            overlap_score = len( set(check_doc) & set(exists_doc) )
            if overlap_score >= overlap_threshold * len(set(check_doc)) or overlap_score >= overlap_threshold * len(set(exists_doc)):
                check = False
        if check:
            uniqueDocs.append(documents[i])
    
    return uniqueDocs


# def split_passage(passage, n = 3):
#     n = len(passage)
#     slice1 = int(n/3)
#     slice2 = int(2*n/3)
#     slice3 = n
    
#     return [passage[:slice1], passage[slice1:slice2], passage[slice2:slice3]]


def rel_ranking(question,documents_raw):
    #Return ranked list of passages from list of documents
    # pool = Pool(4)
    # pool = []

    q_variants = generateVariants(question)
    q_keywords, pos_tokens = keyword_extraction(question)

    q_ngrams = {'unigram': set(generateNgram(question.lower(),1,'_',punct_set | stopwords))
                , 'bigram' : set([]), 'trigram': set([]), 'fourgram': set([])}

    for q in q_variants:
        q = q.lower()
        q_ngrams['bigram']  = q_ngrams['bigram']   | set(generateNgram(q,2,'_',punct_set | stopwords))
        q_ngrams['trigram'] = q_ngrams['trigram']  | set(generateNgram(q,3,'_',punct_set | stopwords))
        q_ngrams['fourgram']= q_ngrams['fourgram'] | set(generateNgram(q,4,'_',punct_set))
    
    documents = [d for d in documents_raw if isRelevantDocument(d[1],q_keywords)]

    print(len(documents))  
    if len(documents) >= 10:
        passages = [generatePassages(d,3) for d in documents]
        passages = [(j[0], j[1].replace('BULLET::::', '')) for i in passages for j in i]
        passages = list(set(passages))
        passages = sorted(passages, reverse=True)
        passages = [p for p in passages if isRelevantDocument(p[1],q_keywords)]
    else:
        documents = [d for d in documents_raw if isRelevantDocumentChau(d[1],q_keywords)]

        passages = [generatePassages(d,3) for d in documents]
        passages = [(j[0], j[1].replace('BULLET::::', '')) for i in passages for j in i]
        passages = list(set(passages))
        passages = sorted(passages, reverse=True)
        passages = [p for p in passages if isRelevantDocumentChau(p[1],q_keywords)]
        
    documents_titles = [d for d in documents_raw if d[0].lower() in pos_tokens]
    passages_titles = []
    if len(documents_titles) > 0:
        passages_titles = [generatePassages(d,3) for d in documents_titles]
        passages_titles = [(j[0], j[1].replace('BULLET::::', '')) for i in passages_titles for j in i]
        passages_titles = list(set(passages_titles))
        passages = sorted(passages, reverse=True)
        passages_titles = [p for p in passages_titles if isRelevantDocumentChau(p[1],[[w] for w in pos_tokens])]
    
    p_scores = map(passage_score_wrap,[(q_ngrams,p[1]) for p in passages])
    p_res = numpy.argsort([-s for s in p_scores])

    # relevantDocs = [p for p in passages_titles]

    # n_passages = len(passages) if len(passages) < 100 else 100
    relevantDocs = [passages[p_res[i]] for i in range(0, len(passages))] 
    if len(documents) <= 500:
        relevantDocs = removeDuplicate(relevantDocs)
    
    if len(passages_titles) > 0:
        for i in range(0, len(passages_titles)):
            relevantDocs.insert(i, passages_titles[i])
    # pool.terminate()
    # del pool
        
    return relevantDocs, len(passages_titles)