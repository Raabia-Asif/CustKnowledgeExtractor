#!/usr/bin/env python
# coding=utf-8

import os
from pycorenlp import StanfordCoreNLP
from nltk.parse.stanford import StanfordDependencyParser
import json

# Assumes that the corenlp server is already running
# To run corenlp server, open cmd
# >cd ..
# >D:
# >cd D:\Ph.D Work\tools\NERs\stanford tools\stanford-corenlp-full-2018-02-27
# >java -mx7g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer 9000 100000000000000000000000
nlp = StanfordCoreNLP('http://localhost:9000')

# import nltk
# from nltk.tag import StanfordPOSTagger
# from nltk import word_tokenize, sent_tokenize
# from nltk.tree import Tree
# from nltk.parse.stanford import StanfordParser
# from nltk.tag import StanfordNERTagger

def annotateKbpRelations(doc,reOutFile):
    #Assumes that the corenlp server is already running
    #To run corenlp server, open cmd
    # >cd D:\phd work\tools\NERs\stanford tools\stanford-corenlp-full-2018-02-27
    # >D:
    # >java -mx7g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer 9000 100000000000000000000000
    nlp = StanfordCoreNLP('http://localhost:9000')
    text = doc

    # FOR STANFORD KBP Relations ANNOTATION
    properties = {'annotators': 'tokenize,ssplit,pos,lemma,ner,parse,coref,kbp', #                  'coref.md.type': 'RULE',
                  'outputFormat': 'json',

                  'parse.model': 'edu/stanford/nlp/models/srparser/englishSR.ser.gz'}

    output = nlp.annotate(text, properties)#output is a dictionary
    relations = output['sentences'][0]['kbp']
    with open(reOutFile, 'w',encoding='utf-8') as outfile:
        json.dump(relations, outfile, sort_keys=True, indent=4, ensure_ascii=False)
    print(relations)

def annotateNER(doc,nerOutFile):

    text = doc

    # FOR STANFORD NER ANNOTATION
    # i changed parse in annotators to depparse
    properties = {'annotators': 'tokenize,ssplit,pos,lemma,ner,parse', 'outputFormat': 'conll', #changed outputFormat from json to conll
                  'parse.model': 'edu/stanford/nlp/models/srparser/englishSR.ser.gz'}

    output = nlp.annotate(text, properties)#output is a dictionary
    # print(output['sentences'][0])
    # print(output['sentences'][0]['entitymentions'])
    # output.format('json')
    # data = output
    # with open(nerOutFile, 'w',encoding='utf-8') as outfile:
    #     data = data.replace('\r','')
    #     outfile.write(data)
    #     # json.dump(data, outfile, sort_keys=True, indent=4, ensure_ascii=False)
    # print(output)


def generateREdatasetConllFormat(): #input is sentences, output is annotated data in conll format, I then manually tag SUBJECT OBJECT and RELATION in the generated data
    #Assumes that the corenlp server is already running
    #To run corenlp server, open cmd
    # >cd ..
    # >D:
    # >cd D:\phd work\tools\NERs\stanford tools\stanford-corenlp-full-2018-02-27
    # >java -mx7g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer 9000 100000000000000000000000
    nlp = StanfordCoreNLP('http://localhost:9000')
    inFile = os.path.normpath("inputOKE\okeEvalInputTexts.txt")#inputOurFREdatasetNew2/sentences.txt")
    outFile = os.path.normpath("inputOKE\okeEval.conll")#inputOurFREdatasetNew2/dataset.conll")
    #outFile = os.path.normpath("inputOurFREdataset/dataset2.conll")
    with open(inFile,'r',encoding='utf-8') as inFile:
        text = inFile.read()
    # text = "\"I\'m so glad you come before we began,\" said Nan, cheerfully."
    # FOR STANFORD POS, NER, DEP ANNOTATION
    # remove ner annotator for getting dashes (-) in one column
    properties = {'annotators': 'ssplit,tokenize,pos,parse',
    #properties = {'annotators': 'tokenize,pos,lemma,ner,depparse',
                  #'output.columns': 'idx,word,role,role,pos,ner,deprel,headidx',#this line is not working
                  'outputFormat': 'conll',
                  'parse.model': 'edu/stanford/nlp/models/srparser/englishSR.ser.gz'}

    output = nlp.annotate(text, properties)
    data = output
    with open(outFile, 'w',encoding='utf-8') as outfile:
        data = data.replace('\r','')
        outfile.write(data)
    print(output)


def annotateStanfordCoref(doc):
    #Assumes that the corenlp server is already running
    #To run corenlp server, open cmd
    # >cd ..
    # >D:
    # >cd D:\phd work\tools\NERs\stanford tools\stanford-corenlp-full-2018-02-27
    # >java -mx7g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer 9000 100000000000000000000000
    nlp = StanfordCoreNLP('http://localhost:9000')
    text = doc
    # text = "Tom and Jane are good friends. They are cool. He knows a lot of things and so does she. His car is red, but hers is blue. It is older than hers. The big cat ate its dinner."

    # For coreference annotation
    # my_proxies = {'http': 'wireless.cust','https': 'wireless.cust'}
    properties = {'annotators': 'dcoref', 'outputFormat': 'json'}#,'parse.model': 'edu/stanford/nlp/models/srparser/englishSR.ser.gz'}
    output = nlp.annotate(text, properties)
    # x = output['corefs']
    return output


def areCorefs(subjStartIndex,subjEndIndex, objStartIndex,objEndIndex, sentence):
    print("inside areCorefs of ClientNER...........")
    corenlp_output = annotateStanfordCoref(sentence)
    for coref in corenlp_output['corefs']:
        subjInCorefs = 0
        objInCorefs = 0
        bothInCorefs = 0
        mentions = corenlp_output['corefs'][coref] #one set of co-referents. Its a list. Check if both subject and object are in this list.
        if len(mentions)>=2:
            #json files added for both.. process subj_start from json for both datasets now
            # check if both subject and object are in the list of co-referents
            for mention in mentions:
                startIndex = mention['startIndex']-1 #because indices of coref start from 1, while indices of TACRED examples start from 0
                endIndex = mention['endIndex']-2 #because indices of coref start from 1, and end at 1 more than the end indices of TACRED examples
                if subjStartIndex == startIndex and subjEndIndex==endIndex:
                    subjInCorefs = 1
                if objStartIndex == startIndex and objEndIndex==endIndex:
                    objInCorefs = 1
                if subjInCorefs==1 and objInCorefs==1:
                    bothInCorefs = 1
                    break
        if bothInCorefs == 1:
            break
    return bothInCorefs


def getDotDepParse(text):
    # make sure nltk can find stanford-parser
    path_to_jar = 'D:/phd work/tools/NERs/stanford tools/stanford-parser-full-2018-02-27/stanford-parser.jar'
    path_to_models_jar = 'D:/phd work/tools/NERs/stanford tools/stanford-parser-full-2018-02-27//stanford-parser-3.9.1-models.jar'

    dep_parser = StanfordDependencyParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)

    p = dep_parser.raw_parse(text)
    for e in p:
        p = e
        break
    print(p.to_dot())
    # Copy paste the output to http://graphs.grevian.org/graph and press the Generate button. You should see the desired graph.