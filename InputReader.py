from datetime import datetime
#!/usr/bin/env python
# coding=utf-8
from ClientNER import *
# from RuleEngineRE import *
# # from neuralNet import *
from ClientSpotlight import annotateSpotlight
from ClientIllinois import annotateIllinois
from RuleEngine import applyRules
# from RuleEngineGender import applyRulesGender
# # from CorefResolution import *
# from scoreREsentenceLenWise import scoreREsenWise
# from nltk.tokenize.treebank import TreebankWordDetokenizer
# from nltk.corpus import wordnet as wn
# from nltk.corpus import stopwords
# # from nltk.tokenize import word_tokenize
# # from mongoDBclient import connectToDB
# # from nltk.parse.corenlp import CoreNLPDependencyParser
# # from graphviz import Source
import io, os
# import os.path
# import json
# import re
# import spacy
# from trainNER import *
# from excelReader import *
# from Evaluator import *

# #for text recognition from image
# try:
#     from PIL import Image
# except ImportError:
#     import Image
# import pytesseract

def cleanStanNer():
    inPath = "outputConll\eval\\ner\outputNER"
    outPath = "outputConll\eval\\ner\cleaned\outputNER"
    i=0
    while i<=230:
        nerInFile = inPath + str(i) + '.txt'
        nerOutFile = outPath + str(i) + '.txt'
        allNerEntities = []
        with open(nerInFile, 'r', encoding='utf-8', errors='ignore') as f:
            nerDict = json.load(f)
        nerSentences = nerDict['sentences']  # list of dicts
        for sentence in nerSentences:
            nerEntities = sentence['entitymentions']  # list of dicts, contains entities identified by NER in one sentence
            allNerEntities += nerEntities
        allNerEntities = sorted(allNerEntities, key=lambda k: k['characterOffsetBegin'])  # sorting nerEntities on offset
        with open(nerOutFile, 'w', encoding='utf-8') as f2:
            json.dump(allNerEntities,f2 , sort_keys=True, indent=4, ensure_ascii=False)
        i+=1

def okeAnnotateNerSpot():#this function is for reading OKE file and writing annotated texts in ner/spot/illinois out files
    # inputFile = 'inputOKE\okeTrainInputTexts.txt'#for TRAINING dataset
    inputFile = 'inputOKE\okeEvalInputTexts.txt' #for EVALUATION dataset
    f = io.open(inputFile,'r',encoding='utf-8')
    i=-1
    for line in f: #read a file one line at a time using for loop, one line represents one document and can have multiple sentences
        # line = line.encode('utf-8')
        # print(line.strip())

        i += 1
        if i<-1:
            continue
        print(i)
        #-------For TRAINING dataset------------
        # illinoisOutFile = 'outputOKE\\train\illinois\\tokenWise\outputIllinois'+str(i)+'.txt' #for training dataset
        # spotOutFile = 'outputOKE\\train\spotlight\outputSpotlight'+str(i)+'.txt' #for UNTYPED #for training dataset
        # spotOutFile = 'outputOKE\\train\spotlight\\typed\outputSpotlight'+str(i)+'.txt' #for TYPED #for training dataset
        # nerOutFile = 'outputOKE\\train\\ner\outputNER'+str(i)+'.txt' #for training dataset
        # nerOutFile = 'outputOKE\\train\\ner\inConllFormat\outputNER' + str(i) + '.txt'  # for train dataset

        # # -------For EVALUATION dataset------------
        illinoisOutFile = 'outputOKE\eval\illinois\miscIncluded\outputIllinois' + str(i) + '.txt'  # for eval dataset
        spotOutFile = 'outputOKE\eval\spotlight\outputSpotlight'+str(i)+'.txt' #for UNTYPED #for training dataset
        # spotOutFile = 'outputOKE\eval\spotlight\\typed\outputSpotlight' + str(i) + '.txt'  # for TYPED #for training dataset
        nerOutFile = 'outputOKE\eval\\ner\inConllFormat\outputNER'+str(i)+'.txt' #for eval dataset

        annotateIllinois(line,illinoisOutFile)
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Current Time After illinois annotation =", current_time)

        # annotateSpotlight(line,spotOutFile)
        annotateNER(line,nerOutFile)
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Current Time After stanford annotation =", current_time)

        # if i>=0: break
    f.close()

def conllConvert(): #this function reads original CoNLL file and converts it to sentences. This is just one time task, no need to do again
    # inputFile = 'inputConll\conll2003.eng.train.txt' #TRAINING DATASET
    # inputFile2 = 'inputConll\conll2003.eng.train.2.xml.ttl'#this has conll input as sentences # TRAINING DATASET
    inputFile = 'inputConll\conll2003.eng.testb.txt' #EVALUATION DATASET
    inputFile2 = 'inputConll\conll2003.eng.testb.sentences.txt'#this has conll input as sentences  #EVALUATION DATASET
    f = io.open(inputFile,'r',encoding='utf-8')
    for line in f: #read a file one line at a time using for loop
        x=line.split(' ')
        with open(inputFile2, 'a', encoding='utf-8') as infile:
            if x[0]=='\n':
                infile.write(x[0])
            else:
                infile.write(x[0]+' ')
    f.close()
    f2 = io.open(inputFile2, 'a', encoding='utf-8')
    f2.write('-DOCEND-')
    f2.close()

def conllSample(): #this function reads original CoNLL file and creates a sample from it. This is just one time task, no need to do again
    # inputFile = "inputConll\conll2003.eng.train.txt" #TRAINING DATASET
    # outputFile = "inputConll\conll2003.eng.train.sample.txt" # TRAINING DATASET Sample
    # inputFile = 'inputConll\conll2003.eng.testa.txt' #Dev DATASET
    # outputFile = 'inputConll\conll2003.eng.testa.sample.txt'#DEV DATASET Sample
    inputFile = 'inputConll\conll2003.eng.testb.txt'  # EVALUATION DATASET
    outputFile = 'inputConll\conll2003.eng.testb.sample.txt'  # EVALUATION DATASET Sample
    f = io.open(inputFile,'r',encoding='utf-8')
    outfile = io.open(outputFile, 'a', encoding='utf-8')
    outfile.truncate(0)
    outfile.write('-DOCSTART- -X- O O\n')
    docCounter = 0

    for line in f: #read a file one line at a time using for loop
        x = line.split(' ')
        if docCounter == 20:
            outfile.write(line)
            if x[0] == '-DOCSTART-':
                docCounter = 0

        if x[0] == '-DOCSTART-':
            docCounter+=1

    outfile.close()
    f.close()
    print("\nInput file "+inputFile)
    print("\nCoNLL Sample written to file "+outputFile)

def countNEs(): # this function counts and prints the number of each type of NE in input CoNLL file
    # inputFile = "inputConll\conll2003.eng.train.sample.txt" # TRAINING DATASET Sample
    # inputFile = 'inputConll\conll2003.eng.testa.sample.txt'#DEV DATASET Sample
    inputFile = 'inputConll\conll2003.eng.testb.sample.txt'  # EVALUATION DATASET Sample
    f = io.open(inputFile, 'r', encoding='utf-8')
    perCounter = 0
    locCounter = 0
    orgCounter = 0
    miscCounter = 0
    prev = 'O'

    for line in f:  # read a file one line at a time using for loop
        line = line.strip()
        x = line.split(' ')

        if line!='':
            if x[3]!=prev and x[3]!='O':
                if str.endswith(x[3],'PER'):
                    perCounter += 1
                elif str.endswith(x[3],'LOC'):
                    locCounter += 1
                elif str.endswith(x[3],'ORG'):
                    orgCounter += 1
                elif str.endswith(x[3],'MISC'):
                    miscCounter += 1
            prev = x[3]

    f.close()
    print("\nInput file " + inputFile)
    print("\nNo of PER NEs in dataset is " + str(perCounter))
    print("\nNo of LOC NEs in dataset is " + str(locCounter))
    print("\nNo of ORG NEs in dataset is " + str(orgCounter))
    print("\nNo of MISC NEs in dataset is " + str(miscCounter))

def tacredAnnotateStanfordRE(dataset, corrected): #this function reads TACRED dataset's inputTexts.txt file and writes annotated texts in outputTACRED/test/familyAll/stanford out files
    if corrected==True:
        folder = 'familyAllCorrected'
    else:
        folder = 'familyAll'
    inputFile = os.path.normpath('inputTACRED/' + folder + '/' + dataset + 'InputTexts.txt')
    folder = 'outputTACRED/' + dataset +'/'+ folder
    if dataset=='ours':
        inputFile = os.path.normpath('inputOurFREdataset/InputTexts.txt')
        folder = 'outputOurFREdataset'
    elif dataset=='cust':
        inputFile = os.path.normpath('inputCustFREdataset/InputTexts.txt')
        folder = 'outputCustFREdataset'

    f = io.open(inputFile, 'r', encoding='utf-8')
    i=-1

    for line in f:  # read a file one line at a time using for loop
        i+=1
        # if i<=440:continue
        print('ExampleNo: '+str(i)+' '+line)
        stanfordREOutFile = os.path.normpath(folder+'/stanford/outputRE' + str(i) + '.json')
        annotateKbpRelations(line, stanfordREOutFile)
    f.close()

def conllAnnotateNerSpot(): #this function reads converted CoNLL file and writes annotated texts in ner/spot/illinois out files
    # conllConvert() #this is a one time task, no need to do again
    # inputFile2 = 'inputConll\conll2003.eng.train.2.xml.ttl'  # TRAINING DATASET -- this has conll input as sentences
    inputFile2 = 'inputConll\conll2003.eng.testb.sentences.txt'  #EVALUATION DATASET # this has conll input as sentences
    text = ''
    f2 = io.open(inputFile2, 'r', encoding='utf-8')
    i=-1
    for line in f2:  # read a file one line at a time using for loop
        if (line.find('-DOCSTART-')==-1 and line.find('-DOCEND-')==-1):
            text = text + line
        elif (line.find('-DOCSTART-')!=-1 or line.find('-DOCEND-')!=-1) and text!='':
            i += 1
            print("start of next doc: "+str(i))
            if i>-1:
                print(i)
                # print(text)
                #------- for EVALUATION DATASET ---------
                spotOutFile = 'outputConll\eval\spotlight\outputSpotlight' + str(i) + '.txt'
                nerOutFile = 'outputConll\eval\\ner\\asItIs\outputNER' + str(i) + '.txt'
                illinoisOutFile = 'outputConll\eval\illinois\miscIncluded\outputIllinois' + str(i) + '.txt'
                # ------- for TRAINING DATASET ---------
                # spotOutFile = 'outputConll\\train\spotlight\outputSpotlight' + str(i) + '.txt'
                # nerOutFile = 'outputConll\\train\\ner\outputNER' + str(i) + '.txt'
                # illinoisOutFile = 'outputConll\\train\illinois\outputIllinois' + str(i) + '.txt'

                # annotateSpotlight(text,spotOutFile)
                annotateNER(text,nerOutFile)

                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                print("Current Time After Stanford annotation =", current_time)

                annotateIllinois(text,illinoisOutFile)

                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                print("Current Time After illinois annotation =", current_time)
            # if i>=149: break
            text=''
    f2.close()

def conllCheckNerSpotFiles():#this function checks if spot and ner files for conll dataset are written correctly
    for i in range(0,946):#for TRAIN: 946, for EVAL: 231
        # ------- for TRAINING DATASET ---------
        # spotOutFile = 'outputConll\\train\spotlight\outputSpotlight' + str(i) + '.txt'
        # nerOutFile = 'outputConll\\train\\ner\outputNER' + str(i) + '.txt'
        # ------- for EVALUATION DATASET ---------
        spotOutFile = 'outputConll\eval\spotlight\outputSpotlight' + str(i) + '.txt'
        nerOutFile = 'outputConll\eval\\ner\outputNER' + str(i) + '.txt'
        f1 = io.open(spotOutFile, 'r', encoding='utf-8')
        f2 = io.open(nerOutFile, 'r', encoding='utf-8')
        for line in f1:
            if (line.find('{') == -1):
                print("Spot File not correct: ")
                print(i)
            break
        for line in f2:
            if (line.find('{') == -1):
                print("NER File not correct: ")
                print(i)
            break
        f1.close()
        f2.close()

def okeAnnotateHmari(trainOrEval,flag):
    #this function reads annotated texts from OKE's ner/spot files, applies rules, and writes hmariEntities in output files
    # trainOrEval: 1=train, 2=eval
    i=-1
    if trainOrEval==1:
        inputFile = 'inputOKE\okeTrainInputTexts.txt'  #for TRAINING dataset
    elif trainOrEval ==2:
        inputFile = 'inputOKE\okeEvalInputTexts.txt'  #for EVALUATION dataset

    f = io.open(inputFile, 'r', encoding='utf-8')
    for line in f: #read a file one line at a time using for loop, one line represents one document and can have multiple sentences
        i+=1
        # if i<3: continue
        print("DOCUMENT NO: "+str(i))
        print(line)

        if trainOrEval == 1:
            # -------------FOR TRAINING DATASET-----------------
            nerOutFile = 'outputOKE\\train\\ner\outputNER' + str(i) + '.txt'
            illinoisOutFile = 'outputOKE\\train\\illinois\outputIllinois' + str(i) + '.txt'
            spotOutFile = 'outputOKE\\train\spotlight\outputSpotlight'+str(i)+'.txt' #for UNTYPED.
            entitiesOutFile = 'outputOKE\\train\hmariEntities\output'+str(i)+'.txt' #for UNTYPED

        elif trainOrEval == 2:
            # -------------FOR EVALUATION DATASET-----------------
            nerOutFile = 'outputOKE\eval\\ner\\asItIs\outputNER' + str(i) + '.txt'
            illinoisOutFile = 'outputOKE\eval\\illinois\miscIncluded\outputIllinois' + str(i) + '.txt'
            spotOutFile = 'outputOKE\eval\spotlight\outputSpotlight'+str(i)+'.txt' #for UNTYPED.
            entitiesOutFile = 'outputOKE\eval\hmariEntities\\allEntities\output'+str(i)+'.txt' #for UNTYPED.
        print()
        print()

        applyRules(line, spotOutFile, nerOutFile, illinoisOutFile, entitiesOutFile, flag)
        # if i>=24: break
    f.close()

def conllAnnotateHmari(trainOrEval,flag=1): #this function reads annotated texts from CoNLL's ner/spot/illinois files, applies rules, and writes hmariEntities in output files
    # trainOrEval: 1=train, 2=eval
    # flag is 1 normally, 0 for conll exp4 i.e. highest preference to illinois
    if trainOrEval==1: inputFile = 'inputConll\conll2003.eng.train.2.xml.ttl'  # TRAINING DATASET -- this has conll input as sentences
    elif trainOrEval==2: inputFile = 'inputConll\conll2003.eng.testb.sentences.txt'  #EVALUATION DATASET # this has conll input as sentences
    text = ''
    f = io.open(inputFile, 'r', encoding='utf-8')
    i=-1
    # textLine = 0
    for line in f:  # read a file one line at a time using for loop
        if (line.find('-DOCSTART-')==-1 and line.find('-DOCEND-')==-1):
            # textLine += 1
            # if textLine == 1 or textLine==2:
            #     if (str.isupper(line)):
            #         line = str.lower(line)
            text = text + line
        elif (line.find('-DOCSTART-')!=-1 or line.find('-DOCEND-')!=-1) and text!='':
            # textLine = 0
            i += 1
            print("DOCUMENT NO: "+str(i))
            if i>=0: #last is 945-----
                print(i)
                # print(text)
                if trainOrEval == 1: # ------- for TRAINING DATASET ---------
                    spotOutFile = 'outputConll\\train\spotlight\outputSpotlight' + str(i) + '.txt'
                    nerOutFile = 'outputConll\\train\\ner\outputNER' + str(i) + '.txt'
                    illinoisOutFile = 'outputConll\\train\illinois\outputIllinois' + str(i) + '.txt'
                    if flag==1:
                        entitiesOutFile = 'outputConll\\train\hmariEntities\output' + str(i) + '.txt'
                    elif flag==0:
                        entitiesOutFile = 'outputConll\\train\hmariEntities\exp4\\r7\output' + str(i) + '.txt'
                        # entitiesOutFile = 'outputConll\\train\hmariEntities\exp4\output' + str(i) + '.txt'
                elif trainOrEval == 2:  # ------- for EVALUATION DATASET ---------
                    spotOutFile = 'outputConll\eval\spotlight\outputSpotlight' + str(i) + '.txt'
                    nerOutFile = 'outputConll\eval\\stanford\\asItIs\outputNER' + str(i) + '.txt'
                    illinoisOutFile = 'outputConll\eval\illinois\miscIncluded\outputIllinois' + str(i) + '.txt'
                    if flag==1:
                        entitiesOutFile = 'outputConll\eval\hmariEntities\output' + str(i) + '.txt'
                    elif flag==0:
                        entitiesOutFile = 'outputConll\eval\hmariEntities\exp4\output' + str(i) + '.txt'
                applyRules(text, spotOutFile, nerOutFile,illinoisOutFile, entitiesOutFile,flag)#last argument=flag is 1 normally, 0 for conll exp4 i.e. highest preference to illinois
                # applyRulesCoNLL(text, spotOutFile, nerOutFile,illinoisOutFile, entitiesOutFile,flag)#last argument=flag is 1 normally, 0 for conll exp4 i.e. highest preference to illinois
            # if i>=245: break
            text=''
    f.close()

def testing():
    example_sent = "member UK government"
    example_sent = example_sent.split(' ')
    t2 = 'Government_of_the_United_Kingdom'
    t2 = t2.split('_')
    stop_words = stopwords.words('english')
    # word_tokens = word_tokenize(example_sent)
    filtered_sentence = [w for w in example_sent if not w in stop_words]
    filtered_sentence2 = [w for w in t2 if not w in stop_words]

    synsets1 = []
    for w in example_sent:
        x = wn.synsets(w)
        for synset in x:
            y = [str(lemma.name()) for lemma in synset.lemmas()]
            synsets1 = synsets1 + y

    print(example_sent)
    print(filtered_sentence)
    print(t2)
    print(filtered_sentence2)

def writeIllinoisNEstoConllFile():
    # folder = 'outputOKE\\train\illinois\inConllFormat\\'
    folder = 'outputOKE\\train\illinois\miscIncluded\\'
    print(os.listdir(folder))
    i = -1
    for iFile in os.listdir(folder):
        i += 1
        illinoisFile = folder + iFile
        # illinoisFile = 'outputOKE\\train\illinois\miscIncluded\outputIllinois' + str(i) + '.txt'  # for train dataset
        nerFile = 'outputOKE\\train\\ner\inConllFormat\outputNER' + str(i) + '.txt'  # for train dataset
        with open(illinoisFile, 'r', encoding='utf-8', errors='ignore') as f:
            nerList = json.load(f)
        f = io.open(nerFile, 'r', encoding='utf-8')
        # writeFile = folder+file
        writeFile = 'outputOKE\\train\illinois\inConllFormat\output' + str(i) + '.txt'
        wF = io.open(writeFile, 'a+', encoding='utf-8')
        for line in f:
            # line = line.encode('utf-8')
            line = line.strip()
            print(line.strip())
            line += '\tO\n'
            wF.write(line)
        f.close()
        wF.close()
        if i>=1:break


def evalAll():
    # p1=2
    for p2 in [0,1,2,4,5]:
        for p1 in [1,2]:
        #     evaluateCoNLL(p1,p2,flagE4=1)  ##for evaluation on CoNLL dataset, 1=train, 2=eval argument2=0 for HmaraNER's evaluation, 1 for Illinois, 2 for Stanford
            evaluateOKE(p1,p2)  # for evaluation on OKE dataset #1=train,2=eval , argument2=0 for HmaraNER's evaluation, 1 for Illinois, 2 for Stanford, 3 spotlight, 4 fox, 5 adel


#FOR RELATION EXTRACTION

def extractTACREDinputTexts():
    # Following are for TACRED FAMILY RELATIONS
    inFile = "inputTACRED\\familyRelations\\trainFamRelInputTokens.txt"
    outFile = "inputTACRED\\familyRelations\\trainFamRelInputTexts.txt"
    # inFile = "inputTACRED\\familyRelations\\devFamRelInputTokens.txt"
    # outFile = "inputTACRED\\familyRelations\\devFamRelInputTexts.txt"
    # inFile = "inputTACRED\\familyRelations\\testFamRelInputTokens.txt"
    # outFile = "inputTACRED\\familyRelations\\testFamRelInputTexts.txt"

    #Following are for FULL TACRED DATASET
    # inFile = "inputTACRED\\trainInputTokens.txt"
    # outFile = "inputTACRED\\trainInputTexts.txt"
    # inFile = "inputTACRED\\devInputTokens.txt"
    # outFile = "inputTACRED\\devInputTexts.txt"
    # inFile = "inputTACRED\\testInputTokens.txt"
    # outFile = "inputTACRED\\testInputTexts.txt"
    with open(inFile, 'r', encoding='utf-8', errors='ignore') as inF, open(outFile, 'a', encoding='utf-8', errors='ignore') as outF:
        oneText = ''
        for line in inF:  # read a file one line at a time using for loop
            # line = line.encode('utf-8')
            line = line.strip('\n')
            if line!='':
                # if not line.isalnum() and len(line)==1:
                #     oneText = oneText[:-1]
                oneText = oneText + line + ' '
            elif line=='' and oneText!='':# if its blank line, it means start of next input text
                outF.write(oneText)
                outF.write('\n')
                oneText = ''

def extractTACREDinputTextsFromJson():
    # Following are for TACRED FAMILY RELATIONS: both positive and negative examples
    # inFile = "inputTACRED\\familyAll\\train.json"
    # outFile = "inputTACRED\\familyAll\\trainInputTexts.txt"
    # inFile = "inputTACRED\\familyAll\\dev.json"
    # outFile = "inputTACRED\\familyAll\\devInputTexts.txt"
    # inFile = "inputTACRED\\familyAll\\test.json"
    # outFile = "inputTACRED\\familyAll\\testInputTexts.txt"

    # inFile = "inputTACRED\\familyAllCorrected\\train.json" #CORRECTED
    # outFile = "inputTACRED\\familyAllCorrected\\trainInputTexts.txt"#CORRECTED
    inFile = "inputTACRED\\familyAllCorrected\\test.json" #CORRECTED
    outFile = "inputTACRED\\familyAllCorrected\\testInputTexts.txt"#CORRECTED

    with open(inFile) as inF_json, open(outFile, 'a', encoding='utf-8', errors='ignore') as outF:
        outF.truncate(0)
        data = json.load(inF_json)
        for exampleNo,example in enumerate(data):  # read json data, one example at a time
            oneText = ' '
            tokens = example['token']
            tokens = ['(' if x == '-LRB-' else x for x in tokens]  # replacing all '-LRB-' tokens by '('
            tokens = [')' if x == '-RRB-' else x for x in tokens]  # replacing all '-RRB-' tokens by ')'
            tokens = ['[' if x == '-LSB-' else x for x in tokens]  # replacing all '-LSB-' tokens by '['
            tokens = [']' if x == '-RSB-' else x for x in tokens]  # replacing all '-RSB-' tokens by ']'
            oneText = oneText.join(tokens)
            outF.write(oneText)
            outF.write('\n')

def writeStanfordOutputFileForTACRED(dataset,corrected):#reads stanford RE output files from outputTACRED\test\familyAll\stanford and writes to one file stanfordOutputRE.txt
    if corrected==True:
        folder = "familyAllCorrected"
    else: folder = "familyAll"

    inFileTacred = os.path.normpath("inputTACRED/"+folder+"/"+dataset+".json")
    inFilesPathStanford = os.path.normpath("outputTACRED/"+dataset+"/"+folder+"/stanford/outputRE")
    outFileStanford = os.path.normpath("outputTACRED/"+dataset+"/"+folder+"/stanfordOutputRE.txt")

    if dataset=='ours':
        inFileTacred = os.path.normpath("inputOurFREdataset/datasetAnnotated.json")
        inFilesPathStanford = os.path.normpath("outputOurFREdataset/stanford/outputRE")
        outFileStanford = os.path.normpath("outputOurFREdataset/stanfordOutputRE.txt")
    elif dataset=='cust':
        inFileTacred = os.path.normpath("inputCustFREdataset/datasetAnnotated.json")
        inFilesPathStanford = os.path.normpath("outputCustFREdataset/stanford/outputRE")
        outFileStanford = os.path.normpath("outputCustFREdataset/stanfordOutputRE.txt")

    with open(inFileTacred) as inFTacred_json, open(outFileStanford, 'a', encoding='utf-8', errors='ignore') as outF:
        outF.truncate(0)
        data = json.load(inFTacred_json)
        for exampleNo, example in enumerate(data):  # read json data, one example at a time
            subjStart = example['subj_start']
            subjEnd = example['subj_end']
            # subjType = example['subj_type']
            objStart = example['obj_start']
            objEnd = example['obj_end']
            # objType = example['obj_type']
            if type(subjStart) is dict:
                subjStart = int(subjStart['$numberInt'])
                objStart = int(objStart['$numberInt'])
                subjEnd = int(subjEnd['$numberInt'])
                objEnd = int(objEnd['$numberInt'])

            inFileStanford = os.path.normpath(inFilesPathStanford+str(exampleNo)+'.json')
            with open(inFileStanford) as inFStanford_json:
                triples = json.load(inFStanford_json)
            # for writing predicted relation to outFile
            # if a triple is found for the subject object in triples, then write its relation to file
            # else write no_relation to file
            relFound = False
            for triple in triples:
                subjPredictedIndex = triple['objectSpan'][0]#start index
                objPredictedIndex = triple['subjectSpan'][0]
                rel = triple['relation']
                if (subjPredictedIndex in range(subjStart, subjEnd + 1) and objPredictedIndex in range(objStart,
                                                                                                       objEnd + 1)):
                    if rel == 'per:children' or rel=='per:parents' or rel == 'per:spouse' or rel == 'per:siblings' or rel == 'per:other_family':
                        relFound = True
                        outF.write(rel)
                        outF.write('\n')
                        print('Example No: ')
                        print(exampleNo + 1, rel)
                        print('\n')
                        break
            if relFound == False:
                outF.write('no_relation')  # if not found, then no_relation
                outF.write('\n')
                print('Example No: ')
                print(exampleNo + 1)
                print('\n')

def extractFamilyNoRelationsInputTexts():
    famRelSentences = [] #list that will hold the input texts of family relation examples
    # Following are for TACRED FAMILY RELATIONS Sentences
    inFile = "inputTACRED\\familyRelations\\trainFamRelInputTexts.txt"
    # inFile = "inputTACRED\\familyRelations\\devFamRelInputTexts.txt"
    # inFile = "inputTACRED\\familyRelations\\testFamRelInputTexts.txt"
    with open(inFile, 'r', encoding='utf-8', errors='ignore') as inF:
        for line in inF:  # read a file one line at a time using for loop
            # line = line.encode('utf-8')
            line = line.strip('\n')
            if line!='':
                famRelSentences.append(line)

    #Following are for
    inFile = "inputTACRED\\noRelation\\trainNoRelations.conll"
    outFile = "inputTACRED\\familyNoRelations\\train.conll"
    outFile2 = "inputTACRED\\familyNoRelations\\trainTexts.txttxt"
    # inFile = "inputTACRED\\noRelation\\devNoRelations.conll"
    # outFile = "inputTACRED\\familyNoRelations\\dev.conll"
    # outFile2 = "inputTACRED\\familyNoRelations\\devTexts.txt"
    # inFile = "inputTACRED\\noRelation\\testNoRelations.conll"
    # outFile = "inputTACRED\\familyNoRelations\\test.conll"
    # outFile2 = "inputTACRED\\familyNoRelations\\testTexts.txt"
    with open(inFile, 'r', encoding='utf-8', errors='ignore') as inF, open(outFile, 'a', encoding='utf-8', errors='ignore') as outF, open(outFile2, 'a', encoding='utf-8', errors='ignore') as outF2:
        oneText = ''
        oneExample = ''
        for line in inF:  # read a file one line at a time using for loop
            # line = line.encode('utf-8')
            line = line.strip('\n')
            if line!='':
                oneExample = oneExample + line +'\n'
                if not(line.startswith('#')):
                    word = line.split('\t')[1]
                    oneText = oneText + word + ' '
            elif line=='' and oneText!='':# if its blank line, it means start of next input text
                if oneText in famRelSentences:
                    outF.write(oneExample)
                    outF.write('\n')
                    outF2.write(oneText)
                    outF2.write('\n')
                oneText = ''
                oneExample = ''

def extractFamilyRelationsFromTACRED():
    # inFile = "inputTACRED\original\\train.conll"
    # outFile = "inputTACRED\\familyRelations\\trainFamilyRelations.conll"
    # inFile = "inputTACRED\original\\dev.conll"
    # outFile = "inputTACRED\\familyRelations\\devFamilyRelations.conll"
    inFile = "inputTACRED\original\\test.conll"
    outFile = "inputTACRED\\familyRelations\\testFamilyRelations.conll"
    familyRels = ['per:spouse', 'per:siblings', 'per:parents', 'per:other_family', 'per:children']
    familyFlag = False
    with open(inFile, 'r', encoding='utf-8', errors='ignore') as inF, open(outFile, 'a', encoding='utf-8',
                                                                           errors='ignore') as outF:
        for line in inF:  # read a file one line at a time using for loop
            # line = line.encode('utf-8')
            if line != '\n':
                lineParts = line.split()
                if lineParts[1].startswith('id='):
                    if lineParts[3].startswith('reln='):
                        relation = lineParts[3].lstrip('reln=')
                    if relation in familyRels:
                        familyFlag = True
                        print("Family Relation")
                    else:
                        familyFlag = False
                        print("Not Family Relation")
            if familyFlag:
                outF.write(line)

def extractNoRelationsFromTACRED():
    inFile = "inputTACRED\original\\train.conll"
    outFile = "inputTACRED\\noRelation\\trainNoRelations.conll"
    # inFile = "inputTACRED\original\\dev.conll"
    # outFile = "inputTACRED\\noRelation\\devNoRelations.conll"
    # inFile = "inputTACRED\original\\test.conll"
    # outFile = "inputTACRED\\noRelation\\testNoRelations.conll"
    noRel = ['no_relation']
    noFlag = False
    with open(inFile, 'r', encoding='utf-8', errors='ignore') as inF, open(outFile, 'a', encoding='utf-8',
                                                                           errors='ignore') as outF:
        for line in inF:  # read a file one line at a time using for loop
            # line = line.encode('utf-8')
            if line != '\n':
                lineParts = line.split()
                if lineParts[1].startswith('id='):
                    if lineParts[3].startswith('reln='):
                        # relation = lineParts[3].lstrip('reln=')
                        relation = lineParts[3][5:] #cutting first 5 characters i.e. reln=
                    if relation in noRel:
                        noFlag = True
                        print("No Relation")
                    else:
                        noFlag = False
                        print("Not a no-Relation")
            if noFlag:
                outF.write(line)

def writeNoOfPersons():#this function reads patterns.txt file from outputTACRED folder and writes the no of persons/perPronouns (excluding subject and object) and no of relation words in each example to file noOfPersons.txt
    inFile = "outputTACRED\\completedTest\\familyAllCorrected\patterns.txt"
    outFile = "outputTACRED\\completedTest\\familyAllCorrected\\noOfPersons.txt"
    # inFile = "outputOurFREdatasetNew2\patterns.txt"
    # outFile = "outputOurFREdatasetNew2\\noOfPersons.txt"
    with open(inFile,'r',encoding='utf-8',errors='ignore') as inF, open(outFile, 'a', encoding='utf-8',errors='ignore') as outF:
        outF.truncate(0)
        for line in inF:
            print(str.strip(line))
            noOfRelWords = str.count(line,"w")+str.count(line,"r") #no of relation words in example
            print("No of relation words: "+str(noOfRelWords))
            persons = re.findall("[pP]+",line)
            noOfPersons = len(persons)
            # line = re.sub("[^pP]"," ",line)
            # line = re.split("\s+",line)
            # noOfPersons = len(line)
            print("No of persons / person pronouns: " + str(noOfPersons)+"\n")
            outF.write(str(noOfPersons))
            outF.write("\t")
            outF.write(str(noOfRelWords))
            outF.write("\n")

def OKEcountTokSent():#only added this func to get no of tokens and sentences in OKE dataset for Thesis purpose
    inputFileOkeTrain = 'inputOKE\okeTrainInputTexts.txt'  # for TRAINING dataset
    inputFileOkeEval = 'inputOKE\okeEvalInputTexts.txt' #for EVALUATION dataset
    fTrain = io.open(inputFileOkeTrain, 'r', encoding='utf-8')
    fEval = io.open(inputFileOkeEval, 'r', encoding='utf-8')
    trainText = fTrain.read()
    evalText = fEval.read()
    nlp = spacy.load('en_core_web_lg')
    docTrain = nlp(trainText)
    docEval = nlp(evalText)
    i = 0
    j = 0
    for (i,sent) in enumerate(docTrain.sents):
        print(str(i)+sent.text)
    for (j,sent) in enumerate(docEval.sents):
        print(str(j)+sent.text)
    print()
    print("Sentence count of OKE Train data set is: "+str(i+1)) # +1 because indices start from 0
    print("Sentence count of OKE Eval data set is: "+str(j+1))
    print("Token count of OKE Train data set is: "+str(len(docTrain)))
    print("Token count of OKE Eval data set is: "+str(len(docEval)))
    print()

def generateREinputTexts():
    inFile = os.path.normpath("inputOurFREdatasetNew2/datasetAnnotated.json")
    outFile = os.path.normpath("inputOurFREdatasetNew2/inputTexts.txt")
    with open(inFile, 'r', encoding='utf-8', errors='ignore') as f, open(outFile,'a',encoding='utf-8',errors='ignore') as outF1:
        datasetList = json.load(f) # list of dicts
        outF1.truncate(0)
        for dataDict in datasetList:
            sentenceTokens = dataDict['token']
            sentence = TreebankWordDetokenizer().detokenize(sentenceTokens)
            sentence = sentence.replace('-LRB-', '(')
            sentence = sentence.replace('-RRB-', ')')
            sentence = sentence.replace('-LSB-', '[')
            sentence = sentence.replace('-RSB-', ']')
            sentence = sentence.replace('\\\"', '\"')
            sentence = sentence.replace('\'\'', '\"')
            sentence = sentence.replace('``', '\"')

            print(sentence)
            outF1.write(sentence)
            outF1.write('\n')

if __name__ == "__main__":
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time Before Running CustNER on test dataset =", current_time)

    # print(wn.synsets('dog'))
    # OKEcountTokSent()
    # testing()
    # okeAnnotateNerSpot()#gets oke dataset annotated from: 1.Illinois NER, 2.Stanford NER, 3. dbpedia spotlight
    # writeIllinoisNEstoConllFile()
    # okeAnnotateHmari(trainOrEval=2, flag=1)# trainOrEval: 1=train, 2=eval
    # cleanStanNer() #one time task
    # removeMISCentities(1,1,1)#one time task #trainOrEval=1=train,2=eval    #system: 1 Illinois, 2 Stanford    # okeOrConll: 1=oke, 2=conll
    # evaluateOKE(2,5) # for evaluation on OKE dataset #1=train,2=eval , argument2=0 for HmaraNER's evaluation, 1 for Illinois, 2 for Stanford, 4 for FOX, 5 for ADEL

    # conllSample()
    # countNEs()
    # conllConvert()
    # writeCoNLLentities(1)#one time task #1=train, 2=eval
    # conllAnnotateNerSpot()
    # conllCheckNerSpotFiles()
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time After Annotations and Before Running CustNER Rules on test dataset =", current_time)

    conllAnnotateHmari(trainOrEval=2,flag=1)#1=train, 2=eval .. flag is 1 normally, 0 for conll exp4 i.e. highest preference to illinois
    # evaluateCoNLL(trainOrEval=2,system=0,flagE4=1) ##for evaluation on CoNLL dataset, 1=train, 2=eval argument2=0 for HmaraNER's evaluation, 1 for Illinois, 2 for Stanford  .. flagE4 is 1 normally, 0 for conll exp4 i.e. highest preference to illinois
    # evalAll()
    # convertFoxAdelFromNIFtoTxt()
    # compareCoNLL(1,flagE4=0) #1=train, 2=eval  .. flagE4 is 1 normally, 0 for conll exp4 i.e. highest preference to illinois

    # sentence = 'Richmond McDavid Flowers was born in Dothan on Nov 11, 1918, the youngest of four sons of John and Ila McDavid Flowers.'
    # getDotDepParse(sentence)

    #FOR RELATION EXTRACTION
    # extractFamilyRelationsFromTACRED()
    # extractTACREDinputTexts()
    # extractNoRelationsFromTACRED()
    # extractFamilyNoRelationsInputTexts()#check this , which files over written
    # extractTACREDinputTextsFromJson()

    #MongoDB
    # connectToDB()

    # For RELATION EXTRACTION
    # generateREdatasetConllFormat()  # for generating our RE dataset #then manually annotated this generated dataset #then converted dataset from conll to json by running generate_json.py
    # generateREinputTexts()
    # writeNoOfPersons() #ONE TIME TASK - writes the no of persons in each TACRED example to file #first apply rules, then run this
        # outputRE1 and 2 for rule 1 and 2

    # applyRulesRE(dataset='completedTest',corrected=True,examples='both') # dataset = 'train'or'dev'or'test' or 'ours'or'ours1'or'ours2'or 'cust', examples = 'positive' or 'negative' or 'both' ### dataset = 'ours' for ourFREdataset, the next 2 arguments are meaningless in our dataset case
                # scoreRE(dataset='train',corrected=True,examples='both') #NOT NEEDED for test set, use TACRED's Scorer instead
    # tacredAnnotateStanfordRE(dataset='completedTest',corrected=True) #dataset='test' or 'ours' or 'cust' # for Stanfords RE system (corenlp's) # One time task
    # writeStanfordOutputFileForTACRED(dataset='completedTest',corrected=True) #for writing Stanford RE output file #One time task

    # scoreREsenWise(dataset='completedTest',corrected=True,predictionFileName="outputCustRE.txt") # dataset=cust, or (train or test or completedTest)if TACTRED. 'ours' # "outputCustRE.txt" "outputRESpanbertBase.txt" # stanfordOutputRE.txt # outputRESpanbertBaseTrainedOnFamilyAllEvaluatedOnFamilyAllCorrected.txt # TACRED's Scorer modified to score sentence lenght wise
#    generateREdatasetConllFormat()
    # annotateStanfordCoref('His brother was the writer Aldous Huxley, and half-brother a fellow biologist and Nobel laureate, Andrew Huxley; his father was writer and editor Leonard Huxley; and his paternal grandfather was biologist T. H. Huxley, famous as a colleague and supporter of Charles Darwin.')#just for testing coref output

    # text = 'CANADA IS IN JAPAN'
    # text = text.capitalize()
    # print(text)

    # writeNERoutputFileConll()
    # writeNERoutputFileOthers('ours', 'eval')

    #FOR TRAINING NER
    # makeSpacyTrainDataFromOKE()#ONE TIME TASK.. no need to do again.
    # trainNERWithSpacy() #training done, models saved, for both original and corrected oke
    # runTrainedNER()
    # writeNERoutputFileOthers(system='stanford',trainOrEval='train')#system: ours , illinois, stanford
    # evaluateNER(system='trainedNer', trainOrEval='eval', dataset='oke')#system: ours , illinois, stanford # dataset is conll or oke
    # writeNERoutputFileOke()#for OKE dataset, this writes the output files to one output file in format required for SemEval

    ## FOR GENDER # #FOR ANNOTATING GENDER OF A PERSON
    ## rules for recognizing gender are added in custNER
    ## custNER entities with gender are written outputOKE\eval\hmariEntities\allEntitiesWithGender
    ## #I have annotated only oke eval corrected dataset person entities with gender
    ## OKE entities with gender are written to folder outputOKE\eval\okeEntitiesCorrected\withGender by running evaluateGenderOKE()
    # evaluateGenderOKE() #from inside this is called convertOKEdatasetFromNIFtoTxt(), ONE TIME TASK

    # applyRulesGender(dataset='conll',trainEval='eval') #dataset='oke' or 'conll', trainEval='train' or 'eval'
    # applyRulesGender(dataset='oke',trainEval='eval') # for evaluation purpose, take oke eval corrected entities and annotate the person entities with gender using custNER
    # writeGenderOutputFileOke(folder='genderAnnotationsByCustNER+DictLookup') #reads files from the folder and writes one file "allEntities.txt" in seqeval format in same folder
    # evaluateNER('luke', 'eval', 'conll')#PRE-REQUISITE: FIRST CHECK THAT THE VARIABLE "directory" INSIDE THIS FUNCTION IS UPDATED  #evaluate using seqeval script

    # for completing TACRED-F test dataset
    # readExcel()
    # readExcel2()

    # #FOR VIEWING DEPENDENCY TREE OF A TEXT
    # # make sure nltk can find stanford-parser
    # # please check your stanford-parser version from brew output (in my case 3.6.0)
    # os.environ['CLASSPATH'] = r'/usr/local/Cellar/stanford-parser/3.6.0/libexec'
    # os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
    # sentence = 'Besides his son Charles , an author and journalist , Kaiser is survived by his wife of 67 years , Hannah Greeley Kaiser ; two other sons , Robert G , a former managing editor and now an associate editor of The Washington Post , and David , a professor of history at the Naval War College in Newport , RI ; and four grandchildren .'
    # sdp = CoreNLPDependencyParser()
    # result = list(sdp.raw_parse(sentence))
    # dep_tree_dot_repr = [parse for parse in result][0].to_dot()
    # source = Source(dep_tree_dot_repr, filename="dep_tree", format="png")
    # source.view()

    #testing text recognition from image
    # If you don't have tesseract executable in your PATH, include the following:
    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
    # print(pytesseract.image_to_string(Image.open('tests\data\\3140051440.jpg')))

    #testing Neural Networks
    # applyNN()

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time After Running CustNER on test dataset =", current_time)
