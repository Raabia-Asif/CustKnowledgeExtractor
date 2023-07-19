# from keras_models.utils.nlp import tokenize
import nltk
import spacy
from seqeval.metrics import f1_score, accuracy_score, classification_report
from rdflib import Graph
from rdflib.util import guess_format
# import pprint
import json
import io
import os

nlp = spacy.load('en_core_web_sm')

def writeNERoutputFileConll():#this writes the named entities of Conll03 dataset files in format required for ner evaluation i.e. like this [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
    # inputFile = 'inputConll\conllpp\conllpp_test.txt'  # Corrected CONLL EVALUATION DATASET
    inputFile = 'inputConll\conll2003.eng.testb.txt'  # CONLL EVALUATION DATASET #'inputConll\conll2003.eng.train.txt'  # CONLL Training DATASET
    # outputFile = 'outputConll\\eval\conllEntities\\conllppAllEntitiesWithMisc.txt'  # Corrected CoNLL Entities output file
    outputFile = 'outputConll\\eval\conllEntities\\allEntitiesWithMisc.txt'  # CoNLL Entities output file
    allEntities = []
    entities = []
    with open(inputFile, 'r', encoding='utf-8', errors='ignore') as inF:
        # inputData = inF.read()
        docNo = -1
        for line in inF:
            if (line.find('-DOCSTART-') == -1 and line.find('-DOCEND-') == -1):  # within a doc
                if line != '\n':
                    line = line.split('\n')[0]
                    x = line.split(' ')
                    lineFirstTok = x[0]  # the token
                    lineLastTok = x[len(x) - 1]  # the ner type
                    entities.append(lineLastTok)
            elif (line.find('-DOCSTART-') != -1 or line.find('-DOCEND-') != -1):  # start of next doc
                if docNo >= 0:
                    allEntities.append(entities)
                docNo += 1
                entities = []
    with open(outputFile, 'w', encoding='utf-8') as outF:
        json.dump(allEntities, outF, sort_keys=True, ensure_ascii=False)


def writeNERoutputFileOke():#this writes the named entities of OKE dataset files in format required for ner evaluation script i.e. like this [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
    inputFile = 'inputOKE\okeEvalInputTexts.txt'  # OKE DATASET sentences
    inputFile2Dir = 'outputOKE\\eval\\hmariEntities\\allEntities\\output'  #'outputOKE\\eval\okeEntitiesOriginal\\doc'  # oke Entities files directory #
    outputFile = 'outputOKE\\eval\\hmariEntities\\allEntities\\allEntities2.txt'  #'outputOKE\\eval\okeEntitiesOriginal\\allEntities.txt'  # oke Entities output file #
    allEntities = []
    entities = []
    with open(inputFile, 'r', encoding='utf-8', errors='ignore') as inF:
        for docNo,line in enumerate(inF):#one line of inF is one doc
            if docNo==50:
                print()
            # tokens = tokenize(line)
            # tokens = nltk.word_tokenize(line)
            doc = nlp(line)
            tokens = [token.text for token in doc]
            inputFile2 = inputFile2Dir + str(docNo) + '.txt'
            inF2 = io.open(inputFile2, 'r', encoding='utf-8', errors='ignore')
            okeEntities = json.load(inF2)
            inF2.close()
            entityNo = 0
            charNoOkeFile = charEndNoOkeFile = 0  # this keeps track of the token being read (its stard and end character no) in the oke input sentences file
            if len(okeEntities)>0:
                entity = okeEntities[entityNo][0] #[0] is added for hmariEntities only
                entityStart = int(entity['characterOffsetBegin'])
                entityEnd = int(entity['characterOffsetEnd'])
            else: entityStart = entityEnd = 0
            consecutiveSameTypeEntities = False
            for token in tokens:
                ner = "O"
                # if token=="``":
                #     token = "\""
                # if token=="\'\'":
                #     # print(token)
                #     token = "\""
                #     # print(token)
                # if docNo==31 and token=="Jr.":
                #     print(len(token))
                temp = str.find(line,token,charEndNoOkeFile-1)
                if temp!= -1:
                    charNoOkeFile = temp
                    charEndNoOkeFile = charNoOkeFile + len(token)
                else:
                    charNoOkeFile = charEndNoOkeFile + 1
                    charEndNoOkeFile = charNoOkeFile + len(token)
                if charNoOkeFile > entityEnd:
                    entityNo += 1
                    if entityNo < len(okeEntities):
                        entity = okeEntities[entityNo][0] #[0] is added for hmariEntities only
                        entityStart = int(entity['characterOffsetBegin'])
                        entityEnd = int(entity['characterOffsetEnd'])
                offsetRangeEntity = range(entityStart, entityEnd)  # offset range of current entity
                offsetRangeTokenRead = range(charNoOkeFile, charEndNoOkeFile)  # offset range of current token being read
                offsetRangesOverlap = [i for i in offsetRangeEntity if i in offsetRangeTokenRead]
                if len(offsetRangesOverlap) > 0:# if entityStart <= charNoOkeFile and charNoOkeFile < entityEnd:
                    ner = entity['ner']
                    text = entity['text']
                    if entityNo > 0 and entityNo < len(okeEntities):
                        prevEntity = okeEntities[entityNo - 1][0] #[0] is added for hmariEntities only
                        prevNer = prevEntity['ner']
                        if prevNer == ner:
                            if entityStart == int(prevEntity['characterOffsetEnd']) + 1:
                                if text.split(' ')[0] == token:  # if its the first token of this entity, because B- tag is only for beginning of entity (which is consecutive and having same type)
                                    consecutiveSameTypeEntities = True

                    ner = ner[0] + ner[1] + ner[2]  # getting just first 3 characters of ner i.e. loc, org and per
                    ner = str.upper(ner)
                    if consecutiveSameTypeEntities == False:
                        ner = "I-" + ner
                    else:
                        ner = "B-" + ner
                        consecutiveSameTypeEntities = False

                entities.append(ner)
            entTokCount1 = entTokCount2 = 0
            for ent1 in okeEntities:
                ent1 = ent1[0] #[0] is added for hmariEntities only
                text = ent1['text']
                # ent1Toks = nltk.word_tokenize(text)
                docE = nlp(text)
                ent1Toks = [token.text for token in docE]
                entTokCount1 += len(ent1Toks)#(text.split(' '))
            for ent2 in entities:
                if ent2 != "O":
                    entTokCount2 += 1
            if entTokCount2 != entTokCount1:
                print("docNo: " + str(docNo) + " , ENTITIES list is NOT CORRECTLY CREATED!!!!!!!!!!!!")
                print(entTokCount1)
                print(entTokCount2)
                print(entities)
                # break
            allEntities.append(entities)
            entities = []
    with open(outputFile, 'w', encoding='utf-8') as outF:
        json.dump(allEntities, outF, sort_keys=True, ensure_ascii=False)

def writeGenderOutputFileOke(folder):  # this writes the male/female of OKE dataset files in format required for ner evaluation script i.e. like this [['O', 'O', 'O', 'B-MALE', 'I-MALE', 'I-MALE', 'O'], ['B-FEMALE', 'I-FEMALE', 'O']]
    inputFile = 'inputOKE\okeEvalInputTexts.txt'  # OKE DATASET sentences
    inputFile2Dir = 'outputOKE\\eval\okeEntitiesCorrected\\'+folder+'\\output'
    outputFile = 'outputOKE\\eval\okeEntitiesCorrected\\'+folder+'\\allEntities.txt'
    allEntities = []
    entities = []
    with open(inputFile, 'r', encoding='utf-8', errors='ignore') as inF:
        for docNo, line in enumerate(inF):  # one line of inF is one doc
            tokens = nltk.word_tokenize(line)
            inputFile2 = inputFile2Dir + str(docNo) + '.txt'
            inF2 = io.open(inputFile2, 'r', encoding='utf-8', errors='ignore')
            okeEntities = json.load(inF2)
            inF2.close()
            entityNo = 0
            charNoOkeFile = charEndNoOkeFile = 0  # this keeps track of the token being read (its stard and end character no) in the oke input sentences file
            if len(okeEntities) > 0:
                entity = okeEntities[entityNo]
                entityStart = int(entity['characterOffsetBegin'])
                entityEnd = int(entity['characterOffsetEnd'])
            else:
                entityStart = entityEnd = 0
            consecutiveSameTypeEntities = False
            for token in tokens:
                gender = "O"
                if token == "``":
                    token = "\""
                if token == "\'\'":
                    # print(token)
                    token = "\""
                    # print(token)
                # if docNo == 38 and token == "de":
                #     print(len(token))
                temp = str.find(line, token, charEndNoOkeFile - 1)
                if temp != -1:
                    charNoOkeFile = temp
                    charEndNoOkeFile = charNoOkeFile + len(token)
                else:
                    charNoOkeFile = charEndNoOkeFile + 1
                    charEndNoOkeFile = charNoOkeFile + len(token)
                if charNoOkeFile > entityEnd:
                    entityNo += 1
                    if entityNo < len(okeEntities):
                        entity = okeEntities[entityNo]
                        entityStart = int(entity['characterOffsetBegin'])
                        entityEnd = int(entity['characterOffsetEnd'])
                offsetRangeEntity = range(entityStart, entityEnd)  # offset range of current entity
                offsetRangeTokenRead = range(charNoOkeFile,charEndNoOkeFile)  # offset range of current token being read
                offsetRangesOverlap = [i for i in offsetRangeEntity if i in offsetRangeTokenRead]
                if len(offsetRangesOverlap) > 0:  # if entityStart <= charNoOkeFile and charNoOkeFile < entityEnd:
                    ner = entity['ner']
                    text = entity['text']
                    if 'gender' in entity.keys():
                        prevGender = ""
                        gender = entity['gender']
                        if entityNo > 0 and entityNo < len(okeEntities):
                            prevEntity = okeEntities[entityNo - 1]
                            prevNer = prevEntity['ner']
                            if 'gender' in prevEntity.keys():
                                prevGender = prevEntity['gender']
                            if prevNer == ner =='PERSON':
                                if prevGender!="" and prevGender == gender:
                                    if entityStart == int(prevEntity['characterOffsetEnd']) + 1:
                                        if text.split(' ')[
                                            0] == token:  # if its the first token of this entity, because B- tag is only for beginning of entity (which is consecutive and having same type)
                                            consecutiveSameTypeEntities = True

                        if consecutiveSameTypeEntities == False:
                            gender = "I-" + gender
                        else:
                            gender = "B-" + gender
                            consecutiveSameTypeEntities = False

                entities.append(gender)
            entTokCount1 = entTokCount2 = 0 #token count of male/female entities
            for ent1 in okeEntities:
                if 'gender' in ent1.keys():
                    text = ent1['text']
                    ent1Toks = nltk.word_tokenize(text)
                    entTokCount1 += len(ent1Toks)  # (text.split(' '))
                    if 'Jr.' in text:
                        entTokCount1 -= 1
            for ent2 in entities:
                if ent2 != "O":
                    entTokCount2 += 1
            if entTokCount2 != entTokCount1:
                print("docNo: " + str(docNo) + " , ENTITIES list is NOT CORRECTLY CREATED!!!!!!!!!!!!")
                print(entTokCount1)
                print(entTokCount2)
                print(entities)
                # break
            allEntities.append(entities)
            entities = []
    with open(outputFile, 'w', encoding='utf-8') as outF:
        json.dump(allEntities, outF, sort_keys=True, ensure_ascii=False)


# def writeNERoutputFileOkeOthers():#for oke dataset, this writes the named entities of input file in format required for ner evaluation i.e. like this [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
#     inputFile = 'inputOKE\okeEvalInputTexts.txt'  # OKE DATASET sentences
#     inputFile2Dir = 'outputOKE\\eval\okeEntities\\doc'  # oke Entities files directory
#     outputFile = 'outputOKE\\eval\okeEntities\\allEntities.txt'  # oke Entities output file
#     allEntities = []
#     entities = []
#     with open(inputFile, 'r', encoding='utf-8', errors='ignore') as inF:
#         for docNo,line in enumerate(inF):#one line of inF is one doc
#             tokens = tokenize(line)
#             inputFile2 = inputFile2Dir + docNo + '.txt'
#             inF2 = io.open(inputFile2, 'r', encoding='utf-8', errors='ignore')
#             okeEntities = json.load(inF2)
#             inF2.close()
#             systemEntityNo = 0
#             charNoConllFile = -1  # this keeps track of the character no being read in the conll dataset file
#             systemEntityEnd = 0
#             consecutiveSameTypeEntities = False
#             for token in tokens:
#                 ner = "O"
#                 if charNoConllFile > systemEntityEnd:
#                     systemEntityNo += 1
#                 if systemEntityNo < len(systemEntities):
#                     systemEntity = systemEntities[systemEntityNo]
#                     if system == 'ours':
#                         # print(line)
#                         systemEntity = systemEntity[0]
#                 systemEntityStart = systemEntity['characterOffsetBegin']
#                 systemEntityEnd = systemEntity['characterOffsetEnd']
#                 if systemEntityStart <= charNoConllFile and charNoConllFile < systemEntityEnd:
#                     if 'ner' in dict.keys(systemEntity):
#                         ner = systemEntity['ner']
#                     elif 'label' in dict.keys(systemEntity):
#                         ner = systemEntity['label']
#                     if 'text' in dict.keys(systemEntity):
#                         text = systemEntity['text']
#                     elif 'tokens' in dict.keys(systemEntity):
#                         text = systemEntity['tokens']
#
#                     if systemEntityNo > 0 and systemEntityNo < len(systemEntities):
#                         prevSystemEntity = systemEntities[systemEntityNo - 1]
#                         if system == 'ours':
#                             prevSystemEntity = prevSystemEntity[0]
#                         if 'ner' in dict.keys(prevSystemEntity):
#                             prevNer = prevSystemEntity['ner']
#                         elif 'label' in dict.keys(prevSystemEntity):
#                             prevNer = prevSystemEntity['label']
#                         if prevNer == ner:
#                             if systemEntityStart == prevSystemEntity['characterOffsetEnd'] + 1:
#                                 if text.split(' ')[
#                                     0] == lineFirstTok:  # if its the first token of this entity, because B- tag is only for beginning of entity (which is consecutive and having same type)
#                                     consecutiveSameTypeEntities = True
#
#                     if ner.lower() == 'city' or ner.lower() == 'country' or ner.upper() == 'STATE_OR_PROVINCE':  # this check added for stanford system
#                         ner = 'LOCATION'
#                     ner = ner[0] + ner[1] + ner[2]  # getting just first 3 characters of ner i.e. loc, org and per
#                     ner = str.upper(ner)
#                     if consecutiveSameTypeEntities == False:
#                         ner = "I-" + ner
#                     else:
#                         ner = "B-" + ner
#                         consecutiveSameTypeEntities = False
#
#                 entities.append(ner)
#                 charNoConllFile = charNoConllFile + len(lineFirstTok) + 1
#
#                 if token matches in entity:
#                     ner = entity['ner']
#
#                 entities.append(ner)
#             for ent1 in systemEntities:
#                 if system == 'ours':
#                     ent1 = ent1[0]
#                 if 'text' in dict.keys(ent1):
#                     text = ent1['text']
#                 elif 'tokens' in dict.keys(ent1):
#                     text = ent1['tokens']
#                 entTokCount1 += len(text.split(' '))
#             for ent2 in entities:
#                 if ent2 != "O":
#                     entTokCount2 += 1
#             if entTokCount2 != entTokCount1:
#                 print("docNo: " + str(docNo) + " , ENTITIES list is NOT CORRECTLY CREATED!!!!!!!!!!!!")
#                 print(entTokCount1)
#                 print(entTokCount2)
#                 print(entities)
#                 # break
#             if len(conllEntities[docNo]) != len(entities):
#                 print("docNo: " + str(docNo) + " , ENTITIES list SIZE is NOT CORRECT!!!!!!!!!!!!")
#                 print(len(conllEntities[docNo]))
#                 print(len(entities))
#                 # break
#             allEntities.append(entities)
#             entities = []
#     with open(outputFile, 'w', encoding='utf-8') as outF:
#         json.dump(allEntities, outF, sort_keys=True, ensure_ascii=False)

def writeNERoutputFileOthers(system,trainOrEval):#for conll dataset, this writes the named entities of input file in format required for ner evaluation i.e. like this [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
    if trainOrEval=='train':
        inputFile = 'inputConll\conll2003.eng.train.txt'  # CONLL Training DATASET #
    elif trainOrEval == 'eval':
        inputFile = 'inputConll\conll2003.eng.testb.txt'  # CONLL EVALUATION DATASET
    directory = 'outputConll\\'+trainOrEval+'\\'
    conllEntitiesFile = 'outputConll\\'+trainOrEval+'\conllEntities\\allEntities.txt'  # CoNLL Entities file
    if system=='ours':
        directory = directory + 'hmariEntities\\allEntities\\'
        inputFile2Dir = directory + '\\output'
        outputFile = directory + '\\allEntities.txt'  #
    elif system=='illinois':
        directory = directory + 'illinois\\miscRemoved\\allEntities\\'
        inputFile2Dir = directory + '\\outputIllinois'
        outputFile = directory + '\\allEntities.txt'
    elif system=='stanford':
        directory = directory + 'stanford\\cleaned\\miscRemoved\\allEntities\\'
        inputFile2Dir = directory + '\\outputNER'
        outputFile = directory + '\\allEntities.txt'
    allEntities = []
    entities = []
    conllEntitiesF = io.open(conllEntitiesFile,'r',encoding='utf-8',errors='ignore')
    conllEntities = json.load(conllEntitiesF)
    conllEntitiesF.close()
    with open(inputFile, 'r', encoding='utf-8', errors='ignore') as inF:
        # inputData = inF.read()
        docNo = -1
        inF2 = None
        systemEntityNo = 0
        charNoConllFile = -1 #this keeps track of the character no being read in the conll dataset file
        systemEntityEnd = 0
        consecutiveSameTypeEntities = False
        for tokNo,line in enumerate(inF):
            ner = "O"
            if (line.find('-DOCSTART-') == -1 and line.find('-DOCEND-') == -1):  # within a doc
                # if docNo<13: continue
                # if systemEntityNo==7:
                #     print()
                if line != '\n':
                    line = line.split('\n')[0]
                    x = line.split(' ')
                    lineFirstTok = x[0]  # first token of a line of conll dataset i.e. the token
                    if charNoConllFile > systemEntityEnd:
                        systemEntityNo += 1
                    if systemEntityNo<len(systemEntities):
                        systemEntity = systemEntities[systemEntityNo]
                        if system=='ours':
                            # print(line)
                            systemEntity = systemEntity[0]
                    systemEntityStart = systemEntity['characterOffsetBegin']
                    systemEntityEnd = systemEntity['characterOffsetEnd']
                    if systemEntityStart<=charNoConllFile and charNoConllFile<systemEntityEnd:
                        if 'ner' in dict.keys(systemEntity):
                            ner = systemEntity['ner']
                        elif 'label' in dict.keys(systemEntity):
                            ner = systemEntity['label']
                        if 'text' in dict.keys(systemEntity):
                            text = systemEntity['text']
                        elif 'tokens' in dict.keys(systemEntity):
                            text = systemEntity['tokens']

                        if systemEntityNo>0 and systemEntityNo<len(systemEntities):
                            prevSystemEntity = systemEntities[systemEntityNo-1]
                            if system=='ours':
                                prevSystemEntity = prevSystemEntity[0]
                            if 'ner' in dict.keys(prevSystemEntity):
                                prevNer = prevSystemEntity['ner']
                            elif 'label' in dict.keys(prevSystemEntity):
                                prevNer = prevSystemEntity['label']
                            if prevNer==ner:
                                if systemEntityStart == prevSystemEntity['characterOffsetEnd']+1:
                                    if text.split(' ')[0] == lineFirstTok: #if its the first token of this entity, because B- tag is only for beginning of entity (which is consecutive and having same type)
                                        consecutiveSameTypeEntities = True

                        if ner.lower()=='city' or ner.lower()=='country' or ner.upper()=='STATE_OR_PROVINCE' :#this check added for stanford system
                            ner = 'LOCATION'
                        ner = ner[0] + ner[1] + ner[2]  # getting just first 3 characters of ner i.e. loc, org and per
                        ner = str.upper(ner)
                        if consecutiveSameTypeEntities==False:
                            ner = "I-"+ner
                        else:
                            ner = "B-"+ner
                            consecutiveSameTypeEntities = False

                    entities.append(ner)
                    charNoConllFile = charNoConllFile + len(lineFirstTok) + 1
                else:
                    charNoConllFile = charNoConllFile + 1

            elif (line.find('-DOCSTART-') != -1 or line.find('-DOCEND-') != -1):  # start of next doc
                if docNo >= 0: #for things that should not be done before first doc
                    #check that the entities list is correct
                    entTokCount1 = entTokCount2 = 0
                    for ent1 in systemEntities:
                        if system == 'ours':
                            ent1 = ent1[0]
                        if 'text' in dict.keys(ent1):
                            text = ent1['text']
                        elif 'tokens' in dict.keys(ent1):
                            text = ent1['tokens']
                        entTokCount1 += len(text.split(' '))
                    for ent2 in entities:
                        if ent2!="O":
                           entTokCount2 +=1
                    if entTokCount2!=entTokCount1:
                        print("docNo: "+str(docNo)+" , ENTITIES list is NOT CORRECTLY CREATED!!!!!!!!!!!!")
                        print(entTokCount1)
                        print(entTokCount2)
                        print(entities)
                        # break
                    if len(conllEntities[docNo]) != len(entities):
                        print("docNo: " + str(docNo) + " , ENTITIES list SIZE is NOT CORRECT!!!!!!!!!!!!")
                        print(len(conllEntities[docNo]))
                        print(len(entities))
                        # break
                    allEntities.append(entities)
                    if inF2!=None:
                        inF2.close()
                    charNoConllFile = 0
                consecutiveSameTypeEntities = False
                systemEntityNo = 0
                if docNo+1 < len(conllEntities):
                    docNo += 1
                    entities = []
                    inputFile2 = inputFile2Dir + str(docNo) + '.txt'  #
                    inF2 = io.open(inputFile2, 'r', encoding='utf-8', errors='ignore')
                    systemEntities = json.load(inF2)
    with open(outputFile, 'w', encoding='utf-8') as outF:
        json.dump(allEntities, outF, sort_keys=True, ensure_ascii=False)

#conll dataset uses IOB1 tagging scheme #this evaluates ner systems using seqeval
def evaluateNER(system,trainOrEval,dataset):#PRE-REQUISITE: FIRST CHECK THAT THE VARIABLE "directory" IS UPDATED
    if dataset=='conll':
        directory = 'outputConll\\' + trainOrEval + '\\'
        datasetEntitiesFile = 'outputConll\\' + trainOrEval + '\conllEntities\\conllppAllEntities.txt'  # Corrected CoNLL Entities file
        # datasetEntitiesFile = 'outputConll\\' + trainOrEval + '\conllEntities\\allEntities.txt'  # CoNLL Entities file
        if system == 'ours':
            directory = directory + 'hmariEntities\\allEntities\\'
            systemFile = directory + '\\allEntities.txt'
        elif system == 'illinois':
            directory = directory + 'illinois\\miscRemoved\\allEntities\\'
            systemFile = directory + '\\allEntities.txt'
        elif system == 'stanford':
            directory = directory + 'stanford\\cleaned\\miscRemoved\\allEntities\\'
            systemFile = directory + '\\allEntities.txt'
        elif system == 'luke':
            datasetEntitiesFile = 'outputConll/eval/conllEntities/allEntitiesIn1.txt'  # CoNLL Entities file
            # datasetEntitiesFile = 'outputConll/eval/conllEntities/conllppAllEntitiesIn1.txt'  # Corrected CoNLL Entities file
            systemFile = 'outputConll/eval/luke/allEntities.txt'
    elif dataset=='oke':
        directory = 'outputOKE\\' + trainOrEval + '\\' #
        datasetEntitiesFile = 'outputOKE\\' + trainOrEval +'\okeEntitiesCorrected\withoutGender\\allEntities2.txt'  # OKE Entities file
        # datasetEntitiesFile = 'outputOKE\\' + trainOrEval +'\okeEntitiesCorrected\withGender\\allEntities.txt'  # OKE Entities file with gender
        if system == 'ours':
            # directory = directory +'\okeEntitiesCorrected\genderAnnotationsByCustNER+DictLookup'# # THIS NEEDS TO BE UPDATED EVERYTIME ACCORDINGLY
            directory = directory +'hmariEntities\\allEntities'# # THIS NEEDS TO BE UPDATED EVERYTIME ACCORDINGLY
        if system == 'trainedNer':
            directory = directory + '\\trainedNER\\trainOnOriginalEvalOnCorrected'
        systemFile = directory + '\\allEntities2.txt'
    with open(datasetEntitiesFile, 'r', encoding='utf-8', errors='ignore') as goldF, open(systemFile, 'r', encoding='utf-8', errors='ignore') as predF:
        y_true = json.load(goldF) #[['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        y_pred = json.load(predF) #[['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
    print(f1_score(y_true, y_pred))
    print(classification_report(y_true, y_pred, digits=4))

def writeCoNLLentities(trainOrEval):#
    # trainOrEval=1=train, trainOrEval=2=eval

    #pre processing
    if trainOrEval==1:#train
        datasetFile = 'inputConll\conll2003.eng.train.txt' #CONLL Training DATASET
        outFilePath = 'outputConll\\train\conllEntities\doc' #CoNLL Entities file path

    elif trainOrEval==2:#eval
        datasetFile = 'inputConll\conll2003.eng.testb.txt' #CONLL EVALUATION DATASET
        outFilePath = 'outputConll\eval\conllEntities\doc' #CoNLL Entities file path

    print("Output file path: "+outFilePath)
    f = io.open(datasetFile, 'r', encoding='utf-8')

    docNo = -1
    prevLineLastTok = 'O'
    e1OffsetBegin = e2OffsetBegin = e1OffsetEnd = -1
    eText = prevEText = ''
    newEntity = True
    entities = []
    for line in f:  # read the file one line at a time using for loop
        if (line.find('-DOCSTART-')==-1 and line.find('-DOCEND-')==-1): #within a doc
            if line != '\n':
                line = line.split('\n')[0]
                offsetBegin = offsetConll
            # print(line)
            x = line.split(' ')
            lineFirstTok = x[0] #first token of a line of conll dataset i.e. the token
            lineLastTok = x[len(x)-1] #last token of a line of conll dataset i.e. the ner type

            offsetConll += len(lineFirstTok)
            offsetEnd = offsetConll
            if line !='\n':
                offsetConll += 1

            if prevLineLastTok.startswith("B-") and lineLastTok.startswith("I-"): # if its not a new entity
                x = prevLineLastTok.strip("B-")
                y = lineLastTok.strip("I-")
                if x == y:
                    newEntity = False
            elif lineLastTok != prevLineLastTok and line!='\n':
                newEntity = True

            if lineLastTok != prevLineLastTok and line!='\n' and newEntity==True: #if its a new entity
                newEntity = True
                entity = {}
                eText = eText.split('\n')[0]
                prevEText = eText
                eText = lineFirstTok
                if prevLineLastTok.find('PER') != -1 or prevLineLastTok.find('LOC') != -1 or prevLineLastTok.find('ORG') != -1:  # if previous was an entity
                    e1Ner = prevLineLastTok # previous entity's ner
                    e1OffsetEnd = offsetBegin - 1  # previous entity's end offset
                    e1OffsetBegin = e2OffsetBegin  # previous entity's begin offset
                    if prevLineEmpty==True:
                        e1OffsetEnd -= 1
                    entity['characterOffsetBegin'] = e1OffsetBegin
                    entity['characterOffsetEnd'] = e1OffsetEnd
                    entity['ner'] = e1Ner.split('-')[1]
                    prevEText = str.strip(prevEText)
                    entity['text'] = prevEText
                    entities.append(entity)

                if lineLastTok.find('PER')!=-1 or lineLastTok.find('LOC')!=-1 or lineLastTok.find('ORG')!=-1: # if its an entity
                    e2Ner = lineLastTok #this entity's ner
                    e2OffsetBegin = offsetBegin #this entity's begin offset

            else:
                eText = eText + " " + lineFirstTok

            if line!='\n':
                prevLineLastTok = lineLastTok  # last token of previous line
                prevLineEmpty = False
            else:
                prevLineEmpty = True
        elif (line.find('-DOCSTART-')!=-1 or line.find('-DOCEND-')!=-1): #start of next doc
            if docNo>=0:
                outFile = outFilePath + str(docNo) + '.txt'
                with open(outFile, 'w', encoding='utf-8') as outfile:
                    json.dump(entities, outfile, sort_keys=True, indent=4, ensure_ascii=False)
            offsetConll = -1
            docNo += 1
            print("\nDOCUMENT NO: "+ str(docNo) + "\n")
            eText = prevEText = ''
            lineLastTok = prevLineLastTok = 'O'
            e1OffsetBegin = e2OffsetBegin = e1OffsetEnd = -1
            # print("start of next doc: "+str(docNo))
            flag = False
            newEntity = True
            entities=[]

def removeMISCentities(trainOrEval,system,okeOrConll):
    # trainOrEval=1=train, trainOrEval=2=eval
    # system: 1 for Illinois, 2 for Stanford
    # okeOrConll: 1=oke, 2=conll
    if trainOrEval == 1:  # train
        subDir = "\\train"
    elif trainOrEval == 2:  # eval
        subDir = "\eval"

    if system == 1:
        annotator = "\illinois\\"
        key = 'label'
        outputFile = 'outputIllinois'
    elif system == 2:
        annotator = "\\ner\cleaned\\"
        key = 'ner'
        outputFile = 'outputNER'

    if okeOrConll == 1:  # oke
        dir = "outputOKE"
    elif okeOrConll == 2:  # conll
        dir = "outputConll"

    filePath = dir+subDir+annotator #Entities file path
    print("Output file path: " + filePath+"miscRemoved\\")

    entities = []
    for filename in os.listdir(filePath):
        if filename.endswith('.txt'):
            readFile = filePath+filename
            writeFile = filePath+"miscRemoved\\"+filename
            with open(readFile, 'r', encoding='utf-8', errors='ignore') as inF:
                entities = json.load(inF)
            newEntities = entities.copy()
            for e in entities:
                if system == 1:#illinois
                    if e[key]=='MISC':
                        newEntities.remove(e)
                elif system == 2:#stanford ner
                    if not(e[key].startswith('PER')or e[key].startswith('LOC')or e[key].startswith('CITY')or e[key].startswith('COUNTRY') or e[key].startswith('STATE_OR_PROVINCE') or e[key].startswith('ORG')):
                        newEntities.remove(e)
            with open(writeFile, 'w', encoding='utf-8', errors='ignore') as outF:
                json.dump(newEntities, outF, sort_keys=True, indent=4, ensure_ascii=False)

def compareCoNLL(trainOrEval,flagE4):#
    # trainOrEval=1=train, trainOrEval=2=eval
    # flagE4 is 1 normally, 0 for conll exp4 i.e. highest preference to illinois

    #pre processing
    if trainOrEval==1:#train
        datasetFile = 'inputConll\conll2003.eng.train.txt' #CONLL Training DATASET
        if flagE4==1:
            hmariFilePath = 'outputConll\\train\hmariEntities\output' #HmaraNER
            resultFile = 'results\ComparisonResultsOnTrainDataCoNLL.txt'  # HmaraNER
        elif flagE4==0:
            hmariFilePath = 'outputConll\\train\hmariEntities\exp4\output' #HmaraNER
            resultFile = 'results\ComparisonResultsExp4OnTrainDataCoNLL.txt' #HmaraNER
        illinoisFilePath = 'outputConll\\train\illinois\outputIllinois' #Illinois
        stanfordFilePath = 'outputConll\\train\\ner\cleaned\outputNER' #Stanford


    elif trainOrEval==2:#eval
        datasetFile = 'inputConll\conll2003.eng.testb.txt' #CONLL EVALUATION DATASET
        if flagE4 == 1:
            hmariFilePath = 'outputConll\eval\hmariEntities\output' #HmaraNER
            resultFile = 'results\ComparisonResultsOnTrainDataCoNLL.txt'  # HmaraNER
        elif flagE4 == 0:
            hmariFilePath = 'outputConll\eval\hmariEntities\exp4\output' #HmaraNER
            resultFile = 'results\ComparisonResultsExp4OnEvalDataCoNLL.txt' #HmaraNER
        illinoisFilePath = 'outputConll\eval\illinois\outputIllinois' # Illinois
        stanfordFilePath = 'outputConll\eval\\ner\cleaned\outputNER' # Stanford


    print("Output file: "+resultFile)

    noOfDatasetEntities = 0  # the total no of entities in training/evaluation dataset
    noOfDatasetPerEntities = 0  # the total no of PERSON entities in training/evaluation dataset
    noOfDatasetLocEntities = 0  # the total no of LOCATION entities in training/evaluation dataset
    noOfDatasetOrgEntities = 0  # the total no of ORGANIZATION entities in training/evaluation dataset

    hpCount=0; hnCount=0; ipCount=0; inCount=0

    f = io.open(datasetFile, 'r', encoding='utf-8')
    f2 = open(resultFile, "w+")
    docNo = -1
    prevLineLastTok = 'O'
    e1OffsetBegin = e2OffsetBegin = e1OffsetEnd = -1
    eText = prevEText = ''
    newEntity = True
    for line in f:  # read the file one line at a time using for loop
        if (line.find('-DOCSTART-')==-1 and line.find('-DOCEND-')==-1): #within a doc
            if line != '\n':
                line = line.split('\n')[0]
                offsetBegin = offsetConll
            # print(line)
            x = line.split(' ')
            lineFirstTok = x[0] #first token of a line of conll dataset
            lineLastTok = x[len(x)-1] #last token of a line of conll dataset

            offsetConll += len(lineFirstTok)
            offsetEnd = offsetConll
            if line !='\n':
                offsetConll += 1

            if hmariEntityNo < len(hmariEntities):
                hmariEntity = hmariEntities[hmariEntityNo][0] #for HmaraNER
            if illinoisEntityNo < len(illinoisEntities):
                illinoisEntity = illinoisEntities[illinoisEntityNo] # for Illinois
                illinoisEntity['ner'] = illinoisEntity['label']  # for Illinois
                illinoisEntity['text'] = illinoisEntity['tokens']  # for Illinois
            # if stanfordEntityNo < len(stanfordEntities):
            #     stanfordEntity = stanfordEntities[stanfordEntityNo] # for Stanford
            #     if stanfordEntity['ner'] == 'CITY' or stanfordEntity['ner'] == 'STATE_OR_PROVINCE' or stanfordEntity['ner'] == 'COUNTRY':
            #         stanfordEntity['ner'] = 'LOCATION'

            #if its not per/loc/org entity e.g misc in case of illinois
            while flag==False and (not(str.startswith(illinoisEntity['ner'],'PER')) and not(str.startswith(illinoisEntity['ner'],'LOC')) and not(str.startswith(illinoisEntity['ner'],'ORG'))) :
                illinoisEntityNo +=1
                if illinoisEntityNo < len(illinoisEntities):
                    illinoisEntity = illinoisEntities[illinoisEntityNo]  # for Illinois
                    illinoisEntity['ner'] = illinoisEntity['label']
                    illinoisEntity['text'] = illinoisEntity['tokens']
                elif illinoisEntityNo >= len(illinoisEntities):#if its the last entity and is not ppo type
                    flag = True
                    continue

            # flag = False
            # # if its not per/loc/org entity e.g date title etc in case of stanford
            # while flag == False and (not (str.startswith(stanfordEntity['ner'], 'PER')) and not (
            # str.startswith(stanfordEntity['ner'], 'LOC')) and not (
            # str.startswith(stanfordEntity['ner'], 'ORG'))):
            #     stanfordEntityNo += 1
            #     if stanfordEntityNo < len(stanfordEntities):
            #         stanfordEntity = stanfordEntities[stanfordEntityNo]  # for Stanford
            #         if stanfordEntity['ner'] == 'CITY' or stanfordEntity['ner'] == 'STATE_OR_PROVINCE' or \
            #                         stanfordEntity['ner'] == 'COUNTRY':
            #             stanfordEntity['ner'] = 'LOCATION'
            #     elif stanfordEntityNo >= len(stanfordEntities):  # if its the last entity and is not ppo type
            #         flag = True
            #         continue

            if prevLineLastTok.startswith("B-") and lineLastTok.startswith("I-"): # if its not a new entity
                x = prevLineLastTok.strip("B-")
                y = lineLastTok.strip("I-")
                if x == y:
                    newEntity = False

            if lineLastTok != prevLineLastTok and line!='\n' and newEntity==True: #if its a new entity
                newEntity = True
                eText = eText.split('\n')[0]
                prevEText = eText
                eText = lineFirstTok
                if prevLineLastTok.find('PER') != -1 or prevLineLastTok.find('LOC') != -1 or prevLineLastTok.find('ORG') != -1:  # if previous was an entity
                    e1Ner = prevLineLastTok # previous entity's ner
                    e1OffsetEnd = offsetBegin - 1  # previous entity's end offset
                    e1OffsetBegin = e2OffsetBegin  # previous entity's begin offset
                    if prevLineEmpty==True:
                        e1OffsetEnd -= 1

                if lineLastTok.find('PER')!=-1 or lineLastTok.find('LOC')!=-1 or lineLastTok.find('ORG')!=-1: # if its an entity
                    noOfDatasetEntities += 1
                    e2Ner = lineLastTok #this entity's ner
                    e2OffsetBegin = offsetBegin #this entity's begin offset
                    if lineLastTok.find('PER')!=-1: #if this entity is tagged person
                        noOfDatasetPerEntities += 1
                    elif lineLastTok.find('LOC')!=-1: #if this entity is tagged location
                        noOfDatasetLocEntities += 1
                    elif lineLastTok.find('ORG')!=-1: #if this entity is tagged organization
                        noOfDatasetOrgEntities += 1
            else:
                eText = eText + " " + lineFirstTok

            if e1OffsetBegin != -1:
                offsetRangeDataset = range(e1OffsetBegin,e1OffsetEnd)
                offsetRangeHmari = range(hmariEntity['characterOffsetBegin'], hmariEntity['characterOffsetEnd'])
                offsetRangeIllinois = range(illinoisEntity['characterOffsetBegin'], illinoisEntity['characterOffsetEnd'])
                # offsetRangeStanford = range(stanfordEntity['characterOffsetBegin'], stanfordEntity['characterOffsetEnd'])
                offsetRangesOverlapHmari = [i for i in offsetRangeDataset if i in offsetRangeHmari]
                offsetRangesOverlapIllinois = [i for i in offsetRangeDataset if i in offsetRangeIllinois]
                # offsetRangesOverlapStanford = [i for i in offsetRangeDataset if i in offsetRangeStanford]
                e1OffsetBegin2 = e1OffsetBegin #this is to back up e1OffsetBegin as it will change in following if block and is yet needed in next if block
                # print("e:\t\t" + str(e1OffsetBegin) + "\t" + str(e1OffsetEnd))
                # print("hmari:\t" + str(hmariEntity['characterOffsetBegin']) + "\t" + str(hmariEntity['characterOffsetEnd']))
                if len(offsetRangesOverlapHmari) > 0 or len(offsetRangesOverlapIllinois) > 0:# or len(offsetRangesOverlapStanford) > 0 :
                    datasetEntityNo += 1
                    e1OffsetBegin = -1

                if len(offsetRangesOverlapHmari) > 0 and hmariEntityNo < len(hmariEntities):
                    hmariEntityNo += 1
                    # following code is added for STRONG annotation
                    if e1OffsetBegin2 == hmariEntity['characterOffsetBegin'] and e1OffsetEnd == hmariEntity['characterOffsetEnd'] and hmariEntityNo <= len(hmariEntities):
                        # CASE 1: offsets match exactly
                        sth=1
                    elif len(offsetRangesOverlapHmari) > 0 and hmariEntityNo < len(hmariEntities):
                        # CASE 2: offsets match partially i.e. they overlap
                        hpCount+=1
                        hnCount+=1
                        f2.write("HmaraNER FP: " + hmariEntity['text'] + "\t" + str(
                            hmariEntity['characterOffsetBegin']) + "\t" + str(
                            hmariEntity['characterOffsetEnd']) + "\t" + hmariEntity['ner']+ "\n")
                        f2.write("HmaraNER FN e: " + prevEText + '\t' + str(e1OffsetEnd) + "\t" + e1Ner+ "\n")
                else:
                    if (hmariEntity['characterOffsetBegin'] > e1OffsetEnd) or (hmariEntityNo >= len(hmariEntities)):
                        f2.write("HmaraNER FN e: " + prevEText + '\t' + str(e1OffsetEnd) + "\t" + e1Ner+ "\n")
                        hnCount += 1
                        datasetEntityNo += 1
                        e1OffsetBegin = -1
                    elif (e1OffsetBegin2 > hmariEntity['characterOffsetEnd']):
                        hmariEntityNo += 1
                        hpCount += 1
                        f2.write("HmaraNER FP: " + hmariEntity['text'] + "\t" + str(
                            hmariEntity['characterOffsetBegin']) + "\t" + str(
                            hmariEntity['characterOffsetEnd']) + "\t" + hmariEntity['ner']+ "\n")

                if len(offsetRangesOverlapIllinois) > 0 and illinoisEntityNo < len(illinoisEntities):
                    illinoisEntityNo += 1
                    if e1OffsetBegin2 == illinoisEntity['characterOffsetBegin'] and e1OffsetEnd == illinoisEntity['characterOffsetEnd'] and illinoisEntityNo <= len(illinoisEntities):
                        # CASE 1: offsets match exactly
                        sth = 1
                    elif len(offsetRangesOverlapIllinois) > 0 and illinoisEntityNo < len(illinoisEntities):
                        # CASE 2: offsets match partially i.e. they overlap
                        ipCount += 1
                        inCount += 1
                        f2.write("IllinoisNER FP: " + illinoisEntity['text'] + "\t" + str(
                            illinoisEntity['characterOffsetBegin']) + "\t" + str(
                            illinoisEntity['characterOffsetEnd']) + "\t" + illinoisEntity['ner']+ "\n")
                        f2.write("IllinoisNER FN e: " + prevEText + '\t' + str(e1OffsetEnd) + "\t" + e1Ner+ "\n")
                else:
                    if (illinoisEntity['characterOffsetBegin'] > e1OffsetEnd) or (illinoisEntityNo >= len(illinoisEntities)):
                        inCount+=1
                        f2.write("IllinoisNER FN e: " + prevEText + '\t' + str(e1OffsetEnd) + "\t" + e1Ner+ "\n")
                        datasetEntityNo += 1
                        e1OffsetBegin = -1
                    elif (e1OffsetBegin2 > illinoisEntity['characterOffsetEnd']):
                        illinoisEntityNo += 1
                        ipCount+=1
                        f2.write("IllinoisNER FP: " + illinoisEntity['text'] + "\t" + str(
                            illinoisEntity['characterOffsetBegin']) + "\t" + str(
                            illinoisEntity['characterOffsetEnd']) + "\t" + illinoisEntity['ner']+ "\n")

                # if len(offsetRangesOverlapStanford) > 0 and stanfordEntityNo < len(stanfordEntities):
                #     stanfordEntityNo += 1
                #     if e1OffsetBegin2 == stanfordEntity['characterOffsetBegin'] and e1OffsetEnd == stanfordEntity['characterOffsetEnd'] and stanfordEntityNo <= len(stanfordEntities):
                #         # CASE 1: offsets match exactly
                #         sth = 1
                #     elif len(offsetRangesOverlapStanford) > 0 and stanfordEntityNo < len(stanfordEntities):
                #         # CASE 2: offsets match partially i.e. they overlap
                #         f2.write("StanfordNER FP: " + stanfordEntity['text'] + "\t" + str(
                #             stanfordEntity['characterOffsetBegin']) + "\t" + str(
                #             stanfordEntity['characterOffsetEnd']) + "\t" + stanfordEntity['ner']+ "\n")
                #         f2.write("StanfordNER FN e: " + prevEText + '\t' + str(e1OffsetEnd) + "\t" + e1Ner+ "\n")
                # else:
                #     if (stanfordEntity['characterOffsetBegin'] > e1OffsetEnd) or (stanfordEntityNo >= len(stanfordEntities)):
                #         f2.write("StanfordNER FN e: " + prevEText + '\t' + str(e1OffsetEnd) + "\t" + e1Ner+ "\n")
                #         datasetEntityNo += 1
                #         e1OffsetBegin = -1
                #     elif (e1OffsetBegin2 > stanfordEntity['characterOffsetEnd']):
                #         stanfordEntityNo += 1
                #         f2.write("StanfordNER FP: " + stanfordEntity['text'] + "\t" + str(
                #             stanfordEntity['characterOffsetBegin']) + "\t" + str(
                #             stanfordEntity['characterOffsetEnd']) + "\t" + stanfordEntity['ner']+ "\n")

            if line!='\n':
                prevLineLastTok = lineLastTok  # last token of previous line
                prevLineEmpty = False
            else:
                prevLineEmpty = True
        elif (line.find('-DOCSTART-')!=-1 or line.find('-DOCEND-')!=-1): #start of next doc
            if hnCount!=inCount:
                f2.write(str(hnCount) + "\t" + str(inCount))
            docNo += 1
            f2.write("\nDOCUMENT NO: "+ str(docNo) + "\n")
            eText = prevEText = ''
            lineLastTok = prevLineLastTok = 'O'
            e1OffsetBegin = e2OffsetBegin = e1OffsetEnd = -1
            # print("start of next doc: "+str(docNo))
            flag = False
            newEntity = True
            if docNo>-1 and docNo<946:
                hmariFile = hmariFilePath + str(docNo) + '.txt'
                illinoisFile = illinoisFilePath + str(docNo) + '.txt'
                # stanfordFile = stanfordFilePath + str(docNo) + '.txt'
                with open(hmariFile, 'r', encoding='utf-8', errors='ignore') as outfile2:
                    hmariEntities = json.load(outfile2)  # hmaraNER output file loaded as json dictionary
                with open(illinoisFile, 'r', encoding='utf-8', errors='ignore') as outfile3:
                    illinoisEntities = json.load(outfile3)  # illonois NER output file loaded as json dictionary
                # with open(stanfordFile, 'r', encoding='utf-8', errors='ignore') as outfile4:
                #     stanfordEntities = json.load(outfile4)  # stanford NER output file loaded as json dictionary
                hmariEntityNo = 0
                illinoisEntityNo = 0
                # stanfordEntityNo = 0
                datasetEntityNo = 0
                offsetConll = -1
        # if docNo==39:
        #     print()
    print(str(hpCount)+"\t"+str(ipCount))
    print(str(hnCount) + "\t" + str(inCount))

def evaluateCoNLL(trainOrEval,system, flagE4):#
    # trainOrEval=1=train, trainOrEval=2=eval
    # system: the system for evaluation, 0=HmaraNER, 1 for Illinois, 2 for Stanford
    # flagE4 is 1 normally, 0 for conll exp4 i.e. highest preference to illinois
    if system==0:
        annotator = "HmaraNER"
    elif system==1:
        annotator = "IllinoisNER"
    elif system==2:
        annotator = "StanfordNER"

    #pre processing
    if trainOrEval==1:#train
        sizeOfDataset=946
        datasetFilePath = 'outputConll\\train\conllEntities\doc'  # CONLL TRAINING DATASET FILES' path
        if system==0:#HmaraNER
            if flagE4==1:
                hmariFilePath = 'outputConll\\train\hmariEntities\\allEntities\output' #HmaraNER
                mistakesFilePath = 'outputConll\\train\hmariEntities\mistakes\output' #HmaraNER
                resultFile = 'results\hmarayResultsOnTrainDataCoNLL.txt'  # HmaraNER
            elif flagE4==0:
                hmariFilePath = 'outputConll\\train\hmariEntities\exp4\\r7\output' #HmaraNER
                # hmariFilePath = 'outputConll\\train\hmariEntities\exp4\output' #HmaraNER
                resultFile = 'results\Exp4 rules\\r7Exp4hmarayResultsOnTrainDataCoNLL.txt'  # HmaraNER
                # resultFile = 'results\Exp4hmarayResultsOnTrainDataCoNLL.txt'  # HmaraNER
        elif system == 1:#Illinois
            hmariFilePath = 'outputConll\\train\illinois\miscRemoved\\allEntities\outputIllinois' # 1: Illinois
            mistakesFilePath = 'outputConll\\train\illinois\miscRemoved\mistakes\outputIllinois' # 1: Illinois
            resultFile = 'results\illinoisResultsOnTrainDataCoNLL.txt' # 2: Illinois
        elif system == 2:#Stanford
            hmariFilePath = 'outputConll\\train\\ner\cleaned\miscRemoved\\allEntities\outputNER' # 1: Stanford
            mistakesFilePath = 'outputConll\\train\\ner\cleaned\miscRemoved\mistakes\outputNER' # 1: Stanford
            resultFile = 'results\stanfordResultsOnTrainDataCoNLL.txt' # 2: Stanford


    elif trainOrEval==2:#eval
        sizeOfDataset=231
        datasetFilePath = 'outputConll\eval\conllEntities\doc' #CONLL EVALUATION DATASET FILES' path
        if system == 0:#HmaraNER
            if flagE4 == 1:
                hmariFilePath = 'outputConll\eval\hmariEntities\\allEntities\output' #HmaraNER
                mistakesFilePath = 'outputConll\eval\hmariEntities\mistakes\output' #HmaraNER
                resultFile = 'results\hmarayResultsOnEvalDataCoNLL.txt' #HmaraNER
            elif flagE4 == 0:
                hmariFilePath = 'outputConll\eval\hmariEntities\exp4\output'  # HmaraNER
                resultFile = 'results\Exp4hmarayResultsOnEvalDataCoNLL.txt' #HmaraNER
        elif system == 1:#Illinois
            hmariFilePath = 'outputConll\eval\illinois\miscRemoved\\allEntities\outputIllinois' # 1: Illinois
            mistakesFilePath = 'outputConll\eval\illinois\miscRemoved\mistakes\outputIllinois' # 1: Illinois
            resultFile = 'results\illinoisResultsOnEvalDataCoNLL.txt' # 2: Illinois
        elif system == 2:#Stanford
            hmariFilePath = 'outputConll\eval\\ner\cleaned\miscRemoved\\allEntities\outputNER' # 1: Stanford
            mistakesFilePath = 'outputConll\eval\\ner\cleaned\miscRemoved\mistakes\outputNER' # 1: Stanford
            resultFile = 'results\StanfordResultsOnEvalDataCoNLL.txt' # 2: Stanford


    print("Output file: "+resultFile)
    print("Output Mistakes file path: "+mistakesFilePath)

    #evaluation script starts here
    #following variables are for weak annotation
    truePositives = 0;    perTP = 0;    locTP = 0;    orgTP = 0
    falsePositives = 0;    perFP = 0;    locFP = 0;    orgFP = 0
    falseNegatives = 0;    perFN = 0;    locFN = 0;    orgFN = 0

    # following variables are for strong annotation
    struePositives = 0;    sperTP = 0;    slocTP = 0;    sorgTP = 0
    sfalsePositives = 0;    sperFP = 0;    slocFP = 0;    sorgFP = 0
    sfalseNegatives = 0;    sperFN = 0;    slocFN = 0;    sorgFN = 0
    sTPentitiesAll = [];    sFPentitiesAll = [];    sFNentitiesAll = []

    noOfDatasetEntities = 0 #the total no of entities in training/evaluation dataset
    noOfDatasetPerEntities = 0 #the total no of PERSON entities in training/evaluation dataset
    noOfDatasetLocEntities = 0 #the total no of LOCATION entities in training/evaluation dataset
    noOfDatasetOrgEntities = 0 #the total no of ORGANIZATION entities in training/evaluation dataset

    for docNo in range(0, sizeOfDataset): #for each file/text
        # if docNo<39:continue
        sTPentities = [];        sFPentities = [];        sFNentities = []
        if docNo-1>=0:
            print("Doc No: " + str(docNo-1) + "\t\tTP: " + str(struePositives) + "\t\tFP: " + str(
                sfalsePositives) + "\t\tFN: " + str(sfalseNegatives) + "\n\n")
        datasetFile = datasetFilePath+str(docNo)+'.txt'
        hmariFile = hmariFilePath+str(docNo)+'.txt' #HmaraNER output file
        with open(datasetFile, 'r', encoding='utf-8', errors='ignore') as outfile:
            datasetEntities = json.load(outfile)  # dataset file of one sentence/doc/text loaded as json dictionary
        with open(hmariFile, 'r', encoding='utf-8', errors='ignore') as outfile2:
            hmariEntities = json.load(outfile2)  # hmaraNER output file loaded as json dictionary
        hmariEntityNo = 0
        datasetEntityNo = 0
        flag = False
        flag2 = False
        nextDatasetEntity = True
        noOfDatasetEntities += len(datasetEntities)
        while datasetEntityNo<len(datasetEntities) or hmariEntityNo<len(hmariEntities): #for each entity in file/text
            if hmariEntityNo<len(hmariEntities):
                if system==0:
                    hmariEntity = hmariEntities[hmariEntityNo][0] #For HmaraNER
                else:
                    hmariEntity = hmariEntities[hmariEntityNo] #For Illinois/Stanford NER
                    if system == 1:  # Illinois
                        hmariEntity['ner'] = hmariEntity['label']  # 4: for Illinois
                        hmariEntity['text'] = hmariEntity['tokens']  # 4: for Illinois
                    if system == 2:  # Stanford
                        if hmariEntity['ner'] == 'CITY' or hmariEntity['ner'] == 'STATE_OR_PROVINCE' or hmariEntity['ner'] == 'COUNTRY':
                            hmariEntity['ner'] = 'LOCATION'

            # if its not per/loc/org entity e.g misc in case of illinois, date title etc in case of stanford
            while flag == False and (not (str.startswith(hmariEntity['ner'], 'PER')) and not (str.startswith(hmariEntity['ner'], 'LOC')) and not (str.startswith(hmariEntity['ner'], 'ORG'))):
                hmariEntityNo += 1
                if hmariEntityNo < len(hmariEntities):
                    if system == 0:  # HmaraNER
                        hmariEntity = hmariEntities[hmariEntityNo][0]  # for HmaraNER
                    elif system == 1 or system == 2:  # Illinois or Stanford
                        hmariEntity = hmariEntities[hmariEntityNo]  # 3: for Illinois or Stanford
                        if system == 1:  # Illinois
                            hmariEntity['ner'] = hmariEntity['label']  # 4: for Illinois
                            hmariEntity['text'] = hmariEntity['tokens']  # 4: for Illinois
                        if system == 2:  # Stanford
                            if hmariEntity['ner'] == 'CITY' or hmariEntity['ner'] == 'STATE_OR_PROVINCE' or \
                                            hmariEntity['ner'] == 'COUNTRY':
                                hmariEntity['ner'] = 'LOCATION'
                elif hmariEntityNo >= len(hmariEntities):  # if its the last entity and is not ppo type
                    flag = True
                    continue

            if datasetEntityNo<len(datasetEntities):
                datasetEntity = datasetEntities[datasetEntityNo]
                if nextDatasetEntity==True:
                    if str.startswith(datasetEntity['ner'],'PER'):
                        noOfDatasetPerEntities += 1
                    elif str.startswith(datasetEntity['ner'],'LOC'):
                        noOfDatasetLocEntities += 1
                    elif str.startswith(datasetEntity['ner'],'ORG'):
                        noOfDatasetOrgEntities += 1
                    nextDatasetEntity = False

            if datasetEntityNo >= (len(datasetEntities) - 1) and hmariEntityNo >= (len(hmariEntities) - 1):
                flag2 = True
            offsetRangeDataset = range(int(datasetEntity['characterOffsetBegin']), int(datasetEntity['characterOffsetEnd']))
            offsetRangeHmari = range(hmariEntity['characterOffsetBegin'], hmariEntity['characterOffsetEnd'])
            offsetRangesOverlap = [i for i in offsetRangeDataset if i in offsetRangeHmari]

            if system==1: # For Illinois NER
                hmariEntity['ner'] = hmariEntity['label']
            if system == 2:  # Stanford
                if hmariEntity['ner'] == 'LOC' or hmariEntity['ner'] == 'CITY' or hmariEntity['ner'] == 'STATE_OR_PROVINCE' or hmariEntity['ner'] == 'COUNTRY':
                    hmariEntity['ner'] = 'LOCATION'

            if datasetEntity['text']!=hmariEntity['text'] or len(offsetRangesOverlap)==0 or not(hmariEntity['ner'].startswith(datasetEntity['ner'])):
                print("datasetE:  "+datasetEntity['text']+":"+datasetEntity['ner']+'\t\t'+"hmari:  "+hmariEntity['text']+":"+hmariEntity['ner'])

            # FOR WEAK ANNOTATION
            if len(offsetRangesOverlap) > 0 and datasetEntityNo<len(datasetEntities) and hmariEntityNo<len(hmariEntities):
                truePositives += 1
                if str.startswith(hmariEntity['ner'],'PER') and str.startswith(datasetEntity['ner'],'PER'):
                    perTP += 1
                elif str.startswith(hmariEntity['ner'],'PER') and not(str.startswith(datasetEntity['ner'],'PER')):
                    perFP += 1
                elif not(str.startswith(hmariEntity['ner'],'PER')) and str.startswith(datasetEntity['ner'],'PER'):
                    perFN += 1
                if str.startswith(hmariEntity['ner'],'LOC') and str.startswith(datasetEntity['ner'],'LOC'):
                    locTP += 1
                elif str.startswith(hmariEntity['ner'],'LOC') and not(str.startswith(datasetEntity['ner'],'LOC')):
                    locFP += 1
                elif not(str.startswith(hmariEntity['ner'],'LOC')) and str.startswith(datasetEntity['ner'],'LOC'):
                    locFN += 1
                if str.startswith(hmariEntity['ner'],'ORG') and str.startswith(datasetEntity['ner'],'ORG'):
                    orgTP += 1
                elif str.startswith(hmariEntity['ner'],'ORG') and not(str.startswith(datasetEntity['ner'],'ORG')):
                    orgFP += 1
                elif not(str.startswith(hmariEntity['ner'],'ORG')) and str.startswith(datasetEntity['ner'],'ORG'):
                    orgFN += 1
                hmariEntityNo += 1
                datasetEntityNo += 1
                nextDatasetEntity = True
            else:
                if (hmariEntity['characterOffsetBegin'] > int(datasetEntity['characterOffsetBegin']) and datasetEntityNo<len(datasetEntities)) or (hmariEntityNo>=len(hmariEntities)):
                    falseNegatives += 1
                    if str.startswith(datasetEntity['ner'], 'PER'):
                        perFN += 1
                    elif str.startswith(datasetEntity['ner'],'LOC'):
                        locFN += 1
                    elif str.startswith(datasetEntity['ner'],'ORG'):
                        orgFN += 1
                    datasetEntityNo += 1
                    nextDatasetEntity = True
                else:
                    falsePositives += 1
                    if str.startswith(hmariEntity['ner'],'PER'):
                        perFP += 1
                    elif str.startswith(hmariEntity['ner'],'LOC'):
                        locFP += 1
                    elif str.startswith(hmariEntity['ner'],'ORG'):
                        orgFP += 1
                    hmariEntityNo += 1

            # FOR STRONG ANNOTATION
            if datasetEntity['characterOffsetBegin']==hmariEntity['characterOffsetBegin'] and datasetEntity['characterOffsetEnd']==hmariEntity['characterOffsetEnd'] and datasetEntityNo-1<len(datasetEntities) and hmariEntityNo-1<len(hmariEntities):
                #CASE 1: Exact boundary match
                struePositives += 1
                sTPentities.append(hmariEntity)
                if str.startswith(hmariEntity['ner'],'PER') and str.startswith(datasetEntity['ner'],'PER'):
                    sperTP += 1
                elif str.startswith(hmariEntity['ner'],'PER') and str.startswith(datasetEntity['ner'],'LOC'):
                    sperFP += 1
                    slocFN += 1
                elif str.startswith(hmariEntity['ner'],'PER') and str.startswith(datasetEntity['ner'],'ORG'):
                    sperFP += 1
                    sorgFN += 1
                elif str.startswith(hmariEntity['ner'],'LOC') and str.startswith(datasetEntity['ner'],'LOC'):
                    slocTP += 1
                elif str.startswith(hmariEntity['ner'], 'LOC') and str.startswith(datasetEntity['ner'], 'PER'):
                    slocFP += 1
                    sperFN += 1
                elif str.startswith(hmariEntity['ner'], 'LOC') and str.startswith(datasetEntity['ner'], 'ORG'):
                    slocFP += 1
                    sorgFN += 1
                elif str.startswith(hmariEntity['ner'],'ORG') and str.startswith(datasetEntity['ner'],'ORG'):
                    sorgTP += 1
                elif str.startswith(hmariEntity['ner'], 'ORG') and str.startswith(datasetEntity['ner'], 'LOC'):
                    sorgFP += 1
                    slocFN += 1
                elif str.startswith(hmariEntity['ner'], 'ORG') and str.startswith(datasetEntity['ner'], 'PER'):
                    sorgFP += 1
                    sperFN += 1
            elif len(offsetRangesOverlap) > 0:#CASE 2: Partial overlap
                sfalseNegatives += 1
                sfalsePositives += 1
                sFPentities.append(hmariEntity)
                sFNentities.append(datasetEntity)
                if str.startswith(datasetEntity['ner'], 'PER'):
                    sperFN += 1
                elif str.startswith(datasetEntity['ner'], 'LOC'):
                    slocFN += 1
                elif str.startswith(datasetEntity['ner'], 'ORG'):
                    sorgFN += 1
                if str.startswith(hmariEntity['ner'], 'PER'):
                    sperFP += 1
                elif str.startswith(hmariEntity['ner'], 'LOC'):
                    slocFP += 1
                elif str.startswith(hmariEntity['ner'], 'ORG'):
                    sorgFP += 1
            else:#CASE 3: No overlap
                if (hmariEntity['characterOffsetBegin'] > int(datasetEntity['characterOffsetBegin']) and datasetEntityNo<len(datasetEntities)) or (hmariEntityNo>=len(hmariEntities)):
                    if str.startswith(datasetEntity['ner'],'PER'):
                        sfalseNegatives += 1
                        sFNentities.append(datasetEntity)
                        sperFN += 1
                    elif str.startswith(datasetEntity['ner'],'LOC'):
                        sfalseNegatives += 1
                        sFNentities.append(datasetEntity)
                        slocFN += 1
                    elif str.startswith(datasetEntity['ner'],'ORG'):
                        sfalseNegatives += 1
                        sFNentities.append(datasetEntity)
                        sorgFN += 1
                else:
                    if str.startswith(hmariEntity['ner'],'PER'):
                        sfalsePositives += 1
                        sFPentities.append(hmariEntity)
                        sperFP += 1
                    elif str.startswith(hmariEntity['ner'],'LOC'):
                        sfalsePositives += 1
                        sFPentities.append(hmariEntity)
                        slocFP += 1
                    elif str.startswith(hmariEntity['ner'],'ORG'):
                        sfalsePositives += 1
                        sFPentities.append(hmariEntity)
                        sorgFP += 1

            if flag2==True:
                break

        sTPentitiesAll = sTPentitiesAll + sTPentities
        sFPentitiesAll = sFPentitiesAll + sFPentities
        sFNentitiesAll = sFNentitiesAll + sFNentities

        # Following block writes the results of STRONG ANNOTATION to file - for each document in separate file
        try:
            mistakesFile = mistakesFilePath + str(docNo) + '.txt'  # file to write TP, FP, FN
            with open(mistakesFile, 'a', encoding='utf-8') as outfile:
                outfile.truncate(0)
                outfile.write("Input file: %s\n\n" % datasetFile)
                outfile.write("---------------------------------------------------------------------")
                outfile.write("\n----------------- FALSE NEGATIVES = "+ str(len(sFNentities))+" ----------------")
                outfile.write("\n--------------------------------------------------------------\n\n")
                json.dump(sFNentities, outfile, sort_keys = True, indent = 4, ensure_ascii = False)
                outfile.write("\n\n------------------------------------------------------------------")
                outfile.write("\n----------------- FALSE POSITIVES = "+ str(len(sFPentities))+" ----------------")
                outfile.write("\n-------------------------------------------------------------------\n\n")
                json.dump(sFPentities, outfile, sort_keys = True, indent = 4, ensure_ascii = False)
                outfile.write("\n\n-------------------------------------------------------------------")
                outfile.write("\n----------------- TRUE POSITIVES = "+ str(len(sTPentities))+" ----------------")
                outfile.write("\n------------------------------------------------------------------\n\n")
                json.dump(sTPentities, outfile, sort_keys=True, indent=4, ensure_ascii=False)
            outfile.close()
        except json.JSONDecodeError:
            print("Decoding JSON has failed")
        print('TPs, FPs, FNs written successfullly to file!')

    # Following block writes the results of STRONG ANNOTATION to file - for all documents in one file
    try:
        mistakesFile = mistakesFilePath + '0All.txt'  # file to write TP, FP, FN
        with open(mistakesFile, 'a', encoding='utf-8') as outfile:
            outfile.truncate(0)
            outfile.write("Input file: %s\n\n" % datasetFile)
            outfile.write("---------------------------------------------------------------------")
            outfile.write(
                "\n----------------- FALSE NEGATIVES = " + str(len(sFNentitiesAll)) + " ----------------")
            outfile.write("\n--------------------------------------------------------------\n\n")
            json.dump(sFNentitiesAll, outfile, sort_keys=True, indent=4, ensure_ascii=False)
            outfile.write("\n\n------------------------------------------------------------------")
            outfile.write(
                "\n----------------- FALSE POSITIVES = " + str(len(sFPentitiesAll)) + " ----------------")
            outfile.write("\n-------------------------------------------------------------------\n\n")
            json.dump(sFPentitiesAll, outfile, sort_keys=True, indent=4, ensure_ascii=False)
            outfile.write("\n\n-------------------------------------------------------------------")
            outfile.write("\n----------------- TRUE POSITIVES = " + str(len(sTPentitiesAll)) + " ----------------")
            outfile.write("\n------------------------------------------------------------------\n\n")
            json.dump(sTPentitiesAll, outfile, sort_keys=True, indent=4, ensure_ascii=False)
        outfile.close()
    except json.JSONDecodeError:
        print("Decoding JSON has failed")
    print('TPs, FPs, FNs written successfullly to file!')

    # if sentenceNo ==60:
    #     sumOfTPs = perTP + locTP + orgTP
    #     sumOfFPs = perFP + locFP + orgFP
    #     sumOfFNs = perFN + locFN + orgFN
    #     microPrec = sumOfTPs / (sumOfTPs + sumOfFPs)
    #     microRec = sumOfTPs / (sumOfTPs + sumOfFNs)
    #     microF1 = 2 * microPrec * microRec / (microPrec + microRec)
    #     # print("SentNo: " + str(sentenceNo - 1) + "\tMicroF1: " + str(microF1) + "\tmPrec: " + str(
    #     #     microPrec) + "\tmRecall: " + str(microRec))
    #     print("SentNo: " + str(sentenceNo - 1) + "\tSumTPs: " + str(sumOfTPs) + "\tsumFPs: " + str(
    #         sumOfFPs) + "\tsumFNs: " + str(sumOfFNs))

    print("Doc No: " + str(docNo) + "\t\tTP: " + str(struePositives) + "\t\tFP: " + str(sfalsePositives) + "\t\tFN: " + str(sfalseNegatives) + "\n\n")

    Precision = truePositives / (truePositives + falsePositives)
    perPrecision = perTP / (perTP + perFP)
    locPrecision = locTP / (locTP + locFP)
    orgPrecision = orgTP / (orgTP + orgFP)
    Recall = truePositives / (truePositives + falseNegatives)
    perRecall = perTP / (perTP + perFN)
    locRecall = locTP / (locTP + locFN)
    orgRecall = orgTP / (orgTP + orgFN)
    F1 = 2 * Precision * Recall / (Precision + Recall)
    perF1 = 2 * perPrecision*perRecall / (perPrecision + perRecall)
    locF1 = 2 * locPrecision * locRecall / (locPrecision + locRecall)
    orgF1 = 2 * orgPrecision * orgRecall / (orgPrecision + orgRecall)
    sumOfTPs = perTP + locTP + orgTP
    sumOfFPs = perFP + locFP + orgFP
    sumOfFNs = perFN + locFN + orgFN
    microPrec = sumOfTPs / (sumOfTPs + sumOfFPs)
    microRec = sumOfTPs / (sumOfTPs + sumOfFNs)
    microF1 = 2 * microPrec * microRec / (microPrec + microRec)
    sumOfPrecs = perPrecision + locPrecision + orgPrecision
    sumOfRecs = perRecall + locRecall + orgRecall
    macroPrec = sumOfPrecs / 3  # because there are 3 types for classification i.e. per, loc, org
    macroRec = sumOfRecs / 3  # because there are 3 types for classification i.e. per, loc, org
    macroF1 = 2 * macroPrec * macroRec / (macroPrec + macroRec)


    #Following block writes the results of WEAK ANNOTATION to file
    f = open(resultFile, "w+")
    f.write("Input file: %s\n\n" % datasetFile)
    f.write("------------------------------------------------")
    f.write("\n-----------------WEAK ANNOTATION RESULTS----------------")
    f.write("\n------------------------------------------------\n\n")
    f.write("\n----------------- MICRO ----------------\n")
    f.write("Micro Precision: %f\n" % microPrec)
    f.write("Micro Recall: %f\n" % microRec)
    f.write("Micro F1: %f\n" % microF1)
    f.write("\n----------------- MACRO ----------------\n")
    f.write("Macro Precision: %f\n" % macroPrec)
    f.write("Macro Recall: %f\n" % macroRec)
    f.write("Macro F1: %f\n" % macroF1)
    f.write("\n-----------------TOTAL ENTITIES: Recognition, not typing----------------\n")
    f.write("Total Entities in Dataset: %d\n" % noOfDatasetEntities)
    f.write("TruePositives: %d\n" % truePositives)
    f.write("FalsePositives: %d\n" % falsePositives)
    f.write("FalseNegatives: %d\n\n" % falseNegatives)
    f.write("Precision: %f\n" % Precision)
    f.write("Recall: %f\n" % Recall)
    f.write("F1: %f\n" % F1)

    f.write("\n-----------------PERSON ENTITIES----------------\n")
    f.write("No of Person entities in dataset: %d\n" % noOfDatasetPerEntities)
    f.write("TruePositives: %d\n" % perTP)
    f.write("FalsePositives: %d\n" % perFP)
    f.write("FalseNegatives: %d\n\n" % perFN)
    f.write("Precision: %f\n" % perPrecision)
    f.write("Recall: %f\n" % perRecall)
    f.write("F1: %f\n" % perF1)

    f.write("\n-----------------LOCATION ENTITIES----------------\n")
    f.write("No of Location entities in dataset: %d\n" % noOfDatasetLocEntities)
    f.write("TruePositives: %d\n" % locTP)
    f.write("FalsePositives: %d\n" % locFP)
    f.write("FalseNegatives: %d\n\n" % locFN)
    f.write("Precision: %f\n" % locPrecision)
    f.write("Recall: %f\n" % locRecall)
    f.write("F1: %f\n" % locF1)

    f.write("\n-----------------ORGANIZATION ENTITIES----------------\n")
    f.write("No of Organization entities in dataset: %d\n" % noOfDatasetOrgEntities)
    f.write("TruePositives: %d\n" % orgTP)
    f.write("FalsePositives: %d\n" % orgFP)
    f.write("FalseNegatives: %d\n\n" % orgFN)
    f.write("Precision: %f\n" % orgPrecision)
    f.write("Recall: %f\n" % orgRecall)
    f.write("F1: %f\n" % orgF1)


    # for STRONG annotation
    Precision = struePositives / (struePositives + sfalsePositives)
    perPrecision = sperTP / (sperTP + sperFP)
    locPrecision = slocTP / (slocTP + slocFP)
    orgPrecision = sorgTP / (sorgTP + sorgFP)
    Recall = struePositives / (struePositives + sfalseNegatives)
    perRecall = sperTP / (sperTP + sperFN)
    locRecall = slocTP / (slocTP + slocFN)
    orgRecall = sorgTP / (sorgTP + sorgFN)
    F1 = 2 * Precision * Recall / (Precision + Recall)
    perF1 = 2 * perPrecision*perRecall / (perPrecision + perRecall)
    locF1 = 2 * locPrecision * locRecall / (locPrecision + locRecall)
    orgF1 = 2 * orgPrecision * orgRecall / (orgPrecision + orgRecall)
    sumOfTPs = sperTP + slocTP + sorgTP
    sumOfFPs = sperFP + slocFP + sorgFP
    sumOfFNs = sperFN + slocFN + sorgFN
    microPrec = sumOfTPs / (sumOfTPs + sumOfFPs)
    microRec = sumOfTPs / (sumOfTPs + sumOfFNs)
    microF1 = 2 * microPrec * microRec / (microPrec + microRec)
    sumOfPrecs = perPrecision + locPrecision + orgPrecision
    sumOfRecs = perRecall + locRecall + orgRecall
    macroPrec = sumOfPrecs / 3  # because there are 3 types for classification i.e. per, loc, org
    macroRec = sumOfRecs / 3  # because there are 3 types for classification i.e. per, loc, org
    macroF1 = 2 * macroPrec * macroRec / (macroPrec + macroRec)

    #Following block writes the results of STRONG ANNOTATION to file
    f.write("\n\n------------------------------------------------")
    f.write("\n-----------------STRONG ANNOTATION RESULTS----------------")
    f.write("\n------------------------------------------------\n\n")
    f.write("\n----------------- MICRO ----------------\n")
    f.write("Micro Precision: %f\n" % microPrec)
    f.write("Micro Recall: %f\n" % microRec)
    f.write("Micro F1: %f\n" % microF1)
    f.write("\n----------------- MACRO ----------------\n")
    f.write("Macro Precision: %f\n" % macroPrec)
    f.write("Macro Recall: %f\n" % macroRec)
    f.write("Macro F1: %f\n" % macroF1)
    f.write("\n-----------------TOTAL ENTITIES: Recognition, not typing----------------\n")
    f.write("Total Entities in Dataset: %d\n" % noOfDatasetEntities)
    f.write("TruePositives: %d\n" % struePositives)
    f.write("FalsePositives: %d\n" % sfalsePositives)
    f.write("FalseNegatives: %d\n\n" % sfalseNegatives)
    f.write("Precision: %f\n" % Precision)
    f.write("Recall: %f\n" % Recall)
    f.write("F1: %f\n" % F1)

    f.write("\n-----------------PERSON ENTITIES----------------\n")
    f.write("No of Person entities in dataset: %d\n" % noOfDatasetPerEntities)
    f.write("TruePositives: %d\n" % sperTP)
    f.write("FalsePositives: %d\n" % sperFP)
    f.write("FalseNegatives: %d\n\n" % sperFN)
    f.write("Precision: %f\n" % perPrecision)
    f.write("Recall: %f\n" % perRecall)
    f.write("F1: %f\n" % perF1)

    f.write("\n-----------------LOCATION ENTITIES----------------\n")
    f.write("No of Location entities in dataset: %d\n" % noOfDatasetLocEntities)
    f.write("TruePositives: %d\n" % slocTP)
    f.write("FalsePositives: %d\n" % slocFP)
    f.write("FalseNegatives: %d\n\n" % slocFN)
    f.write("Precision: %f\n" % locPrecision)
    f.write("Recall: %f\n" % locRecall)
    f.write("F1: %f\n" % locF1)

    f.write("\n-----------------ORGANIZATION ENTITIES----------------\n")
    f.write("No of Organization entities in dataset: %d\n" % noOfDatasetOrgEntities)
    f.write("TruePositives: %d\n" % sorgTP)
    f.write("FalsePositives: %d\n" % sorgFP)
    f.write("FalseNegatives: %d\n\n" % sorgFN)
    f.write("Precision: %f\n" % orgPrecision)
    f.write("Recall: %f\n" % orgRecall)
    f.write("F1: %f\n" % orgF1)


def evaluateOKE(trainOrEval,system):
    # trainOrEval=1=train, trainOrEval=2=eval
    # system: the system for evaluation, 0=HmaraNER, 1 for Illinois, 2 for Stanford
    # pre processing
    if trainOrEval == 1:  # train
        sizeOfDataset = 60
        nifDatasetFile = "inputOKE\oke17task1TrainingCorrected.xml.ttl"
        # nifDatasetFile = "inputOKE\oke17task1Training.xml.ttl"
        txtFilePath = 'outputOKE\\train\okeEntities\\doc'
        if system == 0:  # HmaraNER
            hmariFilePath = 'outputOKE\\train\hmariEntities\\allEntities\output' #HmaraNER Train File path
            mistakesFilePath = 'outputOKE\\train\hmariEntities\mistakes\output' #HmaraNER Train File path
            resultFile = 'results\hmarayResultsOnTrainDataOKE.txt' #HmaraNER Train result file
        elif system == 1:  # Illinois
            hmariFilePath = 'outputOKE\\train\illinois\miscRemoved\\allEntities\outputIllinois' #Illinois NER Train File path
            mistakesFilePath = 'outputOKE\\train\illinois\miscRemoved\mistakes\outputIllinois' #Illinois NER Train File path
            resultFile = 'results\illinoisResultsOnTrainDataOKE.txt' #Illinois NER Train result file
        elif system == 2:  # Stanford
            hmariFilePath = 'outputOKE\\train\\ner\cleaned\miscRemoved\\allEntities\outputNER' #Stanford NER Train File path
            mistakesFilePath = 'outputOKE\\train\\ner\cleaned\miscRemoved\mistakes\outputNER' #Stanford NER Train File path
            resultFile = 'results\stanfordResultsOnTrainDataOKE.txt' #Stanford NER Train result file
        elif system==3: #spotlight
            hmariFilePath = 'outputOKE\\train\spotlight\outputSpotlight' #Spotlight Train File path
            resultFile = 'results\spotlightResultsOnTrainDataOKE.txt' #Spotlight NER Train result file


    elif trainOrEval == 2:  # eval
        sizeOfDataset = 58

        nifDatasetFile = "inputOKE\oke17task1EvalCorrected.xml.ttl" # for evaluation against corrected oke dataset
        txtFilePath = 'outputOKE\\eval\okeEntities\\doc' # for evaluation against corrected oke dataset
        resultFile = 'results' # for evaluation against corrected oke dataset
        mistakesFolder = '\mistakes' # for evaluation against corrected oke dataset

        # nifDatasetFile = "inputOKE\oke17task1Eval.xml.ttl" # for evaluation against original (un-corrected) oke dataset
        # txtFilePath = 'outputOKE\\eval\okeEntitiesOriginal\\doc' # for evaluation against original (un-corrected) oke dataset
        # resultFile = 'results\\resultsOnOriginalOKE' # for evaluation against original (un-corrected) oke dataset
        # mistakesFolder = '\mistakesAgainstOriginalOKE'# for evaluation against original (un-corrected) oke dataset

        if system == 0:  # HmaraNER
            hmariFilePath = 'outputOKE\eval\hmariEntities\\allEntities\output' #HmaraNER Eval File path
            mistakesFilePath = 'outputOKE\eval\hmariEntities'+mistakesFolder+'\output' #HmaraNER Eval File path
            resultFile = resultFile+'\hmarayResultsOnEvalDataOKE.txt' #HmaraNER Eval result file
        elif system == 1:  # Illinois
            hmariFilePath = 'outputOKE\eval\illinois\miscRemoved\\allEntities\outputIllinois'  # Illinois NER Eval File path
            mistakesFilePath = 'outputOKE\eval\illinois\miscRemoved'+mistakesFolder+'\outputIllinois'  # Illinois NER Eval File path
            resultFile = resultFile+'\illinoisResultsOnEvalDataOKE.txt'  # Illinois NER Eval result file
        elif system == 2:  # Stanford
            hmariFilePath = 'outputOKE\eval\\ner\cleaned\miscRemoved\\allEntities\outputNER'  # Stanford NER Eval File path
            mistakesFilePath = 'outputOKE\eval\\ner\cleaned\miscRemoved'+mistakesFolder+'\outputNER'  # Stanford NER Eval File path
            resultFile = resultFile+'\stanfordResultsOnEvalDataOKE.txt'  # Stanford NER Eval result file
        elif system==3: #spotlight
            hmariFilePath = 'outputOKE\eval\spotlight\outputSpotlight'  # Spotlight Eval File path
            resultFile = resultFile+'\spotlightResultsOnEvalDataOKE.txt'  # Spotlight NER Eval result file
        elif system == 4:  # FOX
            hmariFilePath = 'outputOKE\eval\\fox\\entities\\'  # FOX Eval File path
            mistakesFilePath = 'outputOKE\eval\\fox'+mistakesFolder+'\\'  # FOX Eval File path
            resultFile = resultFile+'\\foxResultsOnEvalDataOKE.txt'  # FOX Eval result file
        elif system == 5:  # ADEL
            hmariFilePath = 'outputOKE\eval\\adel\\entities\\'  # ADEL Eval File path
            mistakesFilePath = 'outputOKE\eval\\adel'+mistakesFolder+'\\'  # ADEL Eval File path
            resultFile = resultFile+'\\adelResultsOnEvalDataOKE.txt'  # ADEL Eval result file

    print("Output file: "+resultFile)
    print("Output Mistakes file path: " + mistakesFilePath)
    # convertOKEdatasetFromNIFtoTxt(sizeOfDataset,nifDatasetFile,txtFilePath) #THIS IS A ONE TIME TASK

    #evaluation script starts here
    truePositives = 0;    perTP = 0;    locTP = 0;    orgTP = 0
    falsePositives = 0;    perFP = 0;    locFP = 0;    orgFP = 0
    falseNegatives = 0;    perFN = 0;    locFN = 0;    orgFN = 0

    #for STRONG annotation
    struePositives = 0;    sperTP = 0;    slocTP = 0;    sorgTP = 0
    sfalsePositives = 0;    sperFP = 0;    slocFP = 0;    sorgFP = 0
    sfalseNegatives = 0;    sperFN = 0;    slocFN = 0;    sorgFN = 0
    sTPentitiesAll = [];    sFPentitiesAll = [];    sFNentitiesAll = []

    noOfDatasetEntities = 0 #the total no of entities in training/evaluation dataset
    noOfDatasetPerEntities = 0 #the total no of PERSON entities in training/evaluation dataset
    noOfDatasetLocEntities = 0 #the total no of LOCATION entities in training/evaluation dataset
    noOfDatasetOrgEntities = 0 #the total no of ORGANIZATION entities in training/evaluation dataset


    for sentenceNo in range(0, sizeOfDataset): #for each sentence/file/text
        sTPentities = [];        sFPentities = [];        sFNentities = []
        # if sentenceNo>0:
        #     sumOfTPs = perTP + locTP + orgTP
        #     sumOfFPs = perFP + locFP + orgFP
        #     sumOfFNs = perFN + locFN + orgFN
        #     microPrec = sumOfTPs / (sumOfTPs + sumOfFPs)
        #     microRec = sumOfTPs / (sumOfTPs + sumOfFNs)
        #     microF1 = 2 * microPrec * microRec / (microPrec + microRec)
        #     # if sentenceNo>9: break
        #     # if sentenceNo==0 : continue
        #     # print("SentNo: " + str(sentenceNo - 1) + "\tMicroF1: " + str(microF1) + "\tmPrec: " + str(
        #     #     microPrec) + "\tmRecall: " + str(microRec))
        #     print("SentNo: " + str(sentenceNo - 1) + "\tSumTPs: " + str(sumOfTPs) + "\tsumFPs: " + str(
        #         sumOfFPs) + "\tsumFNs: " + str(sumOfFNs))
        print("SentNo: " + str(sentenceNo))
        datasetFile = txtFilePath+str(sentenceNo)+'.txt'
        hmariFile = hmariFilePath+str(sentenceNo)+'.txt' #HmaraNER output file
        with open(datasetFile, 'r', encoding='utf-8', errors='ignore') as outfile:
            datasetEntities = json.load(outfile)  # dataset file of one sentence/doc/text loaded as json dictionary
        with open(hmariFile, 'r', encoding='utf-8', errors='ignore') as outfile2:
            hmariEntities = json.load(outfile2)  # hmaraNER output file loaded as json dictionary
        hmariEntityNo = 0
        datasetEntityNo = 0
        flag = False
        flag2 = False
        nextDatasetEntity = True
        noOfDatasetEntities += len(datasetEntities)
        while datasetEntityNo<len(datasetEntities) or hmariEntityNo<len(hmariEntities): #for each entity in file/sentence/text
            if hmariEntityNo<len(hmariEntities):
                if system==0:
                    hmariEntity = hmariEntities[hmariEntityNo][0] #For HmaraNER
                else:
                    hmariEntity = hmariEntities[hmariEntityNo] #For Illinois/Stanford NER
                    if system == 1:  # Illinois
                        hmariEntity['ner'] = hmariEntity['label']  # 4: for Illinois
                        hmariEntity['text'] = hmariEntity['tokens']  # 4: for Illinois
                    if system == 2:  # Stanford
                        if hmariEntity['ner'] == 'CITY' or hmariEntity['ner'] == 'STATE_OR_PROVINCE' or hmariEntity['ner'] == 'COUNTRY':
                            hmariEntity['ner'] = 'LOCATION'

            # if its not per/loc/org entity e.g misc in case of illinois, date title etc in case of stanford
            while flag == False and (not (str.startswith(hmariEntity['ner'], 'PER')) and not (str.startswith(hmariEntity['ner'], 'LOC')) and not (str.startswith(hmariEntity['ner'], 'ORG'))):
                hmariEntityNo += 1
                if hmariEntityNo < len(hmariEntities):
                    if system == 0:  # HmaraNER
                        hmariEntity = hmariEntities[hmariEntityNo][0]  # for HmaraNER
                    elif system == 1 or system == 2:  # Illinois or Stanford
                        hmariEntity = hmariEntities[hmariEntityNo]  # 3: for Illinois or Stanford
                        if system == 1:  # Illinois
                            hmariEntity['ner'] = hmariEntity['label']  # 4: for Illinois
                            hmariEntity['text'] = hmariEntity['tokens']  # 4: for Illinois
                        if system == 2:  # Stanford
                            if hmariEntity['ner'] == 'CITY' or hmariEntity['ner'] == 'STATE_OR_PROVINCE' or \
                                            hmariEntity['ner'] == 'COUNTRY':
                                hmariEntity['ner'] = 'LOCATION'
                elif hmariEntityNo == len(hmariEntities):  # if its the last entity and is not ppo type
                    flag = True
                    continue

            # if not (str.startswith(hmariEntity['ner'], 'PER') or str.startswith(hmariEntity['ner'],'LOC') or str.startswith(hmariEntity['ner'], 'ORG')):  # if hmariEntity is not PPO type
            #     hmariEntityNo += 1
            #     continue

            if datasetEntityNo<len(datasetEntities):
                datasetEntity = datasetEntities[datasetEntityNo]
                if nextDatasetEntity==True:
                    if str.startswith(datasetEntity['ner'],'PER'):
                        noOfDatasetPerEntities += 1
                    elif str.startswith(datasetEntity['ner'],'LOC'):
                        noOfDatasetLocEntities += 1
                    elif str.startswith(datasetEntity['ner'],'ORG'):
                        noOfDatasetOrgEntities += 1
                    nextDatasetEntity = False

            if datasetEntityNo >= (len(datasetEntities) - 1) and hmariEntityNo >= (len(hmariEntities) - 1):
                flag2 = True
            offsetRangeDataset = range(int(datasetEntity['characterOffsetBegin']), int(datasetEntity['characterOffsetEnd']))
            offsetRangeHmari = range(hmariEntity['characterOffsetBegin'], hmariEntity['characterOffsetEnd'])
            offsetRangesOverlap = [i for i in offsetRangeDataset if i in offsetRangeHmari]

            if system==1: # For Illinois NER
                hmariEntity['ner'] = hmariEntity['label']
            if system == 2:  # Stanford
                if hmariEntity['ner'] == 'LOC' or hmariEntity['ner'] == 'CITY' or hmariEntity['ner'] == 'STATE_OR_PROVINCE' or hmariEntity['ner'] == 'COUNTRY':
                    hmariEntity['ner'] = 'LOCATION'

            # FOR WEAK ANNOTATION
            if len(offsetRangesOverlap) > 0 and datasetEntityNo<len(datasetEntities) and hmariEntityNo<len(hmariEntities):
                truePositives += 1
                if str.startswith(hmariEntity['ner'],'PER') and str.startswith(datasetEntity['ner'],'PER'):
                    perTP += 1
                elif str.startswith(hmariEntity['ner'],'PER') and not(str.startswith(datasetEntity['ner'],'PER')):
                    perFP += 1
                elif not(str.startswith(hmariEntity['ner'],'PER')) and str.startswith(datasetEntity['ner'],'PER'):
                    perFN += 1
                if str.startswith(hmariEntity['ner'],'LOC') and str.startswith(datasetEntity['ner'],'LOC'):
                    locTP += 1
                elif str.startswith(hmariEntity['ner'],'LOC') and not(str.startswith(datasetEntity['ner'],'LOC')):
                    locFP += 1
                elif not(str.startswith(hmariEntity['ner'],'LOC')) and str.startswith(datasetEntity['ner'],'LOC'):
                    locFN += 1
                if str.startswith(hmariEntity['ner'],'ORG') and str.startswith(datasetEntity['ner'],'ORG'):
                    orgTP += 1
                elif str.startswith(hmariEntity['ner'],'ORG') and not(str.startswith(datasetEntity['ner'],'ORG')):
                    orgFP += 1
                elif not(str.startswith(hmariEntity['ner'],'ORG')) and str.startswith(datasetEntity['ner'],'ORG'):
                    orgFN += 1
                hmariEntityNo += 1
                datasetEntityNo += 1
                nextDatasetEntity = True
            else:
                if (hmariEntity['characterOffsetBegin'] > int(datasetEntity['characterOffsetBegin']) and datasetEntityNo<len(datasetEntities)) or (hmariEntityNo>=len(hmariEntities)):
                    falseNegatives += 1
                    if str.startswith(datasetEntity['ner'], 'PER'):
                        perFN += 1
                    elif str.startswith(datasetEntity['ner'],'LOC'):
                        locFN += 1
                    elif str.startswith(datasetEntity['ner'],'ORG'):
                        orgFN += 1
                    datasetEntityNo += 1
                    nextDatasetEntity = True
                else:
                    falsePositives += 1
                    if str.startswith(hmariEntity['ner'],'PER'):
                        perFP += 1
                    elif str.startswith(hmariEntity['ner'],'LOC'):
                        locFP += 1
                    elif str.startswith(hmariEntity['ner'],'ORG'):
                        orgFP += 1
                    hmariEntityNo += 1

            # FOR STRONG ANNOTATION
            if int(datasetEntity['characterOffsetBegin'])==hmariEntity['characterOffsetBegin'] and int(datasetEntity['characterOffsetEnd'])==hmariEntity['characterOffsetEnd'] and datasetEntityNo-1<len(datasetEntities) and hmariEntityNo-1<len(hmariEntities):
                #CASE 1: Exact boundary match
                struePositives += 1
                sTPentities.append(hmariEntity)
                if str.startswith(hmariEntity['ner'],'PER') and str.startswith(datasetEntity['ner'],'PER'):
                    sperTP += 1
                elif str.startswith(hmariEntity['ner'],'PER') and not(str.startswith(datasetEntity['ner'],'PER')):
                    sperFP += 1
                elif not(str.startswith(hmariEntity['ner'],'PER')) and str.startswith(datasetEntity['ner'],'PER'):
                    sperFN += 1
                if str.startswith(hmariEntity['ner'],'LOC') and str.startswith(datasetEntity['ner'],'LOC'):
                    slocTP += 1
                elif str.startswith(hmariEntity['ner'],'LOC') and not(str.startswith(datasetEntity['ner'],'LOC')):
                    slocFP += 1
                elif not(str.startswith(hmariEntity['ner'],'LOC')) and str.startswith(datasetEntity['ner'],'LOC'):
                    slocFN += 1
                if str.startswith(hmariEntity['ner'],'ORG') and str.startswith(datasetEntity['ner'],'ORG'):
                    sorgTP += 1
                elif str.startswith(hmariEntity['ner'],'ORG') and not(str.startswith(datasetEntity['ner'],'ORG')):
                    sorgFP += 1
                elif not(str.startswith(hmariEntity['ner'],'ORG')) and str.startswith(datasetEntity['ner'],'ORG'):
                    sorgFN += 1
            elif len(offsetRangesOverlap) > 0:#CASE 2: Partial overlap
                sfalseNegatives += 1
                sfalsePositives += 1
                sFPentities.append(hmariEntity)
                sFNentities.append(datasetEntity)
                if str.startswith(datasetEntity['ner'], 'PER'):
                    sperFN += 1
                elif str.startswith(datasetEntity['ner'], 'LOC'):
                    slocFN += 1
                elif str.startswith(datasetEntity['ner'], 'ORG'):
                    sorgFN += 1
                if str.startswith(hmariEntity['ner'], 'PER'):
                    sperFP += 1
                elif str.startswith(hmariEntity['ner'], 'LOC'):
                    slocFP += 1
                elif str.startswith(hmariEntity['ner'], 'ORG'):
                    sorgFP += 1
            else:#CASE 3: No overlap
                if (hmariEntity['characterOffsetBegin'] > int(datasetEntity['characterOffsetBegin']) and datasetEntityNo<len(datasetEntities)) or (hmariEntityNo>=len(hmariEntities)):
                    sfalseNegatives += 1
                    sFNentities.append(datasetEntity)
                    if str.startswith(datasetEntity['ner'],'PER'):
                        sperFN += 1
                    elif str.startswith(datasetEntity['ner'],'LOC'):
                        slocFN += 1
                    elif str.startswith(datasetEntity['ner'],'ORG'):
                        sorgFN += 1
                else:
                    sfalsePositives += 1
                    sFPentities.append(datasetEntity)
                    if str.startswith(hmariEntity['ner'],'PER'):
                        sperFP += 1
                    elif str.startswith(hmariEntity['ner'],'LOC'):
                        slocFP += 1
                    elif str.startswith(hmariEntity['ner'],'ORG'):
                        sorgFP += 1

            if flag2==True:
                break

        sTPentitiesAll = sTPentitiesAll + sTPentities
        sFPentitiesAll = sFPentitiesAll + sFPentities
        sFNentitiesAll = sFNentitiesAll + sFNentities

        # Following block writes the results of STRONG ANNOTATION to file - for each document in separate file
        try:
            mistakesFile = mistakesFilePath + str(sentenceNo) + '.txt'  # file to write TP, FP, FN
            with open(mistakesFile, 'a', encoding='utf-8') as outfile:
                outfile.truncate(0)
                outfile.write("Input file: %s\n\n" % datasetFile)
                outfile.write("---------------------------------------------------------------------")
                outfile.write(
                    "\n----------------- FALSE NEGATIVES = " + str(len(sFNentities)) + " ----------------")
                outfile.write("\n--------------------------------------------------------------\n\n")
                json.dump(sFNentities, outfile, sort_keys=True, indent=4, ensure_ascii=False)
                outfile.write("\n\n------------------------------------------------------------------")
                outfile.write(
                    "\n----------------- FALSE POSITIVES = " + str(len(sFPentities)) + " ----------------")
                outfile.write("\n-------------------------------------------------------------------\n\n")
                json.dump(sFPentities, outfile, sort_keys=True, indent=4, ensure_ascii=False)
                outfile.write("\n\n-------------------------------------------------------------------")
                outfile.write("\n----------------- TRUE POSITIVES = " + str(len(sTPentities)) + " ----------------")
                outfile.write("\n------------------------------------------------------------------\n\n")
                json.dump(sTPentities, outfile, sort_keys=True, indent=4, ensure_ascii=False)
            outfile.close()
        except json.JSONDecodeError:
            print("Decoding JSON has failed")
        print('TPs, FPs, FNs written successfullly to file!')

    # Following block writes the results of STRONG ANNOTATION to file - for all documents in one file
    try:
        mistakesFile = mistakesFilePath + '0All.txt'  # file to write TP, FP, FN
        with open(mistakesFile, 'a', encoding='utf-8') as outfile:
            outfile.truncate(0)
            outfile.write("Input file: %s\n\n" % datasetFile)
            outfile.write("---------------------------------------------------------------------")
            outfile.write(
                "\n----------------- FALSE NEGATIVES = " + str(len(sFNentitiesAll)) + " ----------------")
            outfile.write("\n--------------------------------------------------------------\n\n")
            json.dump(sFNentitiesAll, outfile, sort_keys=True, indent=4, ensure_ascii=False)
            outfile.write("\n\n------------------------------------------------------------------")
            outfile.write(
                "\n----------------- FALSE POSITIVES = " + str(len(sFPentitiesAll)) + " ----------------")
            outfile.write("\n-------------------------------------------------------------------\n\n")
            json.dump(sFPentitiesAll, outfile, sort_keys=True, indent=4, ensure_ascii=False)
            outfile.write("\n\n-------------------------------------------------------------------")
            outfile.write("\n----------------- TRUE POSITIVES = " + str(len(sTPentitiesAll)) + " ----------------")
            outfile.write("\n------------------------------------------------------------------\n\n")
            json.dump(sTPentitiesAll, outfile, sort_keys=True, indent=4, ensure_ascii=False)
        outfile.close()
    except json.JSONDecodeError:
        print("Decoding JSON has failed")
    print('TPs, FPs, FNs written successfullly to file!')

    # if sentenceNo ==60:
    #     sumOfTPs = perTP + locTP + orgTP
    #     sumOfFPs = perFP + locFP + orgFP
    #     sumOfFNs = perFN + locFN + orgFN
    #     microPrec = sumOfTPs / (sumOfTPs + sumOfFPs)
    #     microRec = sumOfTPs / (sumOfTPs + sumOfFNs)
    #     microF1 = 2 * microPrec * microRec / (microPrec + microRec)
    #     # print("SentNo: " + str(sentenceNo - 1) + "\tMicroF1: " + str(microF1) + "\tmPrec: " + str(
    #     #     microPrec) + "\tmRecall: " + str(microRec))
    #     print("SentNo: " + str(sentenceNo - 1) + "\tSumTPs: " + str(sumOfTPs) + "\tsumFPs: " + str(
    #         sumOfFPs) + "\tsumFNs: " + str(sumOfFNs))

    Precision = truePositives / (truePositives + falsePositives)
    perPrecision = perTP / (perTP + perFP)
    locPrecision = locTP / (locTP + locFP)
    orgPrecision = orgTP / (orgTP + orgFP)
    Recall = truePositives / (truePositives + falseNegatives)
    perRecall = perTP / (perTP + perFN)
    locRecall = locTP / (locTP + locFN)
    orgRecall = orgTP / (orgTP + orgFN)
    F1 = 2 * Precision * Recall / (Precision + Recall)
    perF1 = 2 * perPrecision*perRecall / (perPrecision + perRecall)
    locF1 = 2 * locPrecision * locRecall / (locPrecision + locRecall)
    orgF1 = 2 * orgPrecision * orgRecall / (orgPrecision + orgRecall)
    sumOfTPs = perTP + locTP + orgTP
    sumOfFPs = perFP + locFP + orgFP
    sumOfFNs = perFN + locFN + orgFN
    microPrec = sumOfTPs / (sumOfTPs + sumOfFPs)
    microRec = sumOfTPs / (sumOfTPs + sumOfFNs)
    microF1 = 2 * microPrec * microRec / (microPrec + microRec)
    sumOfPrecs = perPrecision + locPrecision + orgPrecision
    sumOfRecs = perRecall + locRecall + orgRecall
    macroPrec = sumOfPrecs / 3 #because there are 3 types for classification i.e. per, loc, org
    macroRec = sumOfRecs / 3 #because there are 3 types for classification i.e. per, loc, org
    macroF1 = 2 * macroPrec * macroRec / (macroPrec + macroRec)


    #Following block writes the results of weak annotation to file
    f = open(resultFile, "w+")
    f.write("Input file: %s\n\n" % nifDatasetFile)
    f.write("------------------------------------------------")
    f.write("\n-----------------WEAK ANNOTATION RESULTS----------------")
    f.write("\n------------------------------------------------\n\n")
    f.write("\n----------------- MICRO ----------------\n")
    f.write("Micro Precision: %f\n" % microPrec)
    f.write("Micro Recall: %f\n" % microRec)
    f.write("Micro F1: %f\n" % microF1)
    f.write("\n----------------- MACRO ----------------\n")
    f.write("Macro Precision: %f\n" % macroPrec)
    f.write("Macro Recall: %f\n" % macroRec)
    f.write("Macro F1: %f\n" % macroF1)
    f.write("\n-----------------TOTAL ENTITIES: Recognition, not typing----------------\n")
    f.write("Total Entities in Dataset: %d\n" % noOfDatasetEntities)
    f.write("TruePositives: %d\n" % truePositives)
    f.write("FalsePositives: %d\n" % falsePositives)
    f.write("FalseNegatives: %d\n\n" % falseNegatives)
    f.write("Precision: %f\n" % Precision)
    f.write("Recall: %f\n" % Recall)
    f.write("F1: %f\n" % F1)

    f.write("\n-----------------PERSON ENTITIES----------------\n")
    f.write("No of Person entities in dataset: %d\n" % noOfDatasetPerEntities)
    f.write("TruePositives: %d\n" % perTP)
    f.write("FalsePositives: %d\n" % perFP)
    f.write("FalseNegatives: %d\n\n" % perFN)
    f.write("Precision: %f\n" % perPrecision)
    f.write("Recall: %f\n" % perRecall)
    f.write("F1: %f\n" % perF1)

    f.write("\n-----------------LOCATION ENTITIES----------------\n")
    f.write("No of Location entities in dataset: %d\n" % noOfDatasetLocEntities)
    f.write("TruePositives: %d\n" % locTP)
    f.write("FalsePositives: %d\n" % locFP)
    f.write("FalseNegatives: %d\n\n" % locFN)
    f.write("Precision: %f\n" % locPrecision)
    f.write("Recall: %f\n" % locRecall)
    f.write("F1: %f\n" % locF1)

    f.write("\n-----------------ORGANIZATION ENTITIES----------------\n")
    f.write("No of Organization entities in dataset: %d\n" % noOfDatasetOrgEntities)
    f.write("TruePositives: %d\n" % orgTP)
    f.write("FalsePositives: %d\n" % orgFP)
    f.write("FalseNegatives: %d\n\n" % orgFN)
    f.write("Precision: %f\n" % orgPrecision)
    f.write("Recall: %f\n" % orgRecall)
    f.write("F1: %f\n" % orgF1)


    # for STRONG annotation
    Precision = struePositives / (struePositives + sfalsePositives)
    perPrecision = sperTP / (sperTP + sperFP)
    locPrecision = slocTP / (slocTP + slocFP)
    orgPrecision = sorgTP / (sorgTP + sorgFP)
    Recall = struePositives / (struePositives + sfalseNegatives)
    perRecall = sperTP / (sperTP + sperFN)
    locRecall = slocTP / (slocTP + slocFN)
    orgRecall = sorgTP / (sorgTP + sorgFN)
    F1 = 2 * Precision * Recall / (Precision + Recall)
    perF1 = 2 * perPrecision*perRecall / (perPrecision + perRecall)
    locF1 = 2 * locPrecision * locRecall / (locPrecision + locRecall)
    orgF1 = 2 * orgPrecision * orgRecall / (orgPrecision + orgRecall)
    sumOfTPs = sperTP + slocTP + sorgTP
    sumOfFPs = sperFP + slocFP + sorgFP
    sumOfFNs = sperFN + slocFN + sorgFN
    microPrec = sumOfTPs / (sumOfTPs + sumOfFPs)
    microRec = sumOfTPs / (sumOfTPs + sumOfFNs)
    microF1 = 2 * microPrec * microRec / (microPrec + microRec)
    sumOfPrecs = perPrecision + locPrecision + orgPrecision
    sumOfRecs = perRecall + locRecall + orgRecall
    macroPrec = sumOfPrecs / 3  # because there are 3 types for classification i.e. per, loc, org
    macroRec = sumOfRecs / 3  # because there are 3 types for classification i.e. per, loc, org
    macroF1 = 2 * macroPrec * macroRec / (macroPrec + macroRec)

    #Following block writes the results of STRONG ANNOTATION to file
    f.write("\n\n------------------------------------------------")
    f.write("\n-----------------STRONG ANNOTATION RESULTS----------------")
    f.write("\n------------------------------------------------\n\n")
    f.write("\n----------------- MICRO ----------------\n")
    f.write("Micro Precision: %f\n" % microPrec)
    f.write("Micro Recall: %f\n" % microRec)
    f.write("Micro F1: %f\n" % microF1)
    f.write("\n----------------- MACRO ----------------\n")
    f.write("Macro Precision: %f\n" % macroPrec)
    f.write("Macro Recall: %f\n" % macroRec)
    f.write("Macro F1: %f\n" % macroF1)
    f.write("\n-----------------TOTAL ENTITIES: Recognition, not typing----------------\n")
    f.write("Total Entities in Dataset: %d\n" % noOfDatasetEntities)
    f.write("TruePositives: %d\n" % struePositives)
    f.write("FalsePositives: %d\n" % sfalsePositives)
    f.write("FalseNegatives: %d\n\n" % sfalseNegatives)
    f.write("Precision: %f\n" % Precision)
    f.write("Recall: %f\n" % Recall)
    f.write("F1: %f\n" % F1)

    f.write("\n-----------------PERSON ENTITIES----------------\n")
    f.write("No of Person entities in dataset: %d\n" % noOfDatasetPerEntities)
    f.write("TruePositives: %d\n" % sperTP)
    f.write("FalsePositives: %d\n" % sperFP)
    f.write("FalseNegatives: %d\n\n" % sperFN)
    f.write("Precision: %f\n" % perPrecision)
    f.write("Recall: %f\n" % perRecall)
    f.write("F1: %f\n" % perF1)

    f.write("\n-----------------LOCATION ENTITIES----------------\n")
    f.write("No of Location entities in dataset: %d\n" % noOfDatasetLocEntities)
    f.write("TruePositives: %d\n" % slocTP)
    f.write("FalsePositives: %d\n" % slocFP)
    f.write("FalseNegatives: %d\n\n" % slocFN)
    f.write("Precision: %f\n" % locPrecision)
    f.write("Recall: %f\n" % locRecall)
    f.write("F1: %f\n" % locF1)

    f.write("\n-----------------ORGANIZATION ENTITIES----------------\n")
    f.write("No of Organization entities in dataset: %d\n" % noOfDatasetOrgEntities)
    f.write("TruePositives: %d\n" % sorgTP)
    f.write("FalsePositives: %d\n" % sorgFP)
    f.write("FalseNegatives: %d\n\n" % sorgFN)
    f.write("Precision: %f\n" % orgPrecision)
    f.write("Recall: %f\n" % orgRecall)
    f.write("F1: %f\n" % orgF1)

def evaluateGenderOKE(): #this evaluates gender annotations of system=CustNER on dataset=okeEvalCorrected
    sizeOfDataset = 58 # oke eval corrected
    nifDatasetFile = "inputOKE\oke17task1EvalCorrected.xml.ttl" # for evaluation against corrected oke dataset
    txtFilePath = 'outputOKE\\eval\okeEntitiesCorrected\withGender\\doc' # for evaluation against corrected oke dataset

    convertOKEdatasetFromNIFtoTxt(sizeOfDataset,nifDatasetFile,txtFilePath) #THIS IS A ONE TIME TASK # This is done

    # resultFile = 'resultsNER\\resultsOnCorrectedOKE\gender'  # for evaluation against corrected oke dataset
    # mistakesFolder = '\mistakesGender'  # for evaluation against corrected oke dataset
    # hmariFilePath = 'outputOKE\eval\hmariEntities\\allEntitiesWithGender\output'  # HmaraNER Eval File path
    # mistakesFilePath = 'outputOKE\eval\hmariEntities' + mistakesFolder + '\output'  # HmaraNER Eval File path
    # resultFile = resultFile + '\custNERResultsOnEvalDataOKE.txt'  # HmaraNER Eval result file

    # print("Output file: " + resultFile)
    # print("Output Mistakes file path: " + mistakesFilePath)
    #
    # #evaluation script starts here
    # truePositives = 0;    maleTP = 0;    femaleTP = 0;
    # falsePositives = 0;    maleFP = 0;    femaleFP = 0;
    # falseNegatives = 0;    maleFN = 0;    femaleFN = 0;
    #
    # noOfDatasetEntities = 0 #the total no of entities in evaluation dataset
    # noOfDatasetMaleEntities = 0 #the total no of MALE entities in evaluation dataset
    # noOfDatasetFemaleEntities = 0 #the total no of FEMALE entities in evaluation dataset
    #
    # for sentenceNo in range(0, sizeOfDataset): #for each sentence/file/text
    #     sTPentities = [];        sFPentities = [];        sFNentities = []
    #     print("SentNo: " + str(sentenceNo))
    #     datasetFile = txtFilePath+str(sentenceNo)+'.txt'
    #     hmariFile = hmariFilePath+str(sentenceNo)+'.txt' #custNER output file
    #     with open(datasetFile, 'r', encoding='utf-8', errors='ignore') as outfile:
    #         datasetEntities = json.load(outfile)  # dataset file of one sentence/doc/text loaded as json dictionary
    #     with open(hmariFile, 'r', encoding='utf-8', errors='ignore') as outfile2:
    #         hmariEntities = json.load(outfile2)  # custNER output file loaded as json dictionary
    #     hmariEntityNo = 0
    #     datasetEntityNo = 0
    #     flag = False
    #     flag2 = False
    #     nextDatasetEntity = True
    #     noOfDatasetEntities += len(datasetEntities)
    #     while datasetEntityNo<len(datasetEntities) or hmariEntityNo<len(hmariEntities): #for each entity in file/sentence/text
    #         if hmariEntityNo<len(hmariEntities):
    #             hmariEntity = hmariEntities[hmariEntityNo][0] #For custNER
    #
    #         # if its not per/loc/org entity e.g misc in case of illinois, date title etc in case of stanford
    #         while flag == False and (not (str.startswith(hmariEntity['ner'], 'PER')) and not (str.startswith(hmariEntity['ner'], 'LOC')) and not (str.startswith(hmariEntity['ner'], 'ORG'))):
    #             hmariEntityNo += 1
    #             if hmariEntityNo < len(hmariEntities):
    #                 hmariEntity = hmariEntities[hmariEntityNo][0]  # for custNER
    #             elif hmariEntityNo == len(hmariEntities):  # if its the last entity and is not ppo type
    #                 flag = True
    #                 continue
    #
    #         # if not (str.startswith(hmariEntity['ner'], 'PER') or str.startswith(hmariEntity['ner'],'LOC') or str.startswith(hmariEntity['ner'], 'ORG')):  # if hmariEntity is not PPO type
    #         #     hmariEntityNo += 1
    #         #     continue
    #
    #         if datasetEntityNo<len(datasetEntities):
    #             datasetEntity = datasetEntities[datasetEntityNo]
    #             if nextDatasetEntity==True and 'gender' in datasetEntity.keys():
    #                 if str.startswith(datasetEntity['gender'],'MALE'):
    #                     noOfDatasetMaleEntities += 1
    #                 elif str.startswith(datasetEntity['gender'],'FEMALE'):
    #                     noOfDatasetFemaleEntities += 1
    #                 nextDatasetEntity = False
    #
    #         if datasetEntityNo >= (len(datasetEntities) - 1) and hmariEntityNo >= (len(hmariEntities) - 1):
    #             flag2 = True
    #         offsetRangeDataset = range(int(datasetEntity['characterOffsetBegin']), int(datasetEntity['characterOffsetEnd']))
    #         offsetRangeHmari = range(hmariEntity['characterOffsetBegin'], hmariEntity['characterOffsetEnd'])
    #         offsetRangesOverlap = [i for i in offsetRangeDataset if i in offsetRangeHmari]
    #
    #         # FOR WEAK ANNOTATION
    #         if len(offsetRangesOverlap) > 0 and datasetEntityNo<len(datasetEntities) and hmariEntityNo<len(hmariEntities):
    #             truePositives += 1
    #             if str.startswith(hmariEntity['gender'],'MALE') and str.startswith(datasetEntity['gender'],'MALE'):
    #                 maleTP += 1
    #             elif str.startswith(hmariEntity['gender'],'MALE') and not(str.startswith(datasetEntity['gender'],'MALE')):
    #                 maleFP += 1
    #             elif not(str.startswith(hmariEntity['gender'],'MALE')) and str.startswith(datasetEntity['gender'],'MALE'):
    #                 maleFN += 1
    #             if str.startswith(hmariEntity['gender'],'FEMALE') and str.startswith(datasetEntity['gender'],'FEMALE'):
    #                 femaleTP += 1
    #             elif str.startswith(hmariEntity['gender'],'FEMALE') and not(str.startswith(datasetEntity['gender'],'FEMALE')):
    #                 femaleFP += 1
    #             elif not(str.startswith(hmariEntity['gender'],'FEMALE')) and str.startswith(datasetEntity['gender'],'FEMALE'):
    #                 femaleFN += 1
    #             hmariEntityNo += 1
    #             datasetEntityNo += 1
    #             nextDatasetEntity = True
    #         else:
    #             if (hmariEntity['characterOffsetBegin'] > int(datasetEntity['characterOffsetBegin']) and datasetEntityNo<len(datasetEntities)) or (hmariEntityNo>=len(hmariEntities)):
    #                 falseNegatives += 1
    #                 if str.startswith(datasetEntity['gender'], 'MALE'):
    #                     maleFN += 1
    #                 elif str.startswith(datasetEntity['gender'],'FEMALE'):
    #                     femaleFN += 1
    #                 datasetEntityNo += 1
    #                 nextDatasetEntity = True
    #             else:
    #                 falsePositives += 1
    #                 if str.startswith(hmariEntity['gender'],'MALE'):
    #                     maleFP += 1
    #                 elif str.startswith(hmariEntity['gender'],'FEMALE'):
    #                     femaleFP += 1
    #                 hmariEntityNo += 1
    #
    #         if flag2==True:
    #             break
    #
    # Precision = truePositives / (truePositives + falsePositives)
    # malePrecision = maleTP / (maleTP + maleFP)
    # femalePrecision = femaleTP / (femaleTP + femaleFP)
    # Recall = truePositives / (truePositives + falseNegatives)
    # maleRecall = maleTP / (maleTP + maleFN)
    # femaleRecall = femaleTP / (femaleTP + femaleFN)
    # F1 = 2 * Precision * Recall / (Precision + Recall)
    # maleF1 = 2 * malePrecision * maleRecall / (malePrecision + maleRecall)
    # femaleF1 = 2 * femalePrecision * femaleRecall / (femalePrecision + femaleRecall)
    # sumOfTPs = maleTP + femaleTP
    # sumOfFPs = maleFP + femaleFP
    # sumOfFNs = maleFN + femaleFN
    # microPrec = sumOfTPs / (sumOfTPs + sumOfFPs)
    # microRec = sumOfTPs / (sumOfTPs + sumOfFNs)
    # microF1 = 2 * microPrec * microRec / (microPrec + microRec)
    # sumOfPrecs = malePrecision + femalePrecision
    # sumOfRecs = maleRecall + femaleRecall
    # macroPrec = sumOfPrecs / 2  # because there are 2 types for classification i.e. male, female
    # macroRec = sumOfRecs / 2  # because there are 2 types for classification i.e. male, female
    # macroF1 = 2 * macroPrec * macroRec / (macroPrec + macroRec)
    #
    # #Following block writes the results to file
    # f = open(resultFile, "w+")
    # f.write("Input file: %s\n\n" % nifDatasetFile)
    # f.write("------------------------------------------------")
    # f.write("\n-----------------WEAK ANNOTATION RESULTS----------------")
    # f.write("\n------------------------------------------------\n\n")
    # f.write("\n----------------- MICRO ----------------\n")
    # f.write("Micro Precision: %f\n" % microPrec)
    # f.write("Micro Recall: %f\n" % microRec)
    # f.write("Micro F1: %f\n" % microF1)
    # f.write("\n----------------- MACRO ----------------\n")
    # f.write("Macro Precision: %f\n" % macroPrec)
    # f.write("Macro Recall: %f\n" % macroRec)
    # f.write("Macro F1: %f\n" % macroF1)
    # f.write("\n-----------------TOTAL ENTITIES: Recognition, not typing----------------\n")
    # f.write("Total Entities in Dataset: %d\n" % noOfDatasetEntities)
    # f.write("TruePositives: %d\n" % truePositives)
    # f.write("FalsePositives: %d\n" % falsePositives)
    # f.write("FalseNegatives: %d\n\n" % falseNegatives)
    # f.write("Precision: %f\n" % Precision)
    # f.write("Recall: %f\n" % Recall)
    # f.write("F1: %f\n" % F1)
    #
    # f.write("\n-----------------MALE ENTITIES----------------\n")
    # f.write("No of MALE entities in dataset: %d\n" % noOfDatasetMaleEntities)
    # f.write("TruePositives: %d\n" % maleTP)
    # f.write("FalsePositives: %d\n" % maleFP)
    # f.write("FalseNegatives: %d\n\n" % maleFN)
    # f.write("Precision: %f\n" % malePrecision)
    # f.write("Recall: %f\n" % maleRecall)
    # f.write("F1: %f\n" % maleF1)
    #
    # f.write("\n-----------------FEMALE ENTITIES----------------\n")
    # f.write("No of FEMALE entities in dataset: %d\n" % noOfDatasetFemaleEntities)
    # f.write("TruePositives: %d\n" % femaleTP)
    # f.write("FalsePositives: %d\n" % femaleFP)
    # f.write("FalseNegatives: %d\n\n" % femaleFN)
    # f.write("Precision: %f\n" % femalePrecision)
    # f.write("Recall: %f\n" % femaleRecall)
    # f.write("F1: %f\n" % femaleF1)


def convertOKEdatasetFromNIFtoTxt(sizeOfDataset,nifDatasetFile,txtFilePath): #gender added
    graph = Graph()
    graph.parse(nifDatasetFile, format=guess_format(nifDatasetFile))
    dataset = [None]*sizeOfDataset #dataset is going to be list of dicts of dicts, and the whole OKE dataset will be loaded in it
    # print(len(graph))
    # print("--- printing raw triples ---")
    # for stmt in graph:
    #     pprint.pprint(stmt)
    # for subject, predicate, object in graph:
    #     print(subject)
    #     print(predicate)
    #     print(object)
    for subject, predicate, object in graph:
        temp = str.split(subject, '#')
        # print(temp[0])
        # print(temp[1])
        entityNo = temp[1]
        temp = str.split(temp[0], 'sentence-')
        sentenceNo = int(temp[1])

        if dataset[sentenceNo] == None:
            dataset[sentenceNo] = {}
        if not(entityNo in dataset[sentenceNo].keys()):
            dataset[sentenceNo][entityNo] = {}

        if str.find(predicate,'beginIndex')!=-1:
            dataset[sentenceNo][entityNo]['characterOffsetBegin'] = object
        elif str.find(predicate,'endIndex')!=-1:
            dataset[sentenceNo][entityNo]['characterOffsetEnd'] = object
        elif str.find(predicate,'anchorOf')!=-1:
            dataset[sentenceNo][entityNo]['text'] = object
        elif str.find(predicate,'taIdentRef')!=-1:
            dataset[sentenceNo][entityNo]['resource'] = object
        elif str.find(predicate,'isString')!=-1:
            dataset[sentenceNo][entityNo]['isString'] = object
        elif str.find(predicate,'nerType')!=-1:
            dataset[sentenceNo][entityNo]['ner'] = object
        elif str.find(predicate,'personGender')!=-1:
            dataset[sentenceNo][entityNo]['gender'] = object

    #following loop removes the isString dicts from entities
    for sentenceNo in range(0, sizeOfDataset):
        for entityNo in dataset[sentenceNo]:
            # print(dataset[sentenceNo][entityNo])
            if 'isString' in dataset[sentenceNo][entityNo].keys():
                isStringEntityNo = entityNo
        dict.pop(dataset[sentenceNo], isStringEntityNo)

    #following loop converts the the entityNo's from 'char=326,348' form to integer form (326)
    #this is necessary to have the entities in order
    for sentenceNo in range(0,sizeOfDataset):
        entities = {}
        for entityNo in dataset[sentenceNo]:
            temp = str.split(entityNo,'=')
            temp = temp[1]
            temp = str.split(temp,',')
            newEntityNo = int(temp[0])
            entities[newEntityNo]= dataset[sentenceNo][entityNo]
        dataset[sentenceNo] = entities

    #following loop converts the dataset FROM a list of dicts of dicts TO a list of list of dicts
    #while doing so, it also sorts the entities on characterOffsetBegin
    #so that its format matches with HmaraNER's format
    for sentenceNo in range(0,sizeOfDataset):
        entities = []#[None]*len(dataset[sentenceNo])
        for entityNo in sorted(dataset[sentenceNo]):
            entities.append(dict.get(dataset[sentenceNo],entityNo))
        dataset[sentenceNo] = entities

    #following block writes entities in each sentence of OKE dataset in a file
    try:
        for sentenceNo in range(0, sizeOfDataset):
            file = txtFilePath+str(sentenceNo)+'.txt'
            with open(file, 'w', encoding='utf-8') as outfile:
                json.dump(dataset[sentenceNo], outfile, sort_keys = True, indent = 4, ensure_ascii = False)
    except json.JSONDecodeError:
        print("Decoding JSON has failed")

    print('Dataset conversion Done!')




def convertFoxAdelFromNIFtoTxt():
    sizeOfOkeEvalDataset = 58

    for i in [0,1]:
        if i==0: #for FOX
            nifFilePath = "outputOKE\eval\\fox\\"
            txtFilePath = 'outputOKE\eval\\fox\\entities\\'
        elif i==1: #for ADEL
            nifFilePath = "outputOKE\eval\\adel\\"
            txtFilePath = 'outputOKE\eval\\adel\\entities\\'
        for j in range(0,sizeOfOkeEvalDataset):
            print("printing i and j .................................................")
            print(i)
            print(j)
            nifFile = nifFilePath+str(j)+".xml.ttl"
            txtFile = txtFilePath+str(j)+'.txt'
            graph = Graph()
            # print(guess_format(nifFile))
            graph.parse(nifFile,format='turtle')#guess_format(nifFile))
            dataset = [None]*1# sizeOfOkeEvalDataset #dataset is going to be list of dicts of dicts, and the whole OKE dataset will be loaded in it
            # print(len(graph))
            # print("--- printing raw triples ---")
            # for stmt in graph:
            #     pprint.pprint(stmt)
            # for subject, predicate, object in graph:
            #     # print(subject)
            #     # print(predicate)
            #     # print(object)
            #     # print()
            for subject, predicate, object in graph:
                temp = str.split(subject, '#')
                # print(temp[0])
                # print(temp[1])
                entityNo = temp[1]
                sentenceNo = 0

                if dataset[sentenceNo] == None:
                    dataset[sentenceNo] = {}
                if not(entityNo in dataset[sentenceNo].keys()):
                    dataset[sentenceNo][entityNo] = {}

                if str.find(predicate,'beginIndex')!=-1:
                    dataset[sentenceNo][entityNo]['characterOffsetBegin'] = int(object)
                elif str.find(predicate,'endIndex')!=-1:
                    dataset[sentenceNo][entityNo]['characterOffsetEnd'] = int(object)
                elif str.find(predicate,'anchorOf')!=-1:
                    dataset[sentenceNo][entityNo]['text'] = object
                elif str.find(predicate,'taIdentRef')!=-1:
                    dataset[sentenceNo][entityNo]['resource'] = object
                elif str.find(predicate,'isString')!=-1:
                    dataset[sentenceNo][entityNo]['isString'] = object
                elif str.find(predicate,'taClassRef')!=-1:
                    if str.find(str.lower(object),'person')!=-1:
                        object = 'PERSON'
                    elif str.find(str.lower(object),'location')!=-1:
                        object = 'LOCATION'
                    elif str.find(str.lower(object),'organization')!=-1:
                        object = 'ORGANIZATION'

                    if object=='PERSON' or object=='LOCATION' or object=='ORGANIZATION':
                        dataset[sentenceNo][entityNo]['ner'] = object

            #following loop removes the isString  and non-PPO dicts from entities
            for sentenceNo in range(0, 1):
                for entityNo in dataset[sentenceNo]:
                    # print(dataset[sentenceNo][entityNo])
                    if 'isString' in dataset[sentenceNo][entityNo].keys():
                        isStringEntityNo = entityNo
                dict.pop(dataset[sentenceNo], isStringEntityNo)
                notPpoEntityNos = []
                for entityNo in dataset[sentenceNo]:
                    # print(dataset[sentenceNo][entityNo])
                    if not('ner' in dataset[sentenceNo][entityNo].keys()):
                        notPpoEntityNos.append(entityNo)
                for notPpoEntityNo in notPpoEntityNos:
                    dict.pop(dataset[sentenceNo], notPpoEntityNo)


            #following loop converts the the entityNo's from 'char=326,348' form to integer form (326)
            #this is necessary to have the entities in order
            for sentenceNo in range(0,1):
                entities = {}
                for entityNo in dataset[sentenceNo]:
                    if str.find(entityNo,'=')!=-1:
                        temp = str.split(entityNo,'=')
                        temp = temp[1]
                    elif str.find(entityNo,'char')!=-1:
                        temp = str.strip(entityNo,'char')
                    temp = str.split(temp,',')
                    newEntityNo = int(temp[0])
                    entities[newEntityNo]= dataset[sentenceNo][entityNo]
                dataset[sentenceNo] = entities

            #following loop converts the dataset FROM a list of dicts of dicts TO a list of list of dicts
            #while doing so, it also sorts the entities on characterOffsetBegin
            #so that its format matches with HmaraNER's format
            for sentenceNo in range(0,1):
                entities = []#[None]*len(dataset[sentenceNo])
                for entityNo in sorted(dataset[sentenceNo]):
                    entities.append(dict.get(dataset[sentenceNo],entityNo))
                dataset[sentenceNo] = entities

            #following block writes entities in each doc to a file
            try:
                with open(txtFile, 'w', encoding='utf-8') as outfile:
                    json.dump(dataset[0], outfile, sort_keys = True, indent = 4, ensure_ascii = False)
            except json.JSONDecodeError:
                print("Decoding JSON has failed")

    print('Dataset conversion Done!')