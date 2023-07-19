from SPARQLWrapper import SPARQLWrapper,JSON
import nltk
from CorefResolution import areCorefs
import json
import os
import copy
import gender_guesser.detector as genderDict

malePronouns = ['he', 'him', 'his','himself']
femalePronouns = ['she', 'her', 'hers', 'herself']
selfPronouns = ['I','me', 'my', 'myself']
#following lists are taken from https://en.wikipedia.org/wiki/English_honorifics and https://ielts.idp.com/prepare/article-grammar-101-feminine-and-masculine-words-in-english
maleTitles = ['Mr.','Mr','Gentleman','Sire','King','Prince','Sir','Lord','Esq','man','boy','actor','waiter']
femaleTitles = ['Mrs.','Mrs','Miss','Ms.','Ms','Mistress','Queen','Princess','Lady','Dame','Madam','Ma\'am','woman','girl','actress','waitress']

maleRelations = ['father', 'dad', 'daddy', 'papa', 'stepfather',
                         'step-father',
                         'son', 'stepson', 'step-son', 'husband', 'ex-husband',
                         'hubby', 'ex-hubby', 'boyfriend', 'ex-boyfriend',
                         'brother', 'stepbrother', 'step-brother',
                         'half-brother', 'halfbrother', 'bro', 'uncle',
                         'nephew',
                         'father-in-law', 'fathers-in-law', 'brother-in-law',
                         'brothers-in-law', 'son-in-law', 'sons-in-law',
                         'grandfather', 'grand-father',
                         'great-grandfather', 'grandson', 'grand-son',
                         'grandson', 'great-grandson', 'co-husband']
femaleRelations = ['mama', 'mother', 'mum', 'mom', 'stepmother',
                           'step-mother', 'daughter', 'stepdaughter',
                           'step-daughter', 'widow', 'ex-widow', 'wife',
                           'ex-wife', 'wives', 'ex-wives',
                           'girlfriend', 'ex-girlfriend', 'mistress',
                           'ex-mistress', 'sister', 'stepsister',
                           'step-sister', 'half-sister', 'halfsister', 'sis',
                           'aunt', 'niece',
                           'mother-in-law', 'mothers-in-law', 'sister-in-law',
                           'sisters-in-law',
                           'daughter-in-law', 'daughters-in-law', 'grandmother',
                           'grand-mother', 'great-grandmother',
                           'granddaughter', 'grand-daughter',
                           'great-granddaughter',
                           'co-wife', 'co-wives']


# because we want to test the performance of gender annotations,
# so for evaluation purpose, this function takes the person annotations in oke corrected dataset, and marks persons as male or female
#For Gender evaluation purpose,
#our system reads from "withoutGender2" folder,
#and writes output to "genderAnnotationsByCustNER" folder
def applyRulesGender(dataset,trainEval):
    if dataset=='oke':
        if trainEval=='train':
            inputTextsFile = "inputOKE\okeTrainInputTexts.txt"
            dir = 'outputOKE\\train\hmariEntities\\'
        elif trainEval=='eval':
            inputTextsFile = "inputOKE\okeEvalInputTexts.txt"
            # dir = 'outputOKE\\eval\hmariEntities\\'
            dir = 'outputOKE\\eval\okeEntitiesCorrected\\' #for gender evaluation purpose
    elif dataset=='conll':
        if trainEval=='train':
            inputTextsFile = "inputConll\conll2003.eng.train.2.txt"
            dir = 'outputConll\\train\hmariEntities\\'
        elif trainEval=='eval':
            inputTextsFile = "inputConll\conll2003.eng.testb.sentences.txt"
            dir = 'outputConll\\eval\hmariEntities\\'
    # dirIn = dir+'allEntities'
    # dirOut = dir+'allEntitiesWithGender'
    dirIn = dir+'withoutGender2\\' #for gender evaluation purpose
    dirOut = dir+'genderAnnotationsByCustNERrulesOnly' #for gender evaluation purpose

    # this block reads the input texts from the corresponding file into the 'inputTexts' list
    with open(inputTextsFile,'r', encoding='utf-8', errors='ignore') as inTFile:
        inputTexts=[]
        if dataset=='oke':
            for line in inTFile:
                inputTexts.append(line)
        elif dataset=='conll':
            text = ''
            for line in inTFile:  # read a file one line at a time using for loop
                if (line.find('-DOCSTART-') == -1 and line.find('-DOCEND-') == -1):
                    text = text + line
                elif (line.find('-DOCSTART-') != -1 or line.find('-DOCEND-') != -1) and text != '':
                    inputTexts.append(text)
                    text=''

    onlyfiles = next(os.walk(dirIn))[2]  # dir is your directory path as string
    # inFileCount = len(onlyfiles)
    inFileCount = 58 #for oke eval  #for gender evaluation purpose

    # x = 19
    for fileNo in range(inFileCount): # for each file
        # if fileNo<x: continue
        # if fileNo>x: break
        personPronouns = []
        text = inputTexts[fileNo]
        # inFile = dirIn+'\output'+str(i)+'.txt'
        inFile = dirIn+'\doc'+str(fileNo)+'.txt' #for gender evaluation purpose
        outFile = dirOut+'\output'+str(fileNo)+'.txt'
        tokens = nltk.word_tokenize(text)
        PoSs = nltk.pos_tag(tokens)
        print('------------------------------------------------------------------')
        print('Example no: '+str(fileNo)+'\t'+text)

        # following loop populates personPronouns array, where each personPronoun is a tuple containing three things: the token, its pos and its index
        for j, PoS in enumerate(PoSs):  # for each token's PoS in example. Each PoS is a tuple containing two things: the token, and its pos
            if (PoS[1] == 'PRP' or PoS[1] == 'PRP$') and str.lower(PoS[0]) != 'its' and str.lower(
                    PoS[0]) != 'itself' and str.lower(PoS[0]) != 'it' and str.lower(PoS[0]) != 'here' and str.lower(
                PoS[0]) != 'there' and str.lower(PoS[0]) != 'us' and str.lower(PoS[0]) != 'we' and str.lower(
                PoS[0]) != 'our':
                # if its a pro-noun. token is not 'it' ensures that this is a person(s) pronoun
                # list of person pronouns: I, you, he, she, we, they, me, him, her, us, them, their, his
                # or rather simply its any pronoun other than 'it'
                # list of pronouns at https://www.analyticsvidhya.com/blog/2017/12/introduction-computational-linguistics-dependency-trees/
                # print(PoS[0])
                personPronoun = [PoS[0], PoS[1],j]  # Each personPronoun is a tuple containing three things: the token, its pos and its index
                personPronouns.append(personPronoun)
        print('Person Pronouns in this example: ')
        print(personPronouns)

        # open input, output files
        with open(inFile,'r',encoding='utf-8', errors='ignore') as json_in_file, open(outFile, 'w', encoding='utf-8', errors='ignore') as outF:
            data = json.load(json_in_file)
            dataUpdated = copy.deepcopy(data)
            outF.truncate(0)

            for (entityNo,entity) in enumerate(data):
                gender = -1
                # thisEntity = entity[0]
                thisEntity = entity  # for gender evaluation purpose
                if 'ner' in dict.keys(thisEntity):
                    if thisEntity['ner']=='PER' or thisEntity['ner']=='PERSON':#IF this entity is a person
                        personCharStart = (int)(thisEntity['characterOffsetBegin'])
                        personCharEnd = (int)(thisEntity['characterOffsetEnd'])
                        personString = text[personCharStart:personCharEnd]
                        print("PERSON ENTITY: "+str(entityNo)+": "+personString)
                        stringB4Person = text[0:personCharStart]
                        toks = nltk.word_tokenize(stringB4Person)
                        personTokens = nltk.word_tokenize(personString)
                        # toks = str.split(stringB4Person)
                        personTokStart = len(toks)
                        personTokEnd = personTokStart+len(personTokens)-1
                        previousToken=''
                        if len(toks)!=0:
                            previousToken = toks[len(toks)-1]

                        # RULE 1: If person precedes (or if its first word is) male/female title (Mr/Ms/king etc.) or male/female relation word (mother, son etc), then add gender accordingly
                        if previousToken in femaleTitles or previousToken in femaleRelations or personTokens[0] in femaleTitles or personTokens[0] in femaleRelations:
                            gender = 'f'
                            print('Previous token is female Title or relation word')
                        elif previousToken in maleTitles or previousToken in maleRelations or personTokens[0] in maleTitles or personTokens[0] in maleRelations:
                            gender = 'm'
                            print('Previous token is male Title or relation word')
                        else:
                            # RULE 2: If any of the corefs of person is male/female pronoun, then add gender accordingly
                            for personPronoun in personPronouns:
                                if str.lower(personPronoun[0]) in femalePronouns or str.lower(personPronoun[0]) in malePronouns:
                                    isCoref = areCorefs(personPronoun[2],personPronoun[2],personTokStart,personTokEnd,tokens)
                                    if isCoref:
                                        if str.lower(personPronoun[0]) in femalePronouns:
                                            gender = 'f'
                                            print('Coref is female')
                                            break
                                        elif str.lower(personPronoun[0]) in malePronouns:
                                            gender = 'm'
                                            print('Coref is male')
                                            break
                        # RULE 3: If person has dbpedia resource, then if rdfs:comment contains male/female pronouns, then add gender according to first pronoun
                        if gender==-1 and 'resource' in dict.keys(thisEntity):
                            if str.startswith(thisEntity['resource'],('http://dbpedia.org')):
                                personDbpediaUri = thisEntity['resource']
                                gender,comment = getGenderFromDBPedia(personDbpediaUri,fileNo,entityNo)
                                # dataUpdated[entityNo][0]['rdfsComment'] = comment
                                dataUpdated[entityNo]['rdfsComment'] = comment

                        # # RULE 4: If gender still not decided, then look-up first name in an external dictionary
                        if gender==-1: #if still gender not decided
                            print()
                            gender = getGenderFromDictionary(personTokens[0]) #dictionary look up
                        if gender!=-1:
                            if gender=='m':
                                # dataUpdated[entityNo][0]['gender']='MALE'
                                dataUpdated[entityNo]['gender']='MALE'
                            elif gender=='f':
                                # dataUpdated[entityNo][0]['gender'] = 'FEMALE'
                                dataUpdated[entityNo]['gender'] = 'FEMALE'
                        else:
                            # dataUpdated[entityNo]['gender'] = 'MALE' #if we can't decide gender, than its male
                            print('gender is not identified for this person')#, declaring male')
            json.dump(dataUpdated, outF, sort_keys=True, indent=4, ensure_ascii=False)
    print("Gender Files written successfully..")


# this function calls the gender_guesser python library to Get the gender from first name, https://pypi.org/project/gender-guesser/#description
# the library has a gazetteer/dictionary of over 40,000 first names as male/female, and makes a simple dictionary look-up of the input first name
# The result from library will be one of: unknown (name not found in dictionary), andy (same probability to be male than to be female), male, female, mostly_male, or mostly_female.
def getGenderFromDictionary(firstName):#
    gender=-1
    d = genderDict.Detector()
    result = d.get_gender(firstName)
    if result=='mostly_female' or result=='female':
        gender='f'
        print("person is female from dictionary")
    elif result=='mostly_male' or result=='male':
        gender='m'
        print("person is male from dictionary")
    return gender

def getGenderFromDBPedia(uri,fileNo,entityNo):
    gender = -1
    comment = ''

    # querying dbpedia takes lot of time, so instead read from the files where dbpedia comments are already written once
    inFile = 'outputOKE\eval\okeEntitiesCorrected\genderAnnotationsByCustNER+defaultGenderMale\output' + str(fileNo) + '.txt'  # for gender evaluation purpose
    with open(inFile, 'r', encoding='utf-8', errors='ignore') as json_in_file:
        data = json.load(json_in_file)
        entity = data[entityNo]
        if 'rdfsComment' in entity.keys():
            comment = entity['rdfsComment']

    # Following blocks brings rdfsComment from dbpedia of passed uri
    # # uri = "http://dbpedia.org/resource/Donald_Trump"
    # query = "SELECT ?comment WHERE {<"+uri+"> rdfs:comment ?comment. FILTER(lang(?comment) = \"en\")}"
    # sparql = SPARQLWrapper("http://dbpedia.org/sparql/")
    # sparql.setReturnFormat(JSON)
    # sparql.setTimeout(60000)  # 60 sec
    # # sparql.setHTTPAuth()
    # sparql.setQuery(query)
    # response = sparql.query().convert()
    # if response != '' and len(response["results"]["bindings"]) != 0:  # if comment returned
    #     comment = response["results"]["bindings"][0]['comment']['value']

    print('Dbpedia Comment: '+comment)
    mIndex = fIndex = -1
    if comment!="":
        commentTokens = nltk.word_tokenize(comment)
        for i,m in enumerate(malePronouns):#this loop gives us the lowest index where a male pronoun is found in the rdfs:comment of person dbpedia resource
            for j,tok in enumerate(commentTokens):
                if m==tok.lower():
                    if mIndex==-1 or j<mIndex:
                        mIndex = j
        for i,f in enumerate(femalePronouns):#this loop gives us the lowest index where a female pronoun is found in the rdfs:comment of person dbpedia resource
            for j,tok in enumerate(commentTokens):
                if f==tok.lower():
                    if fIndex==-1 or j<fIndex:
                        fIndex = j
    if mIndex!=-1 and fIndex!=-1:
        if fIndex < mIndex:
            print("person is female from dbpedia")
            gender = 'f'
        elif mIndex<fIndex:
            print("person is male from dbpedia")
            gender = 'm'
    elif mIndex!=-1 or fIndex!=-1:
        if mIndex != -1:
            print("person is male from dbpedia")
            gender = 'm'
        if fIndex != -1:
            print("person is female from dbpedia")
            gender = 'f'
    else:
        print("we do not know if person is male or female from dbpedia")
    return gender,comment