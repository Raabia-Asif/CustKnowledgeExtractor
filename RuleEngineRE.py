import copy
import re
import json
from nltk.tokenize.treebank import TreebankWordDetokenizer
from CorefResolution import populateCorefs
# from ClientNER import areCorefs
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
objNoOfThisRel = 0 #only for direct cases
malePronouns = ['he', 'him', 'his','himself']
femalePronouns = ['she', 'her', 'hers', 'herself']
selfPronouns = ['I','me', 'my', 'myself']
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

def applyRulesRE(dataset,corrected,examples): # dataset = 'train' or 'dev' or 'test' or 'ours' for ourFREdataset, examples = 'positive' or 'negative'
    '''this function reads the dataset in json format from inFile and writes the predicted relations to outF named outputCustRE.txt in corresponding folder '''
    # example_sent = "Besides his son Charles , an author and journalist , Kaiser is survived by his wife of 67 years , Hannah Greeley Kaiser ; two other sons , Robert G , a former managing editor and now an associate editor of The Washington Post , and David , a professor of history at the Naval War College in Newport , RI ; and four grandchildren ."
    # stop_words = set(stopwords.words('english'))
    # word_tokens = word_tokenize(example_sent)
    # filtered_sentence = [w for w in word_tokens if not w in stop_words]
    # print(TreebankWordDetokenizer().detokenize(filtered_sentence))
    if corrected==True:
        directoryIn = 'inputTACRED\\familyAllCorrected\\'+dataset
        directoryOut = 'outputTACRED\\'+dataset+'\\familyAllCorrected\\'
    else:
        directoryIn = 'inputTACRED\\familyAll\\'+dataset
        directoryOut = 'outputTACRED\\'+dataset+'\\familyAll\\'
    if dataset=='ours':
        directoryIn = 'inputOurFREdataset\\'
        directoryOut = 'outputOurFREdataset\\'
        inFile = directoryIn + 'datasetAnnotated.json'
    elif dataset == 'ours1':
        directoryIn = 'inputOurFREdatasetNew1\\'
        directoryOut = 'outputOurFREdatasetNew1\\'
        inFile = directoryIn + 'datasetAnnotated.json'
    elif dataset == 'ours2':
        directoryIn = 'inputOurFREdatasetNew2\\'
        directoryOut = 'outputOurFREdatasetNew2\\'
        inFile = directoryIn + 'datasetAnnotated.json'
    elif dataset == 'cust':
        directoryIn = 'inputCustFREdataset\\'
        directoryOut = 'outputCustFREdataset\\'
        inFile = directoryIn + 'datasetAnnotated.json'
    else:
        inFile = directoryIn+".json"
    inFileTexts = directoryIn+"InputTexts.txt"
    inFileCoref = directoryOut+"isCoref.txt"
    outFile = directoryOut+"outputCustRE.txt"
    patternsFile = directoryOut+"patterns.txt"
    patternsWordsFile = directoryOut+"patternsWords.txt"
    corefsFile = directoryOut+"corefs.json"
    outFileKey = directoryOut+"relationsKey.txt"

    # the json files of TACRED dataset are in correct format, but the file exported by MongoDB needs formatting: 1. replaced all '\n' in 'familyNoRelationsPerPerFiltered' json files with ',', 2. Added [ at beginning of file and ] at the end of file
    with open(inFile) as json_in_file, open(inFileTexts,'r', encoding='utf-8', errors='ignore') as inFTexts, open(outFile, 'a', encoding='utf-8', errors='ignore') as outF, open(patternsFile, 'a', encoding='utf-8', errors='ignore') as patternsF, open(patternsWordsFile, 'a', encoding='utf-8', errors='ignore') as patternsWordsF, open(inFileCoref, 'r', encoding='utf-8', errors='ignore') as corefFile, open(corefsFile) as coref_json_file, open(outFileKey, 'a', encoding='utf-8', errors='ignore') as outFKey: #outFKey only required for writing output key file, it is one time task, no need to open this file again
        data = json.load(json_in_file)

        # # UN COMMENT following block for writing isCoref and corefs files
        # print('\nStarting Writing isCoref.txt and corefs.json files.......') #one time task
        # sentences = inFTexts.read() #one time task
        # createCorefsFiles(data,sentences,corefsFile,inFileCoref) #one time task
        # exit() #one time task
        # # # END: UN COMMENT above block for writing isCoref and corefs files, and comment all below

        outF.truncate(0)
        patternsF.truncate(0)
        patternsWordsF.truncate(0)
        print('Starting Relation Extraction...')
        corefsList = json.load(coref_json_file)

        # outFKey.truncate(0)# # UN COMMENT this line for writing answer key file + uncomment block below # ONE TIME TASK for writing relations key file
        for exampleNo,(example,sentence,coref,corefs) in enumerate(zip(data, inFTexts, corefFile,corefsList)): #exampleNo,(example,sentence) in enumerate(zip(data, inFTexts)):#
            # if exampleNo+1 < 14:continue
            # if exampleNo >= 1838: break

            # # UN COMMENT following block for writing answer key file + uncomment one line above to truncate outFKey
            # # ONE TIME TASK for writing relations key file
            # relationGold = example['relation'] #relation to be predicted
            # outFKey.write(relationGold) # One time task. no need to do again. writes the relations key file
            # outFKey.write('\n') # One time task. no need to do again.  writes the relations key file
            # continue # One time task. no need to do again.  writes the relations key file

            print('\n\n\nEXAMPLE NO:'+str(exampleNo+1))
            print('\nText: '+sentence)
            coref = coref[0] #because i only want the first character which is 0 or 1, and not the last \n
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
            if subjStart<objStart: #this if-else is for calculating start and end indices of concerned area i.e. area in between subject and object
                s = subjStart
                s2 = subjEnd
                e = objEnd+1
                e1 = objStart
            else:
                s = objStart
                s2 = objEnd
                e = subjEnd+1
                e1 = subjStart
            tokens = example['token']
            tokens = ['(' if x=='-LRB-' else x for x in tokens] # replacing all '-LRB-' tokens by '('
            tokens = [')' if x=='-RRB-' else x for x in tokens] # replacing all '-RRB-' tokens by ')'
            tokens = ['[' if x=='-LSB-' else x for x in tokens] # replacing all '-LRB-' tokens by '('
            tokens = [']' if x=='-RSB-' else x for x in tokens] # replacing all '-RRB-' tokens by ')'
            sentence = sentence.replace('-LRB-','(')
            sentence = sentence.replace('-RRB-',')')
            sentence = sentence.replace('-LSB-','[')
            sentence = sentence.replace('-RSB-',']')

            depHeads = example['stanford_head']# this examples stanford dep head array, column no 11 in conll file
            depRels = example['stanford_deprel']# this examples stanford dep relation array, column no 10 in conll file
            stanfordNERs = example['stanford_ner']# this examples stanford_ner array, column no 9 in conll file
            PoSs = example['stanford_pos']# this examples stanford_pos array, column no 8 in conll file
            subj = tokens[subjStart:subjEnd+1] #this example's subject text
            obj = tokens[objStart:objEnd+1] #this example's object text
            subj = TreebankWordDetokenizer().detokenize(subj)
            obj = TreebankWordDetokenizer().detokenize(obj)
            listWordsFound = []
            listWordsFoundInBtw = []
            personsFoundInBtw = 0#[]
            pattern = " " * len(tokens) #pattern = '' #this string will hold person(P)/pronoun(p), listWord(w) pattern found in one example.
            patternWords = ['~']*len(tokens) #words/texts of this examples pattern
            isPerson = False
            negating = -1
            relFound = False
            subjPredictedIndex = -1
            objPredictedIndex = -1
            relationPredictedIndex = -1
            no = 1
            triples = []
            x = -1

            # the following loop creates the pattern of this example and populates the patternWords list.
            # the following loop also finds all the list words in input text and adds them to listWordsFound list,
            # each as a dictionary of 3 items: index at which word found, the exact token and the relation it represents
            for i, token in enumerate(tokens): # for each token in example
                rel = -1
                if i in range(subjStart,subjEnd+1):# and subjType=='PERSON':
                    #if its the subject
                    pattern = insertCharInString(pattern, 'S', i) # pattern+= 'S'  # add S to pattern
                    patternWords[i]=token
                elif i in range(objStart,objEnd+1):# and objType=='PERSON':
                    #if its the object
                    pattern = insertCharInString(pattern, 'O', i)  # pattern+= 'O'  # add O to pattern
                    patternWords[i]=token
                else:
                    if isPerson==False and stanfordNERs[i]=='PERSON':
                        #if its a Person token, add P to pattern. isPerson is to ensure that multi-word person makes only one entry to the pattern
                        pattern = insertCharInString(pattern, 'P', i)  # pattern+= 'P' #add P to pattern
                        patternWords[i]=token
                        isPerson = True #isPerson is to ensure that multi-word person makes only one entry to the pattern
                    elif stanfordNERs[i]!='PERSON':
                        isPerson = False
                        if (PoSs[i]=='PRP' or PoSs[i]=='PRP$') and str.lower(token)!='its' and str.lower(token)!='itself' and str.lower(token)!='it' and str.lower(token)!='here' and str.lower(token)!='there' and str.lower(token)!='us' and str.lower(token)!='we' and str.lower(token)!='our':
                            #if its a pro-noun, add p to pattern. token is not 'it' ensures that this is a person(s) pronoun
                            # list of person pronouns: I, you, he, she, we, they, me, him, her, us, them, their, his
                            # or rather simply its any pronoun other than 'it'
                            # list of pronouns at https://www.analyticsvidhya.com/blog/2017/12/introduction-computational-linguistics-dependency-trees/
                            pattern = insertCharInString(pattern, 'p', i)  # pattern+= 'p' #add p to pattern
                            patternWords[i]=token
                        elif (PoSs[i]=='POS'): #if its 's, add ' to pattern.
                            # POS (Possessive ending) is for 's
                            pattern = insertCharInString(pattern, '\'', i)  # pattern+= '\'' #add ' to pattern
                            patternWords[i]=token
                        elif (depRels[i]=='punct'): #if its punct, add it to pattern.
                            # e.g. ; : . ,
                            pattern = insertCharInString(pattern,token,i)  # pattern+= token #add the punct mark to pattern
                            patternWords[i]=token
                        elif (PoSs[i]=='CC' and (token=='and' or token=='&')): #if its Coordinating Conjunction 'and' or '&', add it to pattern.
                            # if its 'and' or '&'
                            if stanfordNERs[i-1]=='PERSON' or stanfordNERs[i+1]=='PERSON' or ((i-1) in range(subjStart,subjEnd+1))  or ((i+1) in range(subjStart,subjEnd+1))  or ((i-1) in range(objStart,objEnd+1))  or ((i+1) in range(objStart,objEnd+1)) : #if the previous or next token is person
                                pattern = insertCharInString(pattern, '&', i)  # pattern+= '&' #add & to pattern
                                patternWords[i]=token
                        elif PoSs[i]=='CD':#if token is a number
                            if type(depHeads[i]) is dict:
                                nummodIndex = int(depHeads[i]['$numberInt']) - 1 #the nummod dependency head of number
                            else:
                                nummodIndex = depHeads[i]-1 #to check if this num modifies a relation, coz have to add only those numbers in pattern and not just any numbers
                            numModifiesRel = False
                            if depRels[i]=='nummod' and matchListWord(tokens[nummodIndex],'')[0]!=-1:
                                numModifiesRel = True
                            elif depRels[i]=='nmod': #e.g. 420, In addition to Kempfer , he is survived by four of his other children , Silvestre , Elva Corrales , Robert and Kelly Harris ; 11 grandchildren ; and 11 great-grandchildren .
                                for j,x in enumerate(depHeads):
                                    if x==i+1:
                                        r = depRels[j]
                                        w = tokens[j]
                                        if r=='nmod' and matchListWord(w,'')[0]!=-1:
                                            numModifiesRel = True
                            if numModifiesRel==True:
                                if str.isnumeric(token):
                                    num = token
                                else:
                                    num = str(wordToNum(token))
                                if int(num)>9:
                                    pattern = insertCharInString(pattern, '9', i)  # pattern+='9'
                                else:
                                    pattern = insertCharInString(pattern,num,i)  # pattern+= num # add number to pattern
                                patternWords[i]=token
                        elif depRels[i]=='neg':#if its not , never, n't etc
                            negating = depHeads[i]
                        else:
                            nextTok = ''
                            isRelPlural = False
                            if i+1 < len(tokens): nextTok = tokens[i+1]
                            rel, isRelPlural = matchListWord(token,nextTok)
                            # if rel==-1 and i+1 < len(tokens):
                            #     if i + 2 < len(tokens):
                            #         nextTok = tokens[i + 2]
                            #     rel, isRelPlural = matchListWord(token+' '+tokens[i+1],'')

                            #For the new rules, i might have to uncomment and modify the following
                            if rel!=-1 and i!=(negating-1): # if token is a relation word
                                if rel==-2: #if this token is a notFamilyRelation word e.g. friend
                                    pattern = insertCharInString(pattern, 'n', i)  # add n to pattern
                                    patternWords[i] = token
                                else:
                                    if isRelPlural==True:# if relation is plural e.g. sons, children
                                        pattern = insertCharInString(pattern, 'r', i)  # pattern += 'r'  # add r to pattern
                                        patternWords[i]=token
                                    else: #if relation is singular
                                        pattern = insertCharInString(pattern, 'w', i)  # pattern+= 'w'  # add w to pattern
                                        patternWords[i]=token
                                listWordsFound.append({'index':i,'token':token,'rel':rel})
                                if i>s and i<e:
                                    listWordsFoundInBtw.append({'index':i,'token':token,'rel':rel})
                            else:#if its none of our concerned feature: not person, not relationWord, not number, ...
                                # keeping each char in pattern on same index as its corresponding word's index in text
                                pattern = insertCharInString(pattern, ' ', i)  # pattern+= ' '
                                patternWords[i] = '~'

            # personsFoundInBtw = pattern.count('P') #+ pattern.count('p')

            # concernedNer = stanfordNERs[s:e]
            # perContinued = 0
            # tokensPer = []
            # this loop finds all person entities in between s and o, and adds them to list personsFoundInBtw
            # each as a dictionary of 2 items: index at which person found starts, its text as list named tokens
            # for i in range(s2+1,e1-1):
            #     if stanfordNERs[i] == 'PERSON':
            #         if perContinued == 1:
            #             personsFoundInBtw.pop()
            #         else:
            #             index=i
            #         tokensPer.append(tokens[i])
            #         per = {'index':index,'tokens':tokensPer}
            #         personsFoundInBtw.append(per)
            #         perContinued = 1
            #     else:
            #         perContinued=0
            #         tokensPer = []
            # concernedText = tokens[s:e]  # this is list
            # concernedDepHeads = depHeads[s:e]
            # concernedDepRels = depRels[s:e]
            # concernedPoSs = PoSs[s:e]
            # concernedTextString = TreebankWordDetokenizer().detokenize(concernedText)

            # Writing pattern and pattern words to files
            patternsF.write(pattern)
            patternsF.write('\n')
            patternWordsString = ' '.join(map(str,patternWords)) # convert patternWords from list to string for writing to file
            patternsWordsF.write(patternWordsString)
            patternsWordsF.write('\n')

            # Rule 0: If no list word in sentence, then no_relation
            # Added to rule 0: First check if subj/object are Mr and Mrs someone, then they are spouse
            mrAndMrs = False
            mrIndex = mrsIndex = -1

            if sentence.find('Mrs.')!=-1:
                mrsIndex = tokens.index('Mrs.')
            if sentence.find('Mrs ')!=-1:
                mrsIndex = tokens.index('Mrs')
            if sentence.find('Mr.')!=-1:
                mrIndex = tokens.index('Mr.')#
            if sentence.find('Mr ')!=-1:
                mrIndex = tokens.index('Mr')#
            # for (i,p) in enumerate(pattern):
            #     print(str(i)+'\t'+str(p)+'\t'+patternWords[i])
            if mrIndex!=-1 and mrsIndex!=-1:
                # print(pattern)
                # print(pattern[mrIndex+1])
                # print(pattern[mrsIndex+1])
                if ((pattern[mrIndex+1]=='P' or pattern[mrIndex+1]=='S' or pattern[mrIndex+1]=='O') and (pattern[mrsIndex+1]=='P' or pattern[mrsIndex+1]=='S' or pattern[mrsIndex+1]=='O')):#e.g Mr. James, if index of Mr. is 0 then index of person James is 1 (i.e. 0+1)
                    if patternWords[mrIndex+1]==patternWords[mrsIndex+1] or patternWords[mrIndex+2]==patternWords[mrsIndex+1]: #Mr. Von Neumann and Mrs. Neumann
                        mrAndMrs = True
            if mrAndMrs == True:
                    triples.append([mrIndex, 'per:spouse' , mrsIndex,-1]) #each triple contains 4 items: subject index, relation, object index and relation index
                    triples.append([mrsIndex, 'per:spouse' , mrIndex,-1]) # the fourth item of the triple is relation index
                    if (mrsIndex in range(subjStart, subjEnd + 1) and mrIndex in range(
                            objStart, objEnd + 1)) or (mrIndex in range(subjStart,subjEnd + 1) and mrsIndex in range(objStart, objEnd + 1)):
                        relFound = True

            if mrAndMrs==False and len(listWordsFound)==0:
                relation = 'no_relation'
                print('\nExtracted Family Relation of Example No '+str(exampleNo+1)+' is: '+relation)
                outF.write(relation)
                outF.write('\n')
                continue

            # Rule 1: If any of s or o is pronoun, and both are found co-referents, then r = no_relation
            if (PoSs[subjStart]=="PRP$" or PoSs[subjStart]=="PRP" or PoSs[objStart]=='PRP$' or PoSs[objStart]=='PRP'):
                if coref==str(1) and \
                        not((subj.casefold() in map(str.casefold,malePronouns) and (obj.casefold() in map(str.casefold,femalePronouns) or obj.casefold() in map(str.casefold,selfPronouns)))or
                                (obj.casefold() in map(str.casefold,malePronouns) and (subj.casefold() in map(str.casefold,femalePronouns) or subj.casefold() in map(str.casefold,selfPronouns)))):#checking that subj and obj should not be in opposite gender groups
                    relation = 'no_relation'
                    print('\nExtracted Family Relation of Example No ' + str(exampleNo + 1) + ' is: ' + relation)
                    outF.write(relation)
                    outF.write('\n')
                    continue
            # else:
            #     print(exampleNo)
            #     outF.write('\n')

            # # Rule 2: This is not correct, so remove this rule.. If there is just one list word in between s and o, and no other person entity in btw, then s and o are related by the relation of list word
            # # may be i should modify this: each list word relates the closest persons (or person pronouns) if no punctuation occurs in between
            # if len(listWordsFoundInBtw)==1 and len(personsFoundInBtw)==0:
            #     relation = listWordsFoundInBtw[0]['rel']
            #     print(exampleNo+1, relation)
            #     outF.write(relation)
            #     outF.write('\n')
            #     continue
            # # else:
            # #     print(exampleNo)
            # #     outF.write('\n')
#check example 5

            # Rule 3: Select all persons, pronouns and list words from sentence and match which list word connects which persons/pronouns
                # If list word has nummod dependency, than that number many persons are related by that relation
                # heuristic: connect 1st two persons/pronouns with 1st list word, and so on ...


            # the ROOT of tree might also be important for us, check this in next experiment e.g.
            # example 1         Kaiser  <----    survived-by  ---->    wife
            # 11	Kaiser	SUBJECT	PERSON	_	_	NNP	PERSON	nsubjpass	13
            # 12	is	_	_	_	_	VBZ	O	auxpass	13
            # 13	survived	_	_	_	_	VBN	O	ROOT	0
            # 14	by	_	_	_	_	IN	O	case	16
            # 15	his	_	_	_	_	PRP$	O	nmod:poss	16
            # 16	wife	_	_	_	_	NN	O	nmod	13
            # pattern = "2ppp2"
            # removed them later.. had added in all regexs (?=[^\']) after w/r so that only those relation words are matched whose next character is not '
            rePerson1 = '((p|P|S+|O+)[^a-zA-Z0-9_&\(]*(\((P|S+|O+)?[^a-zA-Z0-9_&]*\))?)'  #person can be proper noun or pronoun # to handle P (P ) -- old rePerson = '((p|S+|O+)[^a-zA-Z0-9_&]*)'
            rePerson2 = '((P|S+|O+)[^a-zA-Z0-9_&\(]*(\((P|S+|O+)?[^a-zA-Z0-9_&]*\))?)'  #person can only be proper noun #to handle P (P ) -- old rePerson = '((p|S+|O+)[^a-zA-Z0-9_&]*)'
            reRelSingle = '((w|r)[^a-zA-Z0-9_&\)]*'+rePerson2+')' #(?=[^\']) #'(w\W*'+rePerson+')' # subj rel obj rel obj rel obj # the regex (?=...) Matches if ... matches next, but doesn’t consume any of the string. This is called a lookahead assertion. So in our case w(?=[^\']) matches only those w which are not immediately followed by '
            reAnd = '(&\W*' + rePerson2 + ')'
            reRelNumbered = '((?P<digit>\d)\W*(w|r)\W*' \
                            '((' + rePerson2 + '|' + reAnd + ')+))'  # reRelNumbered = '((?P<DIGIT>\d)\W*(w|r)\W*(('+rePerson+'&?){1,(?P=DIGIT)}))'#(?P<word>\b\w+)\s+(?P=word) #'(((\d\W*w)|r)\W*('+rePerson+')+)' # subj digit rel obj obj e.g 4 sons PPPP, or rels obj obj obj e.g. his sons A, B, and C.
            # digit = re.search(reRelNumbered, pattern).group('digit')
            # reRelNumbered = '('+reRelNumbered + '((' + rePerson + '|' + reAnd + '){1,' + digit + '}))'

            reRelPlural = '((r\W*('+rePerson2+')+)'+reAnd+')' #'(((\d\W*w)|r)\W*('+rePerson+')+)' # subj digit rel obj obj e.g 4 sons PPPP, or rels obj obj obj e.g. his sons A, B, and C.

            # reTripNumbered = '((' + rePerson + ')?(' + reAnd + ')?(((w|r)(?=[^\']))?\W*(' + reRelNumbered + '))+)'  # usual cases like subj rel obj(s) - the numbered relation case

            reTriple1 = '(('+rePerson1 + ')('+reAnd + ')?(((w|r))?\W*('+reRelNumbered+'|'+reRelPlural+'|'+reRelSingle+'))+)' #case 1 to 5 # usual cases like subj rel obj(s)
            reIndirect = '(' + rePerson1 + '[,\(&]\W*' + rePerson1 + '\'?\W*(w|r)\s*[,.\)])' #case 6: obj subj rel # A, B's son ... A & B, the brothers #
            reIndirect3 = '(' + rePerson1 + '&\W*' + rePerson1 + '\W*(w|r)\s*)'  # case 6: obj subj rel # A & B married #eg no 6
            reIndirect2 = '((w|r)\s*'+ rePerson1 + '[,\(]\W*' + rePerson1 + ')'#'[,.\)])' #case 7: rel subj obj # rel subj, obj, e.g. 912 wife of X, Y, ... I might need to invert the direction of this relation, as its direction is inverted once because of 'of'
            reIndirect22 = '((w|r)\s*'+ rePerson1 + '&\W*' + rePerson1 + ')' #case 7: rel subj obj # rel subj & obj e.g. couple X & Y, siblings x & Y etc.

            reTriples = []
            # print(re.sub(r'^(\d+)(p+).*', lambda m: m.group(1) + m.group(2)[:int(m.group(1))] if len(m.group(2)) >= int(
            #     m.group(1)) else '', pattern))
            # reTrip = '(?P<dig>\d)'


            print('Pattern: '+pattern)
            # reTriples.append(reTrip) # testing .. remove later
            reTriples.append(reTriple1) # 0
            reTriples.append(reIndirect) # 1
            reTriples.append(reIndirect2) # 2
            reTriples.append(reIndirect3)  # 3 # A & B married. eg no 6
            reTriples.append(reIndirect22) # 4 # couple A & B

            # re.escape()
            matchFoundTill = 0 #match is found till this index
            for tripleNo,reTriple in enumerate(reTriples):#tipleNo 0 is direct case, 1 is indirect1 case, 2 is indirect2 case
                print('RE Triple No: '+str(tripleNo))
                regex = re.compile(reTriple)#,re.IGNORECASE)
                matches = regex.finditer(pattern,pos=matchFoundTill)
                # matches = re.finditer(regex,pattern)#,re.IGNORECASE) #pattern is the pattern of symbols we produced from each example
                rel = -1
                for (matchNo,match) in enumerate(matches):
                    pronounIsPlural = False
                    triple2OnePerPassed = False
                    isRelPlural = False
                    obj2PredictedIndex = -1
                    print('Match No: '+str(matchNo)+'\tMatch: '+str(match))
                    # l = len(match.group())
                    # print('Match Groups: '+str(match.groups()))
                    # print('Match Group 1: '+match.group(1)) # same as print(match[1]) # its datatype is str
                    # print(match.group(2)) # same as print(match[2])
                    # print(match.group(3)) # same as print(match[3])
                    # print(match.group(4)) # same as print(match[4])
                    # print(match.group(5)) # same as print(match[5])
                    # print(match[0][0])
                    # print(match.start())
                    # print(match.end())

                    start = match.start()
                    matchFoundTill = match.end()
                    if match[0][0]=='p' or match[0][0]=='P' or match[0][0]=='S' or match[0][0]=='O':#checking the first character of matching string
                        subjPredictedIndex = match.start()
                        #following block is for the case of plural pronouns i.e. they, them, their
                        if match[0][0]=='p':
                            if tokens[subjPredictedIndex]=='they' or tokens[subjPredictedIndex]=='their' or tokens[subjPredictedIndex]=='them' or tokens[subjPredictedIndex]=='our':#if its a plural pronoun
                                # then it connects multiple subjects
                                if len(triples)>0:
                                    subjects = []
                                    pronounIsPlural = True
                                    lastRelAdded = triples.pop()#just in directly getting the last element,
                                    triples.append(lastRelAdded)# by popping and adding it back
                                    subjects.append(lastRelAdded[0])#add subject and object of lastRelAdded to the list of subjects
                                    subjects.append(lastRelAdded[2])#add subject and object of lastRelAdded to the list of subjects
                        start = start+1
                    else:
                        subjPredictedIndex = -2 #-2 represents this relation is without subject, it only has relation and object
                    theresAnEquivalentSubj = False
                    theresAnEquivalentObj = False
                    subjEquivalentPredictedIndex = -1
                    objEquivalentPredictedIndex = -1
                    no = 0
                    print('\nmatch index \t match character')
                    print('----------- \t ---------------')
                    for x in range(start,match.end()):
                        p = pattern[x]
                        print(str(x)+'\t\t\t\t\t'+p)
                        global objNoOfThisRel
                        if relationPredictedIndex != -1 and (p.lower()=='p' or (p=='S' and pattern[x-1]!='S') or (p=='O' and pattern[x-1]!='O')):
                            if (tripleNo!=2 and tripleNo!=4) or triple2OnePerPassed==True: #either its any pattern other than indirect2, or its indirect2's second person (or object)
                                if theresAnEquivalentObj==False:
                                    objPredictedIndex = x
                                    objNoOfThisRel +=1
                                else:
                                    objEquivalentPredictedIndex = x
                                    theresAnEquivalentObj = False
                            elif tripleNo==2 or tripleNo==4:
                                subjPredictedIndex = x #triple2's first person (ie.subject)
                                triple2OnePerPassed = True #means the next person that will be encountered in triple no 2 (indirect2 pattern) will be the second person (or object)
                        elif p=='w' or p=='r':
                            relationPredictedIndex = x
                            if objNoOfThisRel!=0:
                                no = 0
                            objNoOfThisRel = 0

                            if tokens[relationPredictedIndex-1]=='other':#if the relation word is preceded by 'other'
                                print('other')
                                temp = -1
                                temp1 = str.rfind(pattern,'O,',0,relationPredictedIndex-1)
                                if temp1==-1: temp1 = str.rfind(pattern,'O&',0,relationPredictedIndex-1)
                                temp2 = str.rfind(pattern,'S,',0,relationPredictedIndex-1)
                                if temp2 == -1: temp2 = str.rfind(pattern,'S&',0,relationPredictedIndex-1)
                                temp3 = str.rfind(pattern,'P,',0,relationPredictedIndex-1)
                                if temp3 == -1: temp3 = str.rfind(pattern,'P&',0,relationPredictedIndex-1)
                                temp4 = str.rfind(pattern,'p,',0,relationPredictedIndex-1)
                                if temp4 == -1: temp4 = str.rfind(pattern,'p&',0,relationPredictedIndex-1)
                                if temp1!=-1: temp = temp1
                                if temp2!=-1: temp = temp2
                                if temp3!=-1: temp = temp3
                                if temp4!=-1: temp = temp4
                                # if temp!=-1: #there is a 'person comma' preceding 'other relation' e.g 496 besides W, his other children X, Y, Z
                                obj2PredictedIndex = temp # -1 other wise
                        elif p.isdigit():
                            no = int(p)
                        elif (tripleNo==1 or tripleNo==3) and relationPredictedIndex == -1 and (p.lower()=='p' or (p=='S' and pattern[x-1]!='S') or (p=='O' and pattern[x-1]!='O')):
                            #for the case of indirect1 pattern:  obj, subj rel ... obj & subj, rel
                            objPredictedIndex = subjPredictedIndex
                            subjPredictedIndex = x
                        elif theresAnEquivalentSubj == True and relationPredictedIndex == -1 and (p.lower()=='p' or (p=='S' and pattern[x-1]!='S') or (p=='O' and pattern[x-1]!='O')):
                            #for the case P (P )
                            subjEquivalentPredictedIndex = x
                            theresAnEquivalentSubj = False
                        elif p=='(': #to handle P (P )
                            if x+1 < match.end():
                                nextP = pattern[x+1]
                                if nextP=='p' or nextP=='P' or nextP=='S' or nextP=='O':
                                    if objPredictedIndex==-1:
                                        theresAnEquivalentSubj = True
                                    else:
                                        theresAnEquivalentObj = True
                        elif p=='&' and tripleNo==0: #to handle subject p&p in direct case eg. X&Y 2 children Z&W, eg 1012
                            if x+1 < match.end():
                                nextP = pattern[x+1]
                                if nextP=='p' or nextP=='P' or nextP=='S' or nextP=='O':
                                    if objPredictedIndex==-1:
                                        theresAnEquivalentSubj = True

                        if subjPredictedIndex != -1 and relationPredictedIndex != -1 and (objPredictedIndex != -1 or objEquivalentPredictedIndex!=-1):
                            nextTok = ''
                            if relationPredictedIndex+1 < len(tokens) and tripleNo==0:#only for direct case
                                nextTok = tokens[relationPredictedIndex+1]
                            rel, isRelPlural = matchListWord(tokens[relationPredictedIndex],nextTok)
                            if rel!=-1:
                                size1 = len(triples)
                                if objPredictedIndex != -1 and subjPredictedIndex!=objPredictedIndex and not([subjPredictedIndex, rel, objPredictedIndex] in triples): #if subj obj not same, and relation not already in triples
                                    if subjPredictedIndex!=-1 and not(subjPredictedIndex in corefs[str(objPredictedIndex)]) and tokens[subjPredictedIndex]!=tokens[objPredictedIndex]:
                                        print('\tTriple found: ('+tokens[subjPredictedIndex]+','+rel+','+tokens[objPredictedIndex]+') at indices ('+str(subjPredictedIndex)+','+str(relationPredictedIndex)+','+str(objPredictedIndex)+')')
                                        if no==0 or (no>0 and objNoOfThisRel<=no):#either its not the numbered relation case, or if its the numbered relation case then objNoOfThisRel is less than d
                                            triples.append([subjPredictedIndex,rel,objPredictedIndex,relationPredictedIndex])
                                        else:
                                            print('NOT ADDING THIS RELATION, d RELATIONS ALREADY ADDED!')

                                        if (subjPredictedIndex in range(subjStart,subjEnd + 1) and objPredictedIndex in range(
                                                objStart, objEnd + 1)):
                                            relFound = True
                                if objEquivalentPredictedIndex != -1 and subjPredictedIndex!=objEquivalentPredictedIndex and not([subjPredictedIndex, rel, objEquivalentPredictedIndex] in triples): #if subj obj not same, and relation not already in triples
                                    # print(subjPredictedIndex,relationPredictedIndex,objEquivalentPredictedIndex)
                                    # print(rel)
                                    print('\tTriple found: (' + tokens[subjPredictedIndex] + ',' + rel + ',' + tokens[
                                        objEquivalentPredictedIndex] + ') at indices (' + str(
                                        subjPredictedIndex) + ',' + str(
                                        relationPredictedIndex) + ',' + str(objEquivalentPredictedIndex) + ')')
                                    triples.append([subjPredictedIndex,rel,objEquivalentPredictedIndex,relationPredictedIndex])
                                    if (subjPredictedIndex in range(subjStart,subjEnd + 1) and objEquivalentPredictedIndex in range(objStart, objEnd + 1)):
                                        relFound = True
                                if obj2PredictedIndex!=-1 and subjPredictedIndex!=obj2PredictedIndex and not([subjPredictedIndex, rel, obj2PredictedIndex] in triples): #if subj obj not same, and relation not already in triples
                                    # print(subjPredictedIndex, relationPredictedIndex, obj2PredictedIndex)
                                    # print(rel)
                                    print('\tTriple found: (' + tokens[subjPredictedIndex] + ',' + rel + ',' + tokens[
                                        obj2PredictedIndex] + ') at indices (' + str(
                                        subjPredictedIndex) + ',' + str(
                                        relationPredictedIndex) + ',' + str(obj2PredictedIndex) + ')')
                                    triples.append([subjPredictedIndex, rel, obj2PredictedIndex,relationPredictedIndex])
                                    if (subjPredictedIndex in range(subjStart,subjEnd + 1) and obj2PredictedIndex in range(objStart, objEnd + 1)):
                                        relFound = True
                                if subjEquivalentPredictedIndex != -1 and subjEquivalentPredictedIndex!=objPredictedIndex and not([subjEquivalentPredictedIndex, rel, objPredictedIndex] in triples): #if subj obj not same, and relation not already in triples
                                    # print(subjEquivalentPredictedIndex, relationPredictedIndex, objPredictedIndex)
                                    # print(rel)
                                    print('\tTriple found: (' + tokens[subjEquivalentPredictedIndex] + ',' + rel + ',' + tokens[
                                        objPredictedIndex] + ') at indices (' + str(
                                        subjEquivalentPredictedIndex) + ',' + str(
                                        relationPredictedIndex) + ',' + str(objPredictedIndex) + ')')
                                    if (tripleNo==3 and rel=='per:spouse') or tripleNo!=3:
                                        triples.append([subjEquivalentPredictedIndex, rel, objPredictedIndex,relationPredictedIndex])
                                        if (subjEquivalentPredictedIndex in range(subjStart,subjEnd + 1) and objPredictedIndex in range(objStart, objEnd + 1)):
                                            relFound = True
                                    else:
                                        print("TRIPLE 3, NOT SPOUSE RELATION")
                                        objPredictedIndex = -1
                                        objEquivalentPredictedIndex = -1
                                        continue
                                    if (subjEquivalentPredictedIndex in range(subjStart,subjEnd + 1) and objPredictedIndex in range(
                                            objStart, objEnd + 1)):
                                        relFound = True
                                    subjEquivalentPredictedIndex = -1
                                size2 = len(triples)
                                if pronounIsPlural==True:
                                    for tripleTemp in triples[size1:size2]:
                                        for subjTemp in subjects:
                                            if subjTemp!=tripleTemp[2]: #if subj and obj are not same
                                                newTriple = [subjTemp,tripleTemp[1],tripleTemp[2],relationPredictedIndex]
                                                if not(newTriple in triples): # if triple does not already exist
                                                    print(newTriple)
                                                    print(rel)
                                                    triples.append(newTriple)
                                                    if (subjTemp in range(subjStart,subjEnd + 1) and tripleTemp[2] in range(objStart, objEnd + 1)):
                                                        relFound = True

                                if not(size2 > size1):
                                    if len(corefs[str(relationPredictedIndex)])>0:
                                        if corefs[str(relationPredictedIndex)][0]!=subjPredictedIndex:
                                            if not(corefs[str(relationPredictedIndex)][0] in corefs[str(subjPredictedIndex)]):
                                                objPredictedIndex = corefs[str(relationPredictedIndex)][0]
                                                triples.append([subjPredictedIndex,rel,objPredictedIndex,relationPredictedIndex])
                                        if (subjPredictedIndex in range(subjStart,subjEnd + 1) and objPredictedIndex in range(
                                                objStart, objEnd + 1)):
                                            relFound = True
                                        size2 = len(triples)
                                if size2>size1:
                                    if rel == 'per:parents':  # also add the reverse relations
                                        reverseRel = 'per:children'
                                    elif rel == 'per:children':
                                        reverseRel = 'per:parents'
                                    elif rel == 'per:spouse' or rel == 'per:siblings' or rel == 'per:other_family': # if a spouse b, then also add b spouse a
                                        reverseRel = rel
                                    if objPredictedIndex != subjPredictedIndex and not ([objPredictedIndex, reverseRel,subjPredictedIndex] in triples):  # subj and obj are not same and the triple does not already exist
                                        triples.append([objPredictedIndex, reverseRel, subjPredictedIndex,relationPredictedIndex])
                                        if (objPredictedIndex in range(subjStart,subjEnd + 1) and subjPredictedIndex in range(
                                                objStart, objEnd + 1)):
                                            relFound = True
                                    if obj2PredictedIndex != -1:
                                        if obj2PredictedIndex != subjPredictedIndex and not ([obj2PredictedIndex, reverseRel,subjPredictedIndex] in triples):  # subj and obj are not same and the triple does not already exist
                                            triples.append([obj2PredictedIndex, reverseRel, subjPredictedIndex,relationPredictedIndex])
                                            if (obj2PredictedIndex in range(subjStart,subjEnd + 1) and subjPredictedIndex in range(
                                                    objStart, objEnd + 1)):
                                                relFound = True
                                    if (objPredictedIndex in range(subjStart, subjEnd + 1) and subjPredictedIndex in range(objStart, objEnd + 1)):
                                        relFound = True
                                        break
                                    if obj2PredictedIndex!=-1:
                                        if (obj2PredictedIndex in range(subjStart, subjEnd + 1) and subjPredictedIndex in range(objStart, objEnd + 1)):
                                            relFound = True
                                            break


                                objPredictedIndex = -1
                                objEquivalentPredictedIndex = -1
                    if relFound==True:
                        break
                if relFound == True:
                    break
                # if rel!=-1:
                #     break

            print('\nPrinting extracted relations')
            # print('\n'.join(' '.join(map(str, t)) for t in triples)) #Change this to loop and print indices as well as tokens of triple
            for t in triples:
                print('(' + tokens[t[0]] + ',' + t[1] + ',' + tokens[t[2]] + ') at indices ('
                      + str(t[0]) + ',' + str(t[3]) + ',' + str(t[2]) + ')')

            # for adding coreferent's relations
            triples = addCorefRels(triples, sentence, tokens, corefs,example)

            print('\nPrinting extended relations: added coref relations')
            # print('\n'.join(' '.join(map(str, t)) for t in triples))
            for t in triples:
                print('(' + tokens[t[0]] + ',' + t[1] + ',' + tokens[t[2]] + ') at indices ('
                      + str(t[0]) + ',' + str(t[3]) + ',' + str(t[2]) + ')')

            # for adding P (P) cases
            rePersonBracketed = rePerson1+'\('+rePerson2+'\)'
            regex2 = re.compile(rePersonBracketed)#, re.IGNORECASE)
            matches = re.finditer(regex2, pattern)
            for match in matches:
                bracketStarted = False
                if match[0][0] == 'p' or match[0][0] == 'P' or match[0][0] == 'S' or match[0][0] == 'O':  # checking the first character of matching string
                    index1 = match.start()
                    start = match.start() + 1
                for x in range(start, match.end()):
                    p = pattern[x]
                    if p=='(':
                        bracketStarted = True
                    if bracketStarted and (p=='p' or p=='P'or p=='S'or p=='O'):
                        for t in triples:
                            if t[0]==index1 and x!=t[2] and not([x,t[1],t[2],t[3]] in triples):# subj and obj are not same and the triple does not already exist
                                triples.append([x,t[1],t[2],t[3]])
                            elif t[2]==index1 and t[0]!=x and not([t[0], t[1], x,t[3]] in triples):
                                triples.append([t[0], t[1], x,t[3]])
                            elif t[0]==x and index1!=t[2] and not([index1,t[1],t[2],t[3]] in triples):
                                triples.append([index1,t[1],t[2],t[3]])
                            elif t[2]==x and t[0]!=index1 and not([t[0], t[1], index1,t[3]] in triples):
                                triples.append([t[0], t[1], index1,t[3]])
                            if (triples[len(triples)-1][0] in range(subjStart, subjEnd + 1) and triples[len(triples)-1][2] in range(
                                    objStart, objEnd + 1)):
                                relFound = True
                                break


            # make relations transitive, i.e. if a has spouse b and b has child c then a has child c
            triples = addTransitiveRels(triples,corefs,tokens,example)#noOfToks=len(tokens),sentence=sentence)

            print('\nPrinting extended relations: added transitive relations')
            # print('\n'.join(' '.join(map(str, t)) for t in triples))
            for t in triples:
                print('(' + tokens[t[0]] + ',' + t[1] + ',' + tokens[t[2]] + ') at indices ('
                      + str(t[0]) + ',' + str(t[3]) + ',' + str(t[2]) + ')')

            # for writing predicted relation to outFile
            # if a triple is found for the subject object in triples, then write its relation to file
            # else write no_relation to file
            for triple in triples:
                subjPredictedIndex = triple[0]
                objPredictedIndex = triple[2]
                rel = triple[1]
                if (subjPredictedIndex in range(subjStart,subjEnd+1) and objPredictedIndex in range(objStart,objEnd+1)):
                    relFound = True
                    #1
                    # # this if elif block checks if relation and object belong to opposite genders
                    # if triple[3] != -1:
                    #     if tokens[triple[3]] in maleRelations:
                    #         if tokens[objPredictedIndex] in femalePronouns:
                    #             rel = 'no_relation'
                    #     elif tokens[triple[3]] in femaleRelations:
                    #         if tokens[objPredictedIndex] in malePronouns:
                    #             rel = 'no_relation'
                    outF.write(rel)
                    outF.write('\n')
                    # print('Example No: ')
                    # print(exampleNo+1, rel)
                    # print('\n')
                    print('\nExtracted Family Relation of Example No ' + str(exampleNo + 1) + ' is: ' + rel)
                    break
            if relFound==False:
                # Rule 2: If there is just one list word in between s and o, and no other person entity in btw, then s and o are related by the relation of list word
                # may be i should modify this: each list word relates the closest persons (or person pronouns) if no punctuation occurs in between
                inBtwTxt =  TreebankWordDetokenizer().detokenize(tokens[s:e+1])

                if len(listWordsFoundInBtw) == 1 and personsFoundInBtw == 0:
                    # if tokens[subjStart]!=tokens[objStart]:#if text of subj and obj is not same
                    relation = listWordsFoundInBtw[0]['rel']
                    if relation!=-1 and relation!=-2:
                        if objStart < subjStart: # invert the relation if first object then subject
                            if relation == 'per:parents':
                                relation = 'per:children'
                            elif relation == 'per:children':
                                relation = 'per:parents'
                        print('Relation added from rule 2: one list word in between')
                        # print(exampleNo+1, relation)
                        print('\nExtracted Family Relation of Example No ' + str(exampleNo + 1) + ' is: ' + relation)
                        #2
                        # # this if elif block checks if relation and object belong to opposite genders
                        # if triple[3] != -1:
                        #     if tokens[triple[3]] in maleRelations:
                        #         if tokens[objPredictedIndex] in femalePronouns:
                        #             relation = 'no_relation'
                        #     elif tokens[triple[3]] in femaleRelations:
                        #         if tokens[objPredictedIndex] in malePronouns:
                        #             relation = 'no_relation'
                        outF.write(relation)
                        outF.write('\n')
                        continue
                outF.write('no_relation') #if still not found, then no_relation
                outF.write('\n')
                # print('Example No: ')
                # print(exampleNo+1)
                # print('\n')
                print('\nExtracted Family Relation of Example No ' + str(exampleNo + 1) + ' is: ' + 'no_relation')
    print()

def matchListWord(token,nextTok):
    ''' matches the passed token to the family relations' list words, returns the relation if matched, returns -1 otherwise '''
    invertingWords = ['of','for','to','by']#=4
    parentWords = ['parent', 'adopter','born','bornin','bornt','father', 'dad', 'daddy', 'papa','mama','mother', 'mum', 'mom', 'stepfather', 'step-father',
                   'stepmother', 'step-mother', 'step-parent', 'stepparent']#=15
    childrenWords = ['child', 'children','adopt','adopting','adopted','adoption','adoptive', 'kid', 'toddler', 'offspring', 'son', 'daughter', 'stepson', 'step-son', 'stepdaughter',
                     'step-daughter']#=17
    spouseWords = ['couple','ex','ex-couple','spouse','ex-spouse', 'husband','ex-husband', 'hubby', 'ex-hubby', 'widow','ex-widow',
                   'better-half','significant-other','wife','ex-wife', 'wives','ex-wives', 'wed','wedding','wedded', 'marry','marries','marrying','married','marriage','remarry','remarries','remarrying','remarried','life-partner', 'marriage-partner','newlywed',
                   'girlfriend','ex-girlfriend', 'boyfriend','ex-boyfriend', 'concubine','ex-concubine', 'lover','ex-lover','fiance','fiancee','betroth','betrothed', 'affianced', 'engage','engaging','engaged',
                   'love', 'ex-love','dating', 'common-law-spouse', 'cohabitant','ex-cohabitant', 'partner','ex-partner',
                   'consort','ex-consort', 'mistress','ex-mistress', 'domestic-partner', 'significant-other', 'enbyfriend','divorce','divorcing','divorced']#=61
    siblingWords = ['sibling', 'brother', 'sister', 'stepsister', 'step-sister', 'half-sister','halfsister','stepbrother',
                    'step-brother', 'half-brother','halfbrother','bro', 'sis', 'sib']#=8
    otherFamilyWords = ['relative', 'uncle', 'aunt', 'cousin', 'nephew', 'niece',
                        'father-in-law', 'fathers-in-law','mother-in-law', 'mothers-in-law', 'brother-in-law',
                        'brothers-in-law', 'sister-in-law', 'sisters-in-law', 'son-in-law', 'sons-in-law',
                        'daughter-in-law', 'daughters-in-law', 'grandparent', 'grand-parent','grandfather', 'grand-father',
                        'great-grandfather','grandmother', 'grand-mother', 'great-grandmother', 'grandchild', 'grand-child',
                        'great-grandchild', 'grandchildren', 'grand-children','great-grandchildren', 'grandson', 'grand-son',
                        'grandson', 'great-grandson','granddaughter', 'grand-daughter','great-granddaughter', 'co-husband',
                        'co-wife', 'co-wives', 'ancestor', 'descendant', 'lineal-descendant', 'collateral-descendant']#48 .. total = 149
    notFamilyWords = ['neighbor','neighbour','spokesman','escort','aide','spokeswoman','client','booster','companion','fellow','worker',
                      'mentor','lawyer','advisor','roommate','hanger-on','hangers-on','attorney','solicitor','friend'] #words that are used for relations other than family among persons
    #eg 1056 spokesman, 1073 escort, 1082 aide, 1125 spokeswoman, 1205 client,1225 booster,1254 companion, 1288 fellow, 1288 worker, 1306 mentor,
    rel = -1
    isRelPlural = False
    # have to do string match in lower case + without s
    if str.endswith(token, 's') and not(token in parentWords or token in childrenWords or token in spouseWords or token in siblingWords or token in otherFamilyWords):
        isRelPlural = True
        token = str.rstrip(token, 's')
    if str.find(token,'children')!=-1:
        isRelPlural = True
    token = str.lower(token)
    if token in parentWords:
        # if token in ['born', 'bornin', 'bornt']:
        #     if nextTok=='to' or nextTok=='of':
        #         rel = 'per:parents'
        # else:
        rel = 'per:parents'
    elif token in childrenWords:
        rel = 'per:children'
        if nextTok=='with' and objNoOfThisRel==1:
            rel = 'per:spouse'
    elif token in spouseWords:
        rel = 'per:spouse'
    elif token in siblingWords:
        rel = 'per:siblings'
    elif token in otherFamilyWords:
        if token=='family' and nextTok=='with': #ex 1010 she has 2 daughters with him, A and B. = she spouse him, she children A, she children B.
            rel = 'per:spouse'
        else:
            rel = 'per:other_family'
    elif token in notFamilyWords:
        rel = -2  #rel = 'no_relation'
    tokWords = re.split('[^a-zA-Z0-9]', token)#split on any non alphanumeric character, e.g. - / etc
    # tokWords = token.split('-')
    if rel==-1 and len(tokWords)>1:
        for word in tokWords:
            if word in parentWords:
                rel = 'per:parents'
                break
            elif word in childrenWords:
                rel = 'per:children'
                break
            elif word in spouseWords:
                rel = 'per:spouse'
                break
            elif word in siblingWords:
                rel = 'per:siblings'
                break
            elif word in otherFamilyWords:
                rel = 'per:other_family'
                break
    if rel!=-1 and nextTok in invertingWords:#if word next to relation is an inverting word, then invert the relation
        isRelPlural = True # eg. mother of A, B and C.
        if rel == 'per:parents':
            rel = 'per:children'
        elif rel == 'per:children':
            rel = 'per:parents'

    return rel,isRelPlural

def scoreRE(dataset,corrected,examples):
    print('Starting SCORING Relation Extraction...!!!!!!!!')

    if corrected==True:
        folder = "familyAllCorrected"
    else:
        folder = "familyAll"
    relationsOur = "outputTACRED\\"+dataset+"\\"+folder+"\\outputCustRE.txt"
    relationsKey = "outputTACRED\\"+dataset+"\\"+folder+"\\relationsKey.txt"

    correctNo = 0
    inCorrectNo = 0
    notWorkedNo = 0
    if examples=='both':
        with open(relationsOur, 'r', encoding='utf-8', errors='ignore') as relsOur, open(relationsKey, 'r', encoding='utf-8', errors='ignore') as relsKey:
            for relationOur,relationKey in zip(relsOur, relsKey):
                if relationOur==relationKey:
                    correctNo+=1
                elif relationOur!='\n':
                    inCorrectNo+=1
                else:
                    notWorkedNo+=1
    elif examples=='positive':
        with open(relationsOur, 'r', encoding='utf-8', errors='ignore') as relsOur, open(relationsKey, 'r', encoding='utf-8', errors='ignore') as relsKey:
            for relationOur,relationKey in zip(relsOur, relsKey):
                if relationOur==relationKey:
                    correctNo+=1
                elif relationOur!='\n':
                    inCorrectNo+=1
                else:
                    notWorkedNo+=1
    elif examples == 'negative':
        with open(relationsOur, 'r', encoding='utf-8', errors='ignore') as relsOur:
            for relationOur in relsOur:
                if relationOur == 'no_relation\n':
                    correctNo += 1
                elif relationOur != '\n':
                    inCorrectNo += 1
                else:
                    notWorkedNo += 1
    print("No of correct: ",correctNo)
    print("No of in-correct: ",inCorrectNo)
    print("No not worked on yet: ",notWorkedNo)

#wordToNum function converts the passed number word to number digit e.g. two to 2
def wordToNum(num):
    ''' This function converts the passed number word to number digit e.g. two to 2 '''
    num = num.lower();
    dict_w = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
              'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15, 'sixteen': 16,
              'seventeen': '17', 'eighteen': '18', 'nineteen': '19'}
    mydict2 = ['', '', 'twenty', 'thirty', 'fourty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninty']
    divide = num[num.find("ty") + 2:]
    if num:
        if (num in dict_w.keys()):
            return str(dict_w[num])
        elif divide == '':
            for i in range(0, len(mydict2) - 1):
                if mydict2[i] == num:
                    return str(i * 10)
        else:
            str3 = 0
            str1 = num[num.find("ty") + 2:]
            str2 = num[:-len(str1)]
            for i in range(0, len(mydict2)):
                if mydict2[i] == str2:
                    str3 = i;
            if str2 not in mydict2:
                print("----->Invalid Input<-----")
            else:
                try:
                    return str((str3 * 10) + dict_w[str1])
                except:
                    print("----->Invalid Input<-----")
            return -1

#this functions adds transitive relations to the passed list of triples
def addTransitiveRels(triples,corefs,tokens,example):
    ''' make relations transitive, i.e. if a has spouse b and b has child c then a has child c '''
    additionalTriples = []
    corefIndicesAll = corefs  # {}#a dict of lists, one list for each tokenIndex
    # for triple in triples: ##this decreases F1 score
    #     # this if elif block checks if relation and object belong to opposite genders
    #     if tokens[triple[3]] in maleRelations:
    #         if tokens[triple[2]] in femalePronouns:
    #             triple[1] = 'no_relation'
    #     elif tokens[triple[3]] in femaleRelations:
    #         if tokens[triple[2]] in malePronouns:
    #             triple[1] = 'no_relation'
    for triple in triples:
        if triple[1] == 'per:spouse':  # if relation is spouse
            a = triple[0]  # a spouse b
            b = triple[2]
            for triple2 in triples:
                if triple2[1] == 'per:children':
                    c = triple2[0]  # c child d
                    d = triple2[2]
                    if a == c:
                        additionalTriples.append([b, 'per:children', d,-1])  # then also b child d
                        additionalTriples.append([d, 'per:parents', b,-1])  # also add the reverse relation
                    elif b == c:
                        additionalTriples.append([a, 'per:children', d,-1])  # then also a child d
                        additionalTriples.append([d, 'per:parents', a,-1])  # also add the reverse relation
                elif triple2[1] == 'per:parents':
                    c = triple2[0]  # c parent d
                    d = triple2[2]
                    if a == c:
                        additionalTriples.append([b, 'per:other_family', d,-1])  # then also b other_family d
                        additionalTriples.append([d, 'per:other_family', b,-1])  # also add the reverse relation
                    elif b == c:
                        additionalTriples.append([a, 'per:other_family', d,-1])  # then also a other_family d
                        additionalTriples.append([d, 'per:other_family', a,-1])  # also add the reverse relation
                # elif triple2[1] == 'per:spouse':
                #     c = triple2[0]  # c spouse d
                #     d = triple2[2]
                #     if a == c:
                #         additionalTriples.append([b, 'per:spouse', d,-1])  # then also b spouse d
                #         additionalTriples.append([d, 'per:spouse', b,-1])  # also add the reverse relation
                #     elif b == c:
                #         additionalTriples.append([a, 'per:spouse', d,-1])  # then also a spouse d
                #         additionalTriples.append([d, 'per:spouse', a,-1])  # also add the reverse relation
                elif triple2[1] == 'per:siblings':
                    c = triple2[0]  # c sibling d
                    d = triple2[2]
                    if a == c:
                        additionalTriples.append([b, 'per:other_family', d,-1])  # then also b other_family d
                        additionalTriples.append([d, 'per:other_family', b,-1])  # also add the reverse relation
                    elif b == c:
                        additionalTriples.append([a, 'per:other_family', d,-1])  # then also a other_family d
                        additionalTriples.append([d, 'per:other_family', a,-1])  # also add the reverse relation
        elif triple[1] == 'per:children':  # if relation is children
            a = triple[0]  # a child b
            b = triple[2]
            for triple2 in triples:
                if triple2[1] == 'per:spouse':
                    c = triple2[0]  # c spouse d
                    d = triple2[2]
                    if a == c:
                        additionalTriples.append([b, 'per:parents', d,-1])  # then also b parent d
                        additionalTriples.append([d, 'per:children', b,-1])  # also add the reverse relation
                    elif b == c:
                        additionalTriples.append([a, 'per:other_family', d,-1])  # then also a other_family d
                        additionalTriples.append([d, 'per:other_family', a,-1])  # also add the reverse relation
                elif triple2[1] == 'per:children':
                    c = triple2[0]  # c child d
                    d = triple2[2]
                    if a == c and b!=d and d>-1 and not(b in corefIndicesAll[str(d)]):
                        additionalTriples.append([b, 'per:siblings', d,-1])  # then also b siblings d
                        additionalTriples.append([d, 'per:siblings', b,-1])  # also add the reverse relation
                    elif b == c:
                        additionalTriples.append([a, 'per:other_family', d,-1])  # then also a other_family d
                        additionalTriples.append([d, 'per:other_family', a,-1])  # also add the reverse relation
                elif triple2[1] == 'per:parents':
                    c = triple2[0]  # c parent d
                    d = triple2[2]
                    if a == c:
                        additionalTriples.append([b, 'per:other_family', d,-1])  # then also b other_family d
                        additionalTriples.append([d, 'per:other_family', b,-1])  # also add the reverse relation
                    elif b == c and a!=d and tokens[a]!=tokens[d] and d>-1 and not(a in corefIndicesAll[str(d)]) and \
                            not(tokens[b].casefold() in map(str.casefold,malePronouns) and tokens[d].casefold() in map(str.casefold,malePronouns)) and not(tokens[b].casefold() in map(str.casefold,femalePronouns) and tokens[d].casefold() in map(str.casefold,femalePronouns)):#the text of subj and obj is not same  and \
                        additionalTriples.append([a, 'per:spouse', d,-1])  # then also a spouse d
                        additionalTriples.append([d, 'per:spouse', a,-1])  # also add the reverse relation
                elif triple2[1] == 'per:siblings':
                    c = triple2[0]  # c sibling d
                    d = triple2[2]
                    if a == c:
                        additionalTriples.append([b, 'per:other_family', d,-1])  # then also b other_family d
                        additionalTriples.append([d, 'per:other_family', b,-1])  # also add the reverse relation
                    elif b == c:
                        additionalTriples.append([a, 'per:children', d,-1])  # then also a child d
                        additionalTriples.append([d, 'per:parents', a,-1])  # also add the reverse relation
        elif triple[1] == 'per:parents':  # if relation is parents
            a = triple[0]  # a parent b
            b = triple[2]
            for triple2 in triples:
                if triple2[1] == 'per:spouse':
                    c = triple2[0]  # c spouse d
                    d = triple2[2]
                    if a == c:
                        additionalTriples.append([b, 'per:other_family', d,-1])  # then also b other_family d
                        additionalTriples.append([d, 'per:other_family', b,-1])  # also add the reverse relation
                    elif b == c:
                        additionalTriples.append([a, 'per:parents', d,-1])  # then also a parents d
                        additionalTriples.append([d, 'per:children', a,-1])  # also add the reverse relation
                elif triple2[1] == 'per:children':
                    c = triple2[0]  # c child d
                    d = triple2[2]
                    if a == c:
                        additionalTriples.append([b, 'per:other_family', d,-1])  # then also b other_family d
                        additionalTriples.append([d, 'per:other_family', b,-1])  # also add the reverse relation
                    elif b == c and a!=d and d>-1 and not(a in corefIndicesAll[str(d)]):
                        additionalTriples.append([a, 'per:siblings', d,-1])  # then also a siblings d
                        additionalTriples.append([d, 'per:siblings', a,-1])  # also add the reverse relation
                elif triple2[1] == 'per:parents':
                    c = triple2[0]  # c parent d
                    d = triple2[2]
                    if a == c and b!=d and d>-1 and tokens[b]!=tokens[d] and not(b in corefIndicesAll[str(d)]) and \
                            not(tokens[b].casefold() in map(str.casefold,malePronouns) and tokens[d].casefold() in map(str.casefold,malePronouns)) and not(tokens[b].casefold() in map(str.casefold,femalePronouns) and tokens[d].casefold() in map(str.casefold,femalePronouns)):
                        additionalTriples.append([b, 'per:spouse', d,-1])  # then also b spouse d
                        additionalTriples.append([d, 'per:spouse', b,-1])  # also add the reverse relation
                    elif b == c:
                        additionalTriples.append([a, 'per:other_family', d,-1])  # then also a other_family d
                        additionalTriples.append([d, 'per:other_family', a,-1])  # also add the reverse relation
                elif triple2[1] == 'per:siblings':
                    c = triple2[0]  # c sibling d
                    d = triple2[2]
                    if a == c:
                        additionalTriples.append([b, 'per:children', d,-1])  # then also b children d
                        additionalTriples.append([d, 'per:parents', b,-1])  # also add the reverse relation
                    elif b == c:
                        additionalTriples.append([a, 'per:other_family', d,-1])  # then also a other_family d
                        additionalTriples.append([d, 'per:other_family', a,-1])  # also add the reverse relation
        elif triple[1] == 'per:siblings':  # if relation is siblings
            a = triple[0]  # a siblings b
            b = triple[2]
            for triple2 in triples:
                if triple2[1] == 'per:spouse':
                    c = triple2[0]  # c spouse d
                    d = triple2[2]
                    if a == c:
                        additionalTriples.append([b, 'per:other_family', d,-1])  # then also b other_family d
                        additionalTriples.append([d, 'per:other_family', b,-1])  # also add the reverse relation
                    elif b == c:
                        additionalTriples.append([a, 'per:other_family', d,-1])  # then also a other_family d
                        additionalTriples.append([d, 'per:other_family', a,-1])  # also add the reverse relation
                elif triple2[1] == 'per:children':
                    c = triple2[0]  # c child d
                    d = triple2[2]
                    if a == c:
                        additionalTriples.append([b, 'per:other_family', d,-1])  # then also b other_family d
                        additionalTriples.append([d, 'per:other_family', b,-1])  # also add the reverse relation
                    elif b == c:
                        additionalTriples.append([a, 'per:other_family', d,-1])  # then also a other_family d
                        additionalTriples.append([d, 'per:other_family', a,-1])  # also add the reverse relation
                elif triple2[1] == 'per:parents':
                    c = triple2[0]  # c parent d
                    d = triple2[2]
                    if a == c:
                        additionalTriples.append([b, 'per:parents', d,-1])  # then also b parents d
                        additionalTriples.append([d, 'per:children', b,-1])  # also add the reverse relation
                    elif b == c:
                        additionalTriples.append([a, 'per:parents', d,-1])  # then also a parents d
                        additionalTriples.append([d, 'per:children', a,-1])  # also add the reverse relation
                elif triple2[1] == 'per:siblings':
                    c = triple2[0]  # c sibling d
                    d = triple2[2]
                    if a == c and b!=d and d>-1 and not(b in corefIndicesAll[str(d)]):
                        additionalTriples.append([b, 'per:siblings', d,-1])  # then also b siblings d
                        additionalTriples.append([d, 'per:siblings', b,-1])  # also add the reverse relation
                    elif b == c and a!=d and d>-1 and not(a in corefIndicesAll[str(d)]):
                        additionalTriples.append([a, 'per:siblings', d,-1])  # then also a siblings d
                        additionalTriples.append([d, 'per:siblings', a,-1])  # also add the reverse relation
    for t in additionalTriples:
        if t[0]!=t[2] and not(t in triples): #if ((t[0]!=tri[0] and t[2]!=tri[2]) for tri in triples):
                triples.append(t)
    return triples

#this functions writes the corefs and iscoref files
def createCorefsFiles(data,sentences,corefsFile,isCorefFile):
    corefIndicesAllExampels = []
    sentencesList = sentences.split('\n')
    with open(isCorefFile,'a',encoding='utf-8',errors='ignore') as outF1:#, open(corefsFile,'r', encoding='utf-8', errors='ignore') as outF2: # This is for writing isCoref.txt only by reading from corefs.json
        # corefIndicesAllExampels = json.load(outF2) ## This is for writing isCoref.txt only by reading from corefs.json

        outF1.truncate(0)
        prevCorefIndicesAll = []  # corefIndicesAll of previous example
        for exampleNo,(example,sentence) in enumerate(zip(data,sentencesList)):
            # if exampleNo+1 < 1093: continue
            print('\nEXAMPLE NO: '+str(exampleNo+1))
            corefIndicesAll = {}#a dict of lists, one list for each tokenIndex
            i = 0
            tokens = example['token']

            while i < len(example['token']):#loop populates corefIndicesAll dict with empty lists for each tokenIndex
                corefIndices = []#getCorefs(i, sentence)  # coref indices of i
                corefIndicesAll[i] = corefIndices
                i = i+1

            if exampleNo>0 and sentence==sentencesList[exampleNo-1]:#if the text is not the same as previous example's text.. this  is just to save time
                corefIndicesAll = prevCorefIndicesAll
            else:
                corefIndicesAll = populateCorefs(sentence, corefIndicesAll, tokens)  # coref indices of this example
            if len(corefIndicesAll)!=len(tokens):
                print('ERROR!!!!!!!!!!')
            # # following code block updates corefs by breaking corefIndicesAll into male, female, self and other groups
            # corefIndicesAllBefore = copy.deepcopy(corefIndicesAll)#shallow copy
            # print(corefIndicesAll)
            # updatedCorefs = copy.deepcopy(corefIndicesAll)
            # for c,cluster in enumerate(corefIndicesAll):
            #     maleCorefs = []
            #     femaleCorefs = []
            #     selfCorefs = []
            #     otherCorefs = []
            #     for mention in cluster:
            #         if tokens[int(mention['start'])].casefold() in malePronouns:
            #             maleCorefs.append(mention)
            #         elif tokens[int(mention['start'])].casefold() in femalePronouns:
            #             femaleCorefs.append(mention)
            #         elif tokens[int(mention['start'])].casefold() in map(str.casefold, selfPronouns):
            #             selfCorefs.append(mention)
            #         else:
            #             otherCorefs.append(mention)
            #     if ((len(maleCorefs)>0 and len(femaleCorefs)>0) or (len(maleCorefs)>0 and len(selfCorefs)>0) or (len(femaleCorefs)>0 and len(selfCorefs)>0)):#if atleast 2 different groups in corefs
            #         updatedCorefs.remove(cluster)
            #         if len(maleCorefs)>0:
            #             updatedCorefs.append(maleCorefs)
            #         if len(femaleCorefs)>0:
            #             updatedCorefs.append(femaleCorefs)
            #         if len(selfCorefs)>0:
            #             updatedCorefs.append(selfCorefs)
            #         if len(otherCorefs)>0:
            #             updatedCorefs.append(otherCorefs)
            # corefIndicesAll = updatedCorefs.copy()
            # corefIndicesAllAfter = copy.deepcopy(corefIndicesAll)  #  copy
            # if corefIndicesAllAfter!=corefIndicesAllBefore:
            #     print("Coreferences list is modified by breaking corefs that belong to different pronoun groups!!!!")
            #     print(corefIndicesAll)

            # following code block updates corefs by removing any coref that belongs to different group male/female/self
            # print(corefIndicesAll)
            corefIndicesAllBefore = copy.deepcopy(corefIndicesAll)
            for c in corefIndicesAll:
                updatedCorefs = list.copy(corefIndicesAll[c])
                # print(c)
                # print(corefs[c])
                for cx in corefIndicesAll[c]:
                    # print(cx)
                    if tokens[int(c)].casefold() in map(str.casefold, malePronouns):
                        if tokens[cx].casefold() in map(str.casefold, femalePronouns) or tokens[cx].casefold() in map(str.casefold, selfPronouns):
                            list.remove(updatedCorefs, cx)
                    elif tokens[int(c)].casefold() in map(str.casefold, femalePronouns):
                        if tokens[cx].casefold() in map(str.casefold, malePronouns) or tokens[cx].casefold() in map(
                                str.casefold, selfPronouns):
                            list.remove(updatedCorefs, cx)
                    elif tokens[int(c)].casefold() in map(str.casefold, selfPronouns):
                        if tokens[cx].casefold() in map(str.casefold, femalePronouns) or tokens[
                            cx].casefold() in map(str.casefold, malePronouns):
                            list.remove(updatedCorefs, cx)
                corefIndicesAll[c] = updatedCorefs.copy()
            corefIndicesAllAfter = copy.deepcopy(corefIndicesAll)  # copy
            if corefIndicesAllAfter!=corefIndicesAllBefore:
                print("Coreferences list is modified by removing corefs that belong to different pronoun groups!!!!")
                print(corefIndicesAll)
            # print(corefIndicesAll)
            # this block did more bad than good
            # #the following block is to check if a person name appears more than once in a text, then add those to corefs lists
            # corefIndicesAllBefore = copy.deepcopy(corefIndicesAll)  #  copy
            # ners = example['stanford_ner']
            # personsList = []#this will be list of dicts, where each dict contains tokens and indices of one person occurence
            # personContinued = False
            # for i,ner in enumerate(ners):
            #     if ner!='PERSON':
            #         personContinued = False
            #     elif ner=='PERSON':
            #         if personContinued==False:
            #             personsList.append({'tokens':[],'indices':[]})
            #         personsList[len(personsList)-1]['tokens'].append(tokens[i]) #append the token
            #         personsList[len(personsList)-1]['indices'].append(i) #append the token's index
            #         personContinued = True
            # if len(personsList)>1:
            #     for p1,person1 in enumerate(personsList):
            #         word = person1['tokens'][0]
            #         for p2,person2 in enumerate(personsList):
            #             if p1!=p2:
            #                 for i,pI in enumerate(person1['tokens']):
            #                     if i==0: per1=pI
            #                     else: per1 = per1 +" "+pI
            #                 for j,pJ in enumerate(person2['tokens']):
            #                     if j==0: per2=pJ
            #                     else: per2 = per2 +" "+pJ
            #                 if word in per2:
            #                     if per1.find(per2)!=-1 or per2.find(per1)!=-1:#this is to check that either person1 is substring of person2 or vice versa, so that cases such as Mr and Mrs X are eliminated
            #                         if not(personsList[p2]['indices'][0] in corefIndicesAll[personsList[p1]['indices'][0]]):
            #                             corefIndicesAll[personsList[p1]['indices'][0]].append(personsList[p2]['indices'][0])
            #                             for c1 in corefIndicesAll[personsList[p1]['indices'][0]]:
            #                                 if c1!=personsList[p2]['indices'][0]:
            #                                     for c1i in corefIndicesAll[c1]:
            #                                         if not personsList[p2]['indices'][0] == c1i:
            #                                             if not(personsList[p2]['indices'][0] in corefIndicesAll[c1i]):
            #                                                 corefIndicesAll[c1i].append(personsList[p2]['indices'][0])
            #                                             if not(c1i in corefIndicesAll[personsList[p2]['indices'][0]]):
            #                                                 corefIndicesAll[personsList[p2]['indices'][0]].append(c1i)
            #                         if not(personsList[p1]['indices'][0] in corefIndicesAll[personsList[p2]['indices'][0]]):
            #                             corefIndicesAll[personsList[p2]['indices'][0]].append(personsList[p1]['indices'][0])
            #                             for c2 in corefIndicesAll[personsList[p2]['indices'][0]]:
            #                                 if c2 != personsList[p1]['indices'][0]:
            #                                     for c2i in corefIndicesAll[c2]:
            #                                         if not personsList[p1]['indices'][0]==c2i:
            #                                             if not(personsList[p1]['indices'][0] in corefIndicesAll[c2i]):
            #                                                 corefIndicesAll[c2i].append(personsList[p1]['indices'][0])
            #                                             if not(c2i in corefIndicesAll[personsList[p1]['indices'][0]]):
            #                                                 corefIndicesAll[personsList[p1]['indices'][0]].append(c2i)
            # corefIndicesAllAfter = copy.deepcopy(corefIndicesAll) #  copy
            # if corefIndicesAllAfter!=corefIndicesAllBefore:
            #     print("Coreferences list is modified by adding person names multiple occurences!!!!")
            #     print(corefIndicesAll)

            corefIndicesAllExampels.append(corefIndicesAll)

            # corefIndicesAll=corefIndicesAllExampels[exampleNo] # This line is for writing isCoref.txt only by reading from corefs.json
            subjStart = example['subj_start']
            subjEnd = example['subj_end']
            objStart = example['obj_start']
            objEnd = example['obj_end']
            if type(subjStart) is dict:
                subjStart = int(subjStart['$numberInt'])
                subjEnd = int(subjEnd['$numberInt'])
                objStart = int(objStart['$numberInt'])
                objEnd = int(objEnd['$numberInt'])
            if subjStart in corefIndicesAll[objStart]:#if areCorefs(subjStart, subjEnd, objStart, objEnd, example, corefIndicesAll)==True:
                outF1.write('1\n')
            else:
                outF1.write('0\n')

            prevCorefIndicesAll = corefIndicesAll

    with open(corefsFile,'w', encoding='utf-8', errors='ignore') as outF2:
        outF2.truncate(0)
        jsonData = json.dump(corefIndicesAllExampels,outF2)#indent=0 is to add newlines between items when printing on file

# #this function checks if the subj and obj are coreferences, returns True if yes, False otherwise
# def areCorefs(subjStart,subjEnd,objStart,objEnd,example,corefIndicesAll):#corefIndicesAll of this example, and the example as json are passed to the function
#     subjNer = example['stanford_ner'][subjStart]
#     objNer = example['stanford_ner'][objStart]
#     bothInCorefs = False
#     for cluster in corefIndicesAll:
#         subjInCorefs = False
#         objInCorefs = False
#         for mention in cluster:
#             # if subjNer == 'PERSON':
#             #     if len(range(max(subjStart, mention['start']),
#             #                  min(subjEnd + 1, mention['end']))) > 0:  # to check overlap of subject and mention
#             #         subjInCorefs = True
#             # elif subjNer != 'PERSON':  # if subj is person pronoun
#             if ((subjStart == mention['start'])):# and (
#                     #subjEnd + 1 == mention['end'])):  # to check subject exactly matches mention
#                 subjInCorefs = True
#             # if objNer == 'PERSON':
#             #     if len(range(max(objStart, mention['start']),
#             #                  min(objEnd + 1, mention['end']))) > 0:  # to check overlap of object and mention
#             #         objInCorefs = True
#             # elif objNer != 'PERSON':  # if obj is person pronoun
#             if ((objStart == mention['start'])):# and (
#                     #objEnd + 1 == mention['end'])):  # to check object exactly matches mention
#                 objInCorefs = True
#         if subjInCorefs and objInCorefs:
#             bothInCorefs = True
#     return bothInCorefs
#
# #this function returns as a list, starting indices of all the corefs of the passed startIndex
# def getCorefs(startIndex,example,corefIndicesAll):#corefIndicesAll of this example, and the example as json are passed to the function
#     corefs = []
#     for cluster in corefIndicesAll:
#         for mention in cluster:
#             if startIndex==mention['start']:#if the start index matches in any coref cluster: Case 1: exact match, can be pronoun or person
#                 if ((example['stanford_ner'][startIndex]=='PERSON')or(mention['end']==mention['start']+1)):#either its a PERSON proper noun or its lenght is one (as person pronouns have length one)
#                     #then all other mentions in this cluster are corefs of the passed startIndex, so add start indices of all other mentions to the corefs list
#                     for coref in cluster:
#                         if coref['start']!=startIndex:
#                             if len(coref)>1:
#                                 corefNers = example['stanford_ner'][coref['start']:coref['end']]
#                                 if 'PERSON' in corefNers:
#                                     personAt = list.index(corefNers,'PERSON')
#                                     corefs.append(personAt)
#                             elif len(coref)==1:
#                                 corefs.append(coref['start'])
#                     return corefs
#             # elif startIndex in range(mention['start'],mention['end']):  # if the start index matches in any coref cluster: Case 2: not exact match, can be person only
#             #     if ((example['stanford_ner'][startIndex]=='PERSON')):# its a PERSON proper noun
#             #         #then all other mentions in this cluster are corefs of the passed startIndex, so add start indices of all other mentions to the corefs list
#             #         for coref in cluster:
#             #             if not(coref['start'] in range(mention['start'],mention['end'])):
#             #                 if len(coref)>1:
#             #                     corefNers = example['stanford_ner'][coref['start']:coref['end']]
#             #                     if 'PERSON' in corefNers:
#             #                         personAt = list.index(corefNers,'PERSON')
#             #                         corefs.append(personAt)
#             #                 elif len(coref)==1:
#             #                     corefs.append(coref['start'])
#             #         return corefs
#     return corefs

#this functions adds coreferents' relations to the passed list of triples
def addCorefRels(triples,sentence,tokens,corefs,example):
    ''' adds coreferents' relations to the passed list of triples '''
    print("Inside addCorefRels function...")
    additionalTriples = []
    corefIndices = []
    for triple in triples:
        # # this if elif block checks if relation and object belong to opposite genders
        # if sentence[triple[3]] in maleRelations:
        #     if sentence[triple[2]] in femalePronouns:
        #         triple[1] = 'no_relation'
        # elif sentence[triple[3]] in femaleRelations:
        #     if sentence[triple[2]] in malePronouns:
        #         triple[1] = 'no_relation'
        sIndex = triple[0]
        oIndex = triple[2]
        subj = example['token'][sIndex]#sentence[sIndex]
        obj = example['token'][oIndex]#sentence[oIndex]
        rel = triple[1]
        if sIndex>-1:
            corefIndices = corefs[str(sIndex)]#getCorefs(sIndex,example,corefs) #coref indices of subj
        for i in corefIndices:
            if i!=oIndex and not([i, rel, oIndex] in triples):# subj and obj are not same and the triple does not already exist
                if not((subj.casefold() in map(str.casefold,malePronouns) and (tokens[i].casefold() in map(str.casefold,femalePronouns) or tokens[i].casefold() in map(str.casefold,selfPronouns) ))or(subj.casefold() in map(str.casefold,femalePronouns) and (tokens[i].casefold() in map(str.casefold,malePronouns) or tokens[i].casefold() in map(str.casefold,selfPronouns))) or (subj.casefold() in map(str.casefold,selfPronouns) and (tokens[i].casefold() in map(str.casefold,femalePronouns) or tokens[i].casefold() in map(str.casefold,malePronouns) ))):#checking that subj and its coref should not be in opposite gender groups
                    additionalTriples.append([i, rel, oIndex,-1])  # add relation for each coref
        if oIndex > -1:
            corefIndices = corefs[str(oIndex)]#getCorefs(oIndex,example,corefs) # coref indices of obj
        for i in corefIndices:
            if sIndex!= i and not ([sIndex, rel, i] in triples):  # subj and obj are not same and the triple does not already exist
                if not ((obj.casefold() in map(str.casefold,malePronouns) and tokens[i].casefold() in map(str.casefold,femalePronouns)) or (obj.casefold() in map(str.casefold,femalePronouns) and tokens[i].casefold() in map(str.casefold,malePronouns))):#checking that obj and its coref should not be in opposite gender groups
                    additionalTriples.append([sIndex, rel, i,-1])  # add relation for each coref
    for t in additionalTriples:
        if t[0]!=t[2] and not(t in triples): # subj and obj are not same and the triple does not already exist
                triples.append(t)
    # triples = triples + additionalTriples
    return triples

def insertCharInString(theString,theCharToInsert,position):
    temp = list(theString)
    temp[position] = theCharToInsert
    theString = "".join(temp)
    return theString