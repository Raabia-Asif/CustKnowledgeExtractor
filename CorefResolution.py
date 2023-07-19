# Load your usual SpaCy model (one of SpaCy English models)
import spacy
import neuralcoref

def populateCorefs(sentence, corefIndicesAll,sentenceTokens):  # populate coref indices of this example
    # nlp = spacy.load('en')
    nlp = spacy.load('en_core_web_lg')
    neuralcoref.add_to_pipe(nlp)  # Add neural coref to SpaCy's pipe

    # You're done. You can now use NeuralCoref as you usually manipulate a SpaCy document annotations.
    # sentence = 'Her father was Joseph P. Kennedy, Sr., and her mother was Rose Kennedy.'
    # sentenceTokens = ['His', 'brother', 'was', 'the', 'writer', 'Aldous', 'Huxley',';', 'his', 'father', 'was', 'writer', 'and', 'editor', 'Leonard', 'Huxley',';', 'his', 'paternal', 'grandfather', 'was', 'biologist', 'T.', 'H.', 'Huxley',',', 'famous', 'as', 'a', 'colleague', 'and', 'supporter', 'of', 'Charles', 'Darwin', '.']
    # doc = nlp(sentence)
    nlp.tokenizer = nlp.tokenizer.tokens_from_list #because of this, nlp will now take as input tokens instead of text. this is to ensure that tokens indices by spacy and stanford are same
    doc = nlp(sentenceTokens)

    if len(doc)!=len(sentenceTokens):
        print("WARNING!! Length of input tokens not the same as length of spacy tokens")
        print(len(doc))
        exit()
    if doc._.has_coref:
        clusters = doc._.coref_clusters #All the clusters of corefering mentions in the doc
        corefLists = [] #MY MADE: holds lists of coref clusters where each cluster is a list of mentions, each mention is a dict and contains start and end

        #this loop populates the corefLists
        for cluster in clusters:
            corefs = [] #MY MADE: each cluster is a list of mentions, each mention is a dict and contains start and end
            for mention in cluster:
                coref = {} #MY MADE: each mention is a dict and contains start and end
                coref['start'] = mention.start
                coref['end'] = mention.end
                corefs.append(coref)
            corefLists.append(corefs)

        #following loop is to resolve the ambiguity if a token is in more than one clusters, one token has to be in one cluster
        for cluster in clusters:
            for mention in cluster:
                token = doc[mention.start]
                tokClusters = token._.coref_clusters #All the clusters of corefering mentions that contain the token
                if len(tokClusters)>1:#if this token is in more than one clusters
                    print("WARNING!! This token is in more than one clusters: "+token.text)
                    s = mention.start
                    cIndex1 = -1
                    mIndex1 = -1
                    cIndex2 = -1
                    mIndex2 = -1
                    e = -1
                    e2 = -1
                    for i,tokCluster in enumerate(tokClusters):#resolve the ambiguity, one token has to be in one cluster
                        for j,tokMention in enumerate(tokCluster):
                            s2 = tokMention.start
                            if s in range(s2,tokMention.end) or s2 in range(s,mention.end): #if two mentions overlap
                                if e==-1: #token's mention in first cluster
                                    cIndex1 = tokCluster.i#index of this cluster in doc
                                    mIndex1 = j
                                    e = tokMention.end #end index of token's mention in first cluster
                                    main1 = tokCluster
                                else: #token's mention in sencond cluster
                                    cIndex2 = tokCluster.i
                                    mIndex2 = j
                                    e2 = tokMention.end #end index of token's mention in second cluster
                    if e<e2:
                        corefLists[cIndex2][mIndex2]['start'] = e
                    elif e2< e:
                        corefLists[cIndex1][mIndex1]['start'] = e2
                    elif corefLists[cIndex1][mIndex1]['start'] < corefLists[cIndex2][mIndex2]['start']:
                        corefLists[cIndex1][mIndex1]['end'] = corefLists[cIndex2][mIndex2]['start']
                    elif corefLists[cIndex2][mIndex2]['start'] < corefLists[cIndex1][mIndex1]['start']:
                        corefLists[cIndex2][mIndex2]['end'] = corefLists[cIndex1][mIndex1]['start']

        #this loop populates the corefIndicesAll list
        for corefs in corefLists:
            for coref in corefs:
                s = coref['start']
                # e = coref['end']
                for coref2 in corefs:
                    s2 = coref2['start']
                    # e2 = coref2['end']
                    if s!=s2:
                        corefIndicesAll[s].append(s2)

    return corefIndicesAll


# # this functions takes as input tokenIndex and returns a list of its coreferents' indices
# def getCorefs(tokenIndex,sentence):
#     '''this functions takes as input tokenIndex and returns a list of its coreferents' indices'''
#     # nlp = spacy.load('en')
#     nlp = spacy.load('en_core_web_lg')
#     neuralcoref.add_to_pipe(nlp)# Add neural coref to SpaCy's pipe
#
#     # You're done. You can now use NeuralCoref as you usually manipulate a SpaCy document annotations.
#     doc = nlp(sentence)
#
#     passedTokenCharIndex = getPassedTokenCharIndex(tokenIndex,sentence)
#
#     corefIndices = []
#     token = doc[tokenIndex]
#
#     if token.idx!=passedTokenCharIndex:
#         token = getTokenFromCharIndex(doc, passedTokenCharIndex)
#     if token.idx != passedTokenCharIndex:
#         print("Inside function getCorefs of CorefResolution class, the char offset of passed tokenIndex and token do not match.")
#         exit(1)
#     indicesMapping = getIndicesMapping(doc,sentence)
#     if token._.in_coref:
#         clusters = token._.coref_clusters
#         # if len(clusters)>1:
#
#         mentions = token._.coref_clusters[0].mentions  # mentions is a list of Spans
#
#         # Have to only add corefs' starting indices. ex 2 Britney Spears and her are corefs, have to add (1,6) and not (2,6)
#         isCorrect = False
#         for mention in mentions:  # each mention is a Span
#             s = indicesMapping[mention.start]
#             if s == token.i:
#                 isCorrect = True
#
#         if isCorrect:
#             for mention in mentions:  # each mention is a Span
#                 s = indicesMapping[mention.start]
#                 if s != token.i:
#                     corefIndices.append(s)
#
#     return corefIndices


##This func is only called from inside RuleEngineGender file, and checks if the personPronoun and Person are corefs
##Returns 1 if personPronoun and Person are coreferents, 0 otherwise
def areCorefs(perPronounTokStart,perPronounTokEnd,personTokStart, personTokEnd, sentenceTokens):
    ''' Returns 1 if subj and obj are coreferents, 0 otherwise '''
    # nlp = spacy.load('en')
    nlp = spacy.load('en_core_web_lg')
    neuralcoref.add_to_pipe(nlp)# Add neural coref to SpaCy's pipe

    # You're done. You can now use NeuralCoref as you usually manipulate a SpaCy document annotations.
    # doc = nlp(u'My sister has a dog. She loves him. She likes cake.')
    nlp.tokenizer = nlp.tokenizer.tokens_from_list  # because of this, nlp will now take as input tokens instead of text. this is to ensure that tokens indices by spacy and that sent in argument are same
    doc = nlp(sentenceTokens)
    # # print([token.text for token in doc])
    #
    span1 = doc[perPronounTokStart:perPronounTokEnd+1]#for NeuralCoref, add 1 in End indices
    span2 = doc[personTokStart:personTokEnd+1]
    # span2 = doc.char_span(personCharStart,personCharEnd)

    for token in span1:
        # token = doc[-1]
        if token._.in_coref:
            mentions = token._.coref_clusters[0].mentions #mentions is a list of Spans
            for mention in mentions: #each mention is a Span
                for token2 in span2:
                    if token2 in mention:
                        return 1

    return 0