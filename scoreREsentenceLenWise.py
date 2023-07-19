#!/usr/bin/env python3

"""
Score the predictions with gold labels, using precision, recall and F1 metrics.
"""

import sys
from collections import Counter
import os.path
import json

NO_RELATION = "no_relation"

def scoreREsenWise(dataset,corrected,predictionFileName="outputCustRE.txt", verbose=True): #dataset=cust, or (train or test)if TACTRED. 'ours' were the old versions of custFRE dataset
    # PRE REQUISITE - added by raabia
    # Before running check that file paths are correct
    # complete path = "C:\Users\Raabia Asif\PycharmProjects\HmaraNER v9 14May19 conllLists removed\outputTACRED\test\familyAll\outputCustRE.txt"
    args = {}
    if dataset=='ours':
        directoryOut = 'outputOurFREdataset/'
        directoryIn = 'inputOurFREdataset/'
        inFile = directoryIn + 'datasetAnnotated.json'
    elif dataset == 'ours1':
        directoryOut = 'outputOurFREdatasetNew1/'
        directoryIn = 'inputOurFREdatasetNew1/'
        inFile = directoryIn + 'datasetAnnotated.json'
    elif dataset == 'ours2':
        directoryOut = 'outputOurFREdatasetNew2/'
        directoryIn = 'inputOurFREdatasetNew2/'
        inFile = directoryIn + 'datasetAnnotated.json'
    elif dataset == 'cust': #for CustFRE dataset
        directoryIn = 'inputCustFREdataset\\'
        directoryOut = 'outputCustFREdataset\\'
        inFile = directoryIn + 'datasetAnnotated.json'
    else: #for TACRED-F dataset
        if corrected==True:
            folder = "familyAllCorrected"
        else: folder = "familyAll"
        directoryIn = 'inputTACRED/'+folder+'/'+dataset
        directoryOut = 'outputTACRED/'+dataset+'/'+folder+'/'
        inFile = directoryIn +'.json'
    args['gold_file'] = os.path.normpath(directoryOut+"relationsKey.txt")
    args['pred_file'] = os.path.normpath(directoryOut+predictionFileName)
    args['sentences_file'] = os.path.normpath(directoryIn+"InputTexts.txt")
    args['dataset_file'] = os.path.normpath(inFile)
    args['noOfPersons_file'] = os.path.normpath(directoryOut+"noOfPersons.txt")

    key = [str(line).rstrip('\n') for line in open(args['gold_file'])]
    prediction = [str(line).rstrip('\n') for line in open(str(args['pred_file']))]
    # sentences = [str(line).rstrip('\n') for line in open(str(args['sentences_file']))]
    with open(args['dataset_file'], 'r', encoding='utf-8', errors='ignore') as f:
        data = json.load(f)
        sentences = [example['token'] for example in data]
    noOfPersonsRels = [str(line).rstrip('\n') for line in open(str(args['noOfPersons_file']))]
    # sentencesLengths = [len(line.split(' ')) for line in sentences] # sentence length in terms of number of tokens in sentence.
    sentencesLengths = [len(sentence) for sentence in sentences] # sentence length in terms of number of tokens in sentence.
    noOfPersons = [int(line.split('\t')[0]) for line in noOfPersonsRels] # because file contained on each line 2 tab separated nos: noOfPersons and noOfRelWords
    noOfRelWords = [int(line.split('\t')[1]) for line in noOfPersonsRels] # because file contained on each line 2 tab separated nos: noOfPersons and noOfRelWords
    avgSenLen = sum(sentencesLengths)/len(sentencesLengths)
    print('Average sentence length of TACRED-F '+dataset+' dataset (Corrected = '+str(corrected)+'): '+str(avgSenLen))
    print('predictionFile: '+directoryOut+predictionFileName)

    # Check that the lengths match
    if len(prediction) != len(key):
        print("Gold and prediction file must have same number of elements: {} in gold vs {} in prediction".format(len(key), len(prediction)))
        exit(1)
    if len(prediction) != len(sentences):
        print("Sentences file must have same number of elements as prediction file: {} in gold vs {} in prediction".format(len(key), len(prediction)))
        exit(1)

    sentenceLengthSlabs = [20,25,30,35,40,45,5000,-1] #the last -1 is used as special number for calculating scores for all sentence lengths
    noOfPersonsSlabs = [0,1,2,3,4,5,5] # the last no is repeated for any no greater than the 2nd last index (11)
    noOfRelationsSlabs = [0,1,2,3,3] # the last no is repeated for any no greater than the 2nd last index (7)
    senLenRangeStart = 0
    for (i,noOfPer) in enumerate(noOfPersonsSlabs):
        correct_by_relation = Counter()
        guessed_by_relation = Counter()
        gold_by_relation = Counter()
        no_of_negative_examples = 0

        # Loop over the data to compute a score
        for row in range(len(key)):
            gold = key[row]
            guess = prediction[row]
            noOfPersonThisExample = noOfPersons[row]

            if (noOfPersonThisExample==noOfPer and len(noOfPersonsSlabs)-1 != i) or \
                    (noOfPersonThisExample>noOfPer and len(noOfPersonsSlabs)-1 == i):# if its the last no in list noOfPersonsSlabs, then check that its greater than
                if gold == NO_RELATION and guess == NO_RELATION: # True Negative
                    no_of_negative_examples += 1
                    pass
                elif gold == NO_RELATION and guess != NO_RELATION: # False Positive
                    no_of_negative_examples += 1
                    guessed_by_relation[guess] += 1
                elif gold != NO_RELATION and guess == NO_RELATION: # False Negative
                    gold_by_relation[gold] += 1
                    # print(sentences[row]+gold)
                elif gold != NO_RELATION and guess != NO_RELATION: # True Positive
                    guessed_by_relation[guess] += 1
                    gold_by_relation[gold] += 1
                    if gold == guess:
                        correct_by_relation[guess] += 1

        print('\n--------------------------------------------------------------------')
        if len(noOfPersonsSlabs)-1 != i:
            print('------------------- No of Persons =  '+str(noOfPer)+' -----------------------')
        else:
            print('------------------- No of Persons > '+str(noOfPer)+' -----------------------')
        print('--------------------------------------------------------------------')

        # Print verbose information
        if verbose:
            print("Per-relation statistics:")
            relations = gold_by_relation.keys()
            longest_relation = 0
            for relation in sorted(relations):
                longest_relation = max(len(relation), longest_relation)
            for relation in sorted(relations):
                # (compute the score)
                correct = correct_by_relation[relation]
                guessed = guessed_by_relation[relation]
                gold    = gold_by_relation[relation]
                prec = 1.0
                if guessed > 0:
                    prec = float(correct) / float(guessed)
                recall = 0.0
                if gold > 0:
                    recall = float(correct) / float(gold)
                f1 = 0.0
                if prec + recall > 0:
                    f1 = 2.0 * prec * recall / (prec + recall)
                # (print the score)
                sys.stdout.write(("{:<" + str(longest_relation) + "}").format(relation))
                sys.stdout.write("  P: ")
                if prec < 0.1: sys.stdout.write(' ')
                if prec < 1.0: sys.stdout.write(' ')
                sys.stdout.write("{:.2%}".format(prec))
                sys.stdout.write("  R: ")
                if recall < 0.1: sys.stdout.write(' ')
                if recall < 1.0: sys.stdout.write(' ')
                sys.stdout.write("{:.2%}".format(recall))
                sys.stdout.write("  F1: ")
                if f1 < 0.1: sys.stdout.write(' ')
                if f1 < 1.0: sys.stdout.write(' ')
                sys.stdout.write("{:.2%}".format(f1))
                sys.stdout.write("  #: %d" % gold) #no of examples
                sys.stdout.write("\n")
            sys.stdout.write("No. of no_relation examples is . . #: %d" % no_of_negative_examples)
            sys.stdout.write("\n")
            print("")

        # Print the aggregate score
        if verbose:
            print("Final Score:")
        prec_micro = 1.0
        if sum(guessed_by_relation.values()) > 0:
            prec_micro   = float(sum(correct_by_relation.values())) / float(sum(guessed_by_relation.values()))
        recall_micro = 0.0
        if sum(gold_by_relation.values()) > 0:
            recall_micro = float(sum(correct_by_relation.values())) / float(sum(gold_by_relation.values()))
        f1_micro = 0.0
        if prec_micro + recall_micro > 0.0:
            f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
        print("Precision (micro): {:.3%}".format(prec_micro))
        print("   Recall (micro): {:.3%}".format(recall_micro))
        print("       F1 (micro): {:.3%}".format(f1_micro))
    for (i, noOfRel) in enumerate(noOfRelationsSlabs):
        correct_by_relation = Counter()
        guessed_by_relation = Counter()
        gold_by_relation = Counter()
        no_of_negative_examples = 0

        # Loop over the data to compute a score
        for row in range(len(key)):
            gold = key[row]
            guess = prediction[row]
            noOfRelThisExample = noOfRelWords[row]

            if (noOfRelThisExample == noOfRel and len(noOfRelationsSlabs) - 1 != i) or \
                    (noOfRelThisExample > noOfRel and len(
                        noOfRelationsSlabs) - 1 == i):  # if its the last no in list noOfRelsonsSlabs, then check that its greater than
                if gold == NO_RELATION and guess == NO_RELATION:  # True Negative
                    no_of_negative_examples += 1
                    pass
                elif gold == NO_RELATION and guess != NO_RELATION:  # False Positive
                    no_of_negative_examples += 1
                    guessed_by_relation[guess] += 1
                elif gold != NO_RELATION and guess == NO_RELATION:  # False Negative
                    gold_by_relation[gold] += 1
                    # print(sentences[row]+gold)
                elif gold != NO_RELATION and guess != NO_RELATION:  # True Positive
                    guessed_by_relation[guess] += 1
                    gold_by_relation[gold] += 1
                    if gold == guess:
                        correct_by_relation[guess] += 1

        print('\n--------------------------------------------------------------------')
        if len(noOfRelationsSlabs) - 1 != i:
            print('------------------- No of Relation Words =  ' + str(noOfRel) + ' -----------------------')
        else:
            print('------------------- No of Relation Words > ' + str(noOfRel) + ' -----------------------')
        print('--------------------------------------------------------------------')

        # Print verbose information
        if verbose:
            print("Per-relation statistics:")
            relations = gold_by_relation.keys()
            longest_relation = 0
            for relation in sorted(relations):
                longest_relation = max(len(relation), longest_relation)
            for relation in sorted(relations):
                # (compute the score)
                correct = correct_by_relation[relation]
                guessed = guessed_by_relation[relation]
                gold = gold_by_relation[relation]
                prec = 1.0
                if guessed > 0:
                    prec = float(correct) / float(guessed)
                recall = 0.0
                if gold > 0:
                    recall = float(correct) / float(gold)
                f1 = 0.0
                if prec + recall > 0:
                    f1 = 2.0 * prec * recall / (prec + recall)
                # (print the score)
                sys.stdout.write(("{:<" + str(longest_relation) + "}").format(relation))
                sys.stdout.write("  P: ")
                if prec < 0.1: sys.stdout.write(' ')
                if prec < 1.0: sys.stdout.write(' ')
                sys.stdout.write("{:.2%}".format(prec))
                sys.stdout.write("  R: ")
                if recall < 0.1: sys.stdout.write(' ')
                if recall < 1.0: sys.stdout.write(' ')
                sys.stdout.write("{:.2%}".format(recall))
                sys.stdout.write("  F1: ")
                if f1 < 0.1: sys.stdout.write(' ')
                if f1 < 1.0: sys.stdout.write(' ')
                sys.stdout.write("{:.2%}".format(f1))
                sys.stdout.write("  #: %d" % gold)
                sys.stdout.write("\n")
            sys.stdout.write("No. of no_relation examples is . . #: %d" % no_of_negative_examples)
            sys.stdout.write("\n")
            print("")

        # Print the aggregate score
        if verbose:
            print("Final Score:")
        prec_micro = 1.0
        if sum(guessed_by_relation.values()) > 0:
            prec_micro = float(sum(correct_by_relation.values())) / float(sum(guessed_by_relation.values()))
        recall_micro = 0.0
        if sum(gold_by_relation.values()) > 0:
            recall_micro = float(sum(correct_by_relation.values())) / float(sum(gold_by_relation.values()))
        f1_micro = 0.0
        if prec_micro + recall_micro > 0.0:
            f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
        print("Precision (micro): {:.3%}".format(prec_micro))
        print("   Recall (micro): {:.3%}".format(recall_micro))
        print("       F1 (micro): {:.3%}".format(f1_micro))

    for senLenRangeEnd in sentenceLengthSlabs:
        correct_by_relation = Counter()
        guessed_by_relation = Counter()
        gold_by_relation = Counter()
        no_of_negative_examples = 0

        # Loop over the data to compute a score
        for row in range(len(key)):
            gold = key[row]
            guess = prediction[row]
            sentenceLength = sentencesLengths[row]

            if (senLenRangeStart<sentenceLength and sentenceLength<=senLenRangeEnd) or senLenRangeEnd==-1:
                if gold == NO_RELATION and guess == NO_RELATION: # True Negative
                    no_of_negative_examples += 1
                    pass
                elif gold == NO_RELATION and guess != NO_RELATION: # False Positive
                    no_of_negative_examples += 1
                    guessed_by_relation[guess] += 1
                elif gold != NO_RELATION and guess == NO_RELATION: # False Negative
                    gold_by_relation[gold] += 1
                    # print(sentences[row]+gold)
                elif gold != NO_RELATION and guess != NO_RELATION: # True Positive
                    guessed_by_relation[guess] += 1
                    gold_by_relation[gold] += 1
                    if gold == guess:
                        correct_by_relation[guess] += 1

        print('\n--------------------------------------------------------------------')
        if senLenRangeEnd!=-1:
            print('------------------- SENTENCE LENGTHS '+str(senLenRangeStart+1)+' to '+str(senLenRangeEnd)+' -----------------------')
        else:
            print('------------------- FOR ALL SENTENCE LENGTHS -----------------------')
        print('--------------------------------------------------------------------')

        # Print verbose information
        if verbose:
            print("Per-relation statistics:")
            relations = gold_by_relation.keys()
            longest_relation = 0
            for relation in sorted(relations):
                longest_relation = max(len(relation), longest_relation)
            for relation in sorted(relations):
                # (compute the score)
                correct = correct_by_relation[relation]
                guessed = guessed_by_relation[relation]
                gold    = gold_by_relation[relation]
                prec = 1.0
                if guessed > 0:
                    prec = float(correct) / float(guessed)
                recall = 0.0
                if gold > 0:
                    recall = float(correct) / float(gold)
                f1 = 0.0
                if prec + recall > 0:
                    f1 = 2.0 * prec * recall / (prec + recall)
                # (print the score)
                sys.stdout.write(("{:<" + str(longest_relation) + "}").format(relation))
                sys.stdout.write("  P: ")
                if prec < 0.1: sys.stdout.write(' ')
                if prec < 1.0: sys.stdout.write(' ')
                sys.stdout.write("{:.2%}".format(prec))
                sys.stdout.write("  R: ")
                if recall < 0.1: sys.stdout.write(' ')
                if recall < 1.0: sys.stdout.write(' ')
                sys.stdout.write("{:.2%}".format(recall))
                sys.stdout.write("  F1: ")
                if f1 < 0.1: sys.stdout.write(' ')
                if f1 < 1.0: sys.stdout.write(' ')
                sys.stdout.write("{:.2%}".format(f1))
                sys.stdout.write("  #: %d" % gold)
                sys.stdout.write("\n")
            sys.stdout.write("No. of no_relation examples is . . #: %d" % no_of_negative_examples)
            sys.stdout.write("\n")
            print("")

        # Print the aggregate score
        if verbose:
            print("Final Score:")
        prec_micro = 1.0
        if sum(guessed_by_relation.values()) > 0:
            prec_micro   = float(sum(correct_by_relation.values())) / float(sum(guessed_by_relation.values()))
        recall_micro = 0.0
        if sum(gold_by_relation.values()) > 0:
            recall_micro = float(sum(correct_by_relation.values())) / float(sum(gold_by_relation.values()))
        f1_micro = 0.0
        if prec_micro + recall_micro > 0.0:
            f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
        print("Precision (micro): {:.3%}".format(prec_micro))
        print("   Recall (micro): {:.3%}".format(recall_micro))
        print("       F1 (micro): {:.3%}".format(f1_micro))

        senLenRangeStart = senLenRangeEnd
    return
