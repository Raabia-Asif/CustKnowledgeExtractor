#!/usr/bin/env python
# coding=utf-8

#Import the modules
from ccg_nlpy import local_pipeline
# from ccg_nlpy import remote_pipeline
# import json
pipeline = local_pipeline.LocalPipeline()

def annotateIllinois(line,illinoisOutFile):

    # pipeline = remote_pipeline.RemotePipeline()
    doc = pipeline.doc(text=line)
    if doc is not None:
        ner_view = doc.get_ner_conll
        offsets = doc.get_token_char_offsets

        x = doc.as_json.keys()
        y = doc.get_sentence_end_token_indices
        z = doc.get_sentence_boundaries
        tokens = doc.get_tokens
        # print(ner_view)
        # print(doc.get_ner_ontonotes)
        # print(x)
        # print(y)
        # print(z)
        # print(tokens)
        # print(offsets)
        # print()

        illinoisEntities = []  # find NER constituents that are labeled PER (label from NER view)
        if ner_view.cons_list is not None:
            for ner in ner_view:
                temp = ner.copy()
                temp['characterOffsetBegin'] = offsets[ner['start']][0]
                temp['characterOffsetEnd'] = offsets[ner['end'] - 1][1]
                illinoisEntities.append(temp)

        # try:
        #     with open(illinoisOutFile, 'w', encoding='utf-8') as outfile:
        #         json.dump(illinoisEntities, outfile, sort_keys=True, indent=4, ensure_ascii=False)
        # except json.JSONDecodeError:
        #     print("Decoding JSON has failed")
        #     f = open(illinoisOutFile, "w+")
        #     f.write("Decoding JSON has failed")
    else:
        print("None returned in output")


    return