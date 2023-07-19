# Can easily annotate spotlight by using spacy library  https://pypi.org/project/spacy-dbpedia-spotlight/

#!/usr/bin/env python
# coding=utf-8

#Import the modules
import requests
import json
import spotlight

def annotateSpotlight( doc,spotOutFile ):
    url = "http://model.dbpedia-spotlight.org/en/annotate"

    # --------For UNTYPED---------------
    params = {'text': doc, 'confidence': '0.0', 'spotter': 'Default'}  # for untyped

    # #--------For TYPED---------------
    # personTypes = 'DUL:NaturalPerson,Schema:Person,DBpedia:Person,Http://xmlns.com/foaf/0.1/Person,foaf:Person,dbo:Person,schema:Person,yago:Person100007846'
    # placeTypes = 'Schema:Place,Schema:City,Schema:Country,DBpedia:Settlement,DBpedia:PopulatedPlace,DBpedia:Place,DBpedia:Location,DBpedia:City,dbo:Place,yago:Location100027167,yago:Building102913152,umbel-rc:Place,yago:Area102735688'
    # orgTypes = 'Http://xmlns.com/foaf/0.1/Organization,foaf:Organization,dbo:Organisation,yago:Group100031264,yago:Magazine106595351,DBpedia:Newspaper,Schema:Organization,DBpedia:Organisation,DBpedia:Company,DBpedia:Website'
    # types = personTypes+placeTypes+orgTypes
    # params = {'text': doc, 'confidence': '0.3', 'types':types, 'spotter': 'Default'}#for typed

    headers = {'Accept': 'application/json'}
    try:

        r = requests.get(url, params=params,headers=headers)  # ,proxies=proxies)
        spotResponseText = r.text

        if r.status_code!=200:#==414: #414= Request-URI Too Long, 200=got Response, 400=Bad request
            l = len(doc)
            doc1 = doc[:int(l/2)]
            doc2 = doc[int(l/2):]
            params1 = {'text': doc1, 'confidence': '0.0', 'spotter': 'Default'}
            params2 = {'text': doc2, 'confidence': '0.0', 'spotter': 'Default'}
            r1 = requests.get(url, params=params1, headers=headers)
            r2 = requests.get(url, params=params2, headers=headers)
            data1 = json.loads(r1.text, encoding='utf-8', strict=True)
            data2 = json.loads(r2.text, encoding='utf-8', strict=True)
            for e in data2["Resources"]:
                x = int(e["@offset"])
                x += len(doc1)
                e["@offset"] = str(x)
            list.extend(data1["Resources"], data2["Resources"])
            data1['@text'] += data2['@text']
            with open(spotOutFile, 'w', encoding='utf-8') as outfile:
                json.dump(data1, outfile, sort_keys=True, indent=4, ensure_ascii=False)
            print(data1)
            return
            # t1 = r1.text[:len(r1.text)-2]
            # t2 = r2.text[str.find(r2.text,"Resources") + len("Resources") + 3:]
            # spotResponseText = t1+','+t2


        # Convert it to a Python dictionary
        data = json.loads(spotResponseText,encoding='utf-8',strict=True)
        with open(spotOutFile, 'w',encoding='utf-8') as outfile:
            json.dump(data, outfile, sort_keys = True, indent = 4, ensure_ascii = False)
        print(spotResponseText)
    except json.JSONDecodeError:
        print("Decoding JSON has failed")
        f = open(spotOutFile, "w+")
        f.write("Decoding JSON has failed")
    return

