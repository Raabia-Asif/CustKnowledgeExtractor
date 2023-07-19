#For the sake of measuring run time, i have commented all code of online linking to dbpedia, should uncomment them for normal run
#Have also commented file writing code towards end: json.dump(hmariEntities,outfile, sort_keys=True, indent=4, ensure_ascii=False)
from datetime import datetime
import nltk
from SPARQLWrapper import SPARQLWrapper,JSON
from urllib import error
import socket
from http import client
import time
import json
from nltk.tree import Tree

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time Before Running CustNER on test dataset =", current_time)

T1 = ['PER','LOC','ORG', 'PERSON', 'LOCATION', 'ORGANIZATION', 'CITY', 'STATE_OR_PROVINCE', 'COUNTRY']
T2 = ['MONEY', 'NUMBER', 'ORDINAL', 'PERCENT', 'DATE', 'TIME', 'DURATION', 'SET', 'EMAIL', 'URL', 'RELIGION', 'IDEOLOGY', 'CRIMINAL_CHARGE', 'CAUSE_OF_DEATH']
T3 = ['NATIONALITY', 'TITLE']
personTypes = ['DUL:NaturalPerson','http://www.ontologydesignpatterns.org/ont/dul/DUL.owl/NaturalPerson','Schema:Person','http://schema.org/Person', 'DBpedia:Person','dbo:Person','http://dbpedia.org/ontology/Person', 'http://xmlns.com/foaf/0.1/Person', 'yago:Person100007846','http://dbpedia.org/class/yago/Person100007846','yago:WikicatFictionalBritishPeople','http://dbpedia.org/class/yago/WikicatFictionalBritishPeople']
placeTypes = ['Schema:Place','http://schema.org/Place', 'Schema:City','http://schema.org/City', 'DBpedia:Settlement','dbo:Settlement','http://dbpedia.org/ontology/Settlement', 'DBpedia:PopulatedPlace','dbo:PopulatedPlace','http://dbpedia.org/ontology/PopulatedPlace', 'DBpedia:Place','dbo:Place','http://dbpedia.org/ontology/Place', 'DBpedia:Location','dbo:Location','http://dbpedia.org/ontology/Location', 'DBpedia:City','dbo:City','http://dbpedia.org/ontology/City', 'yago:Location100027167','http://dbpedia.org/class/yago/Location100027167', 'yago:Building102913152','http://dbpedia.org/class/yago/Building102913152','DBpedia:Building','dbo:Building','http://dbpedia.org/ontology/Building','dbo:Museum','http://dbpedia.org/ontology/Museum','dbo:ArchitecturalStructure','http://dbpedia.org/ontology/ArchitecturalStructure', 'umbel-rc:Place','http://umbel.org/umbel/rc/Place', 'yago:Area102735688', 'http://dbpedia.org/class/yago/Area102735688']
orgTypes = ['DBpedia:Company','dbo:Company','http://dbpedia.org/ontology/Company','Http://xmlns.com/foaf/0.1/Organization', 'yago:Group100031264','http://dbpedia.org/class/yago/Group100031264','yago:Organization108008335','http://dbpedia.org/class/yago/Organization108008335', 'yago:Magazine106595351', 'http://dbpedia.org/class/yago/Magazine106595351', 'DBpedia:Newspaper','dbo:Newspaper','http://dbpedia.org/ontology/Newspaper','Schema:Organization','http://schema.org/Organization', 'DBpedia:Organisation','dbo:Organisation','http://dbpedia.org/ontology/Organisation','yago:AdministrativeUnit108077292','http://dbpedia.org/class/yago/AdministrativeUnit108077292','dbr:Single_market','http://dbpedia.org/resource/Single_market','dbr:Bloc','http://dbpedia.org/resource/Bloc','yago:WikicatTradeBlocs','http://dbpedia.org/class/yago/WikicatTradeBlocs']#adding ,'dbr:Single_market','http://dbpedia.org/resource/Single_market','dbr:Bloc','http://dbpedia.org/resource/Bloc','yago:WikicatTradeBlocs','http://dbpedia.org/class/yago/WikicatTradeBlocs'
notPpoTypes = ['dbr:Unit_of_account','http://dbpedia.org/resource/Unit_of_account','dbr:Scheme','http://dbpedia.org/resource/Scheme','Dbpedia:Film','dbo:Musical','http://dbpedia.org/ontology/Musical','dbo:MusicalWork','http://dbpedia.org/ontology/MusicalWork','dbo:Film','http://dbpedia.org/ontology/Film','Schema:Movie','http://schema.org/Movie','DBpedia:TelevisionShow','dbo:TelevisionShow','http://dbpedia.org/ontology/TelevisionShow','Dbpedia:Award','dbo:Award','http://dbpedia.org/ontology/Award','Dbpedia:Event','dbo:Event','http://dbpedia.org/ontology/Event','yago:WikicatDrugRings','http://dbpedia.org/class/yago/WikicatDrugRings','yago:Test100791078','http://dbpedia.org/class/yago/Test100791078','yago:WikicatTrials','http://dbpedia.org/class/yago/WikicatTrials','yago:WikicatFictionalDragons','http://dbpedia.org/class/yago/WikicatFictionalDragons','yago:Dragon109494388','http://dbpedia.org/class/yago/Dragon109494388','yago:WikicatHonorifics','http://dbpedia.org/class/yago/WikicatHonorifics','yago:ExpressiveStyle107066659','http://dbpedia.org/class/yago/ExpressiveStyle107066659','dbo:Book','http://dbpedia.org/ontology/Book']#removed ,'yago:Treaty106773434','http://dbpedia.org/class/yago/Treaty106773434','dbr:Single_market','http://dbpedia.org/resource/Single_market','dbr:Bloc','http://dbpedia.org/resource/Bloc','yago:WikicatTradeBlocs','http://dbpedia.org/class/yago/WikicatTradeBlocs','yago:Agreement106770275','http://dbpedia.org/class/yago/Agreement106770275','yago:Bloc108171094','http://dbpedia.org/class/yago/Bloc108171094','yago:CommercialTreaty106773857','http://dbpedia.org/class/yago/CommercialTreaty106773857','yago:Document106470073','http://dbpedia.org/class/yago/Document106470073','yago:LegalDocument106479665','http://dbpedia.org/class/yago/LegalDocument106479665','yago:Writing106362953','http://dbpedia.org/class/yago/Writing106362953','yago:WrittenAgreement106771653','http://dbpedia.org/class/yago/WrittenAgreement106771653','yago:WikicatCommercialTreaties','http://dbpedia.org/class/yago/WikicatCommercialTreaties','yago:Communication100033020','http://dbpedia.org/class/yago/Communication100033020'
hypernyms = ['http://dbpedia.org/resource/Bloc','http://dbpedia.org/resource/Scheme']#,'http://dbpedia.org/resource/Unit']
#got the following list of titles from https://www.codeproject.com/Questions/262876/Titles-or-Salutation-list
Titles = ['the','Mr.','Mr','Mrs.','Mrs','Miss','Dr.','Dr','Ms.','Ms','Prof.','Prof','Rev.','Rev','Lady','Sir','Capt.','Capt','Major','Lt.-Col.','Lt-Col','Col.','Col','Lady','Lt.-Cmdr.','Lt-Cmdr','The Hon.','The Hon','Cmdr.','Cmdr','Flt. Lt.','Flt Lt','Brgdr.','Brgrd','Judge','Lord','The Hon. Mrs','The Hon Mrs','Wng. Cmdr.','Wng Cmdr','Group Capt.','Group Capt','Rt. Hon. Lord','Rt Hon Lord','Revd. Father','Revd Father','Revd Canon','Maj.-Gen.','Maj-Gen','Maj Gen','Air Cdre.','Air Cdre','Viscount','Dame','Rear Admrl.','Rear Admrl']


#this function is called from inside the link function
# def queryDBpedia(entity,z,flag):#z=4 if expanded was true.. z=1 if acronym, z=0 if not, z=2 if nationality/Title, z=3 if recognized by only spotlight
#     # flag is 1 normally, 0 for conll exp4 i.e. highest preference to illinois
#     personTypesStr = "{?s rdf:type dul:NaturalPerson}UNION{?s rdf:type schema:Person}UNION{?s rdf:type dbo:Person}UNION{?s rdf:type foaf:Person}UNION{?s rdf:type yago:Person100007846}UNION{?s rdf:type yago:WikicatFictionalBritishPeople}"
#     placeTypesStr = "{?s rdf:type schema:Place}UNION{?s rdf:type schema:City}UNION{?s rdf:type dbo:Settlement}UNION{?s rdf:type dbo:PopulatedPlace}UNION{?s rdf:type dbo:Place}UNION{?s rdf:type dbo:Location}UNION{?s rdf:type dbo:City}UNION{?s rdf:type yago:Location100027167}UNION{?s rdf:type yago:Building102913152}UNION{?s rdf:type dbo:Building}UNION{?s rdf:type dbo:Museum}UNION{?s rdf:type dbo:ArchitecturalStructure}UNION{?s rdf:type umbel-rc:Place}UNION{?s rdf:type yago:Area102735688}"
#     orgTypesStr = "{?s rdf:type dbo:Company} UNION {?s rdf:type foaf:Organization} UNION {?s rdf:type yago:Group100031264} UNION {?s rdf:type yago:Organization108008335} UNION {?s rdf:type yago:Magazine106595351} UNION {?s rdf:type dbo:Newspaper} UNION {?s rdf:type schema:Organization} UNION {?s rdf:type dbo:Organisation} UNION {?s rdf:type yago:AdministrativeUnit108077292} UNION {?s rdf:type dbo:Single_market} UNION {?s rdf:type dbo:Bloc} UNION {?s rdf:type yago:WikicatTradeBlocs}"
#     url = 'http://dbpedia.org/resource'
#     #qText1,label,4,label,5,2 = SELECT distinct ?s WHERE {{?s rdfs:label "G20"@en. } UNION { ?altName rdfs:label "G20"@en ;dbo:wikiPageRedirects ?s.}FILTER (regex(?s, "http://dbpedia.org/resource/")).FILTER (!regex(?s, "Category"))).} LIMIT 10
#     qText1 = "SELECT Distinct (?s as ?URI) WHERE {{ ?s ?p ?o. ?altName rdfs:label \""
#     qText4 = "\"@en;dbo:wikiPageRedirects ?s. } UNION { ?s rdfs:label \""
#     qText5 = "\"@en.}."
#     qText2 = "FILTER (regex(?s, \'"+url+"\')).FILTER (!regex(?s, \"Category\")).} LIMIT 10"
#     qText3 = "SELECT Distinct (?s as ?URI) WHERE { ?s ?p ?o. dbr:"
#
#     sparql = SPARQLWrapper("http://dbpedia.org/sparql/")
#     sparql.setReturnFormat(JSON)
#     sparql.setTimeout(60000)#60 sec
#
#     isNotPpo = True
#     #RULE: Consider the biggest annotation that is not classed notPpo
#     if flag==1: #flag is 1 normally, 0 for conll exp4 i.e. highest preference to illinois
#         for e in entity:
#             if isNotPpo:
#                 isNotPpo = False
#                 uri = ''
#                 text = e['text']
#                 try:
#                     if e['annotator']=='SPOTLIGHT':
#                         i = list.index(entity,e)
#                         if i == 2 or (i==1 and len(entity)==2):  # i.e. spotlight's annotation is the smallest in size
#                             break
#                         if len(entity)==1:#if its only annotated by spotlgiht
#                             queryUnTyped = 'SELECT distinct (?s as ?URI) WHERE {{?s rdfs:label \"'+text+'\"@en. } FILTER (regex(?s, \'"+url+"\')).FILTER (!(regex(?s, \"Category\"))).} LIMIT 10'
#                             sparql.setQuery(queryUnTyped)
#                             response = sparql.query().convert()
#                             break
#                     if str.isupper(e['text']) and len(str.split(e['text'],' '))==1: #if e has all cap letters and has no spaces
#                         z=1 #then it might be acronym
#                     entityType = [e['ner'] if 'ner' in e.keys() else -1]
#                     entityType = entityType[0]
#                     types=''
#                     queryTyped=''
#                     response=''
#
#                     if entityType=='PERSON' or entityType=='PER':
#                         types = personTypesStr
#                     elif entityType=='LOCATION'or entityType=='LOC':
#                         types = placeTypesStr
#                     elif entityType=='ORGANIZATION'or entityType=='ORG':
#                         types = orgTypesStr
#
#                     allTypes = personTypesStr+"UNION"+placeTypesStr+"UNION"+orgTypesStr
#                     if z==1:#if acronym
#                         text = str.replace(text,'.','')#if there are any .s in acronym, remove them
#                         queryTyped = qText3+text+" dbo:wikiPageRedirects ?s. "+allTypes+qText2
#                         if text == "N'DJAMENA":
#                             queryTyped = "SELECT distinct ?s as ?URI WHERE {{?s rdfs:label \"N'Djamena\"@en. }UNION{ ?altName rdfs:label \"N'Djamena\"@en ;dbo:wikiPageRedirects ?s. }FILTER (regex(?s, \"http://dbpedia.org/resource/\")).FILTER (!(regex(?s, \"Category\"))).} Limit 10"
#                         sparql.setQuery(queryTyped)  # the typed query as a literal string
#                         response = sparql.query().convert()
#                         if response != '' and len(response["results"]["bindings"]) != 0:  # if uri returned
#                             uri = response["results"]["bindings"][0]['URI']['value']
#                             t = queryDBpediaType(uri)
#                             if t == 'notPpo':
#                                 response = 'notPpo'
#                                 isNotPpo = True
#                                 continue
#                             if str.count(e['text'],'('):#if the longer annotation consists of ( for ex European Economic Area (EEA) contains accronym, then go for shorter annotation instead
#                                 isNotPpo = True
#                                 continue
#                             entity[0] = {}
#                             entity[0]['resource'] = uri
#                             entity[0]['text'] = e['text']
#                             entity[0]['annotator'] = e['annotator']
#                             entity[0]['characterOffsetBegin'] = e['characterOffsetBegin']
#                             entity[0]['characterOffsetEnd'] = e['characterOffsetEnd']
#                             if 'ner' in e.keys():
#                                 entity[0]['ner'] = e['ner']
#                             if 'Rule' in e.keys():
#                                 entity[0]['Rule'] = e['Rule']
#                             break
#
#                     if types!='':
#                         queryTyped = qText1+text+qText4+text+qText5+types+qText2
#                     queryAllTypes = qText1+text+qText4+text+qText5+allTypes+qText2
#                     # queryContainsUnTyped = "SELECT Distinct (?s as ?URI) WHERE { ?s ?p ?o. ?s rdfs:label ?label. FILTER (contains(?label, \"" + text + "\")).FILTER (regex(?s, \'" + url + "\')).} LIMIT 10"
#                     # queryRegexAllTypes = "SELECT Distinct (?s as ?URI) WHERE { ?s ?p ?o. ?s rdfs:label ?label. "+allTypes+" FILTER (regex(?label, \""+text+"\", \"i\")).FILTER (regex(?s, \'"+url+"\')).} LIMIT 10"
#                     # queryRegexUnTyped = "SELECT Distinct (?s as ?URI) WHERE { ?s ?p ?o. ?s rdfs:label ?label. FILTER (regex(?label, \""+text+"\", \"i\")).FILTER (regex(?s, \'"+url+"\')).} LIMIT 10"
#
#                     if queryTyped!='':
#                         sparql.setQuery(queryTyped)  # the typed query as a literal string
#                         response = sparql.query().convert()
#                         if len(response["results"]["bindings"])==0:#if no uri returned
#                             if text.find("and")!=-1:
#                                 temp = text
#                                 temp.replace("and","&")
#                                 queryTyped = qText1+temp+qText4+temp+qText5+types+qText2
#                                 sparql.setQuery(queryTyped)
#                     if response=='' or len(response["results"]["bindings"]) == 0:# if no uri returned
#                         sparql.setQuery(queryAllTypes)
#                         response = sparql.query().convert()
#
#                     if response == '' or len(response["results"]["bindings"]) == 0:  # if no uri returned
#                         queryUnTyped = qText1+text+qText4+text+qText5+qText2
#                         sparql.setQuery(queryUnTyped)
#                         response = sparql.query().convert()
#                     if z==2:
#                         break
#                     if 'ner' in e.keys():
#                         if e['ner']=='PERSON' or e['ner']=='PER':
#                             if response != '' and len(response["results"]["bindings"]) != 0:  # if uri returned
#                                 uri = response["results"]["bindings"][0]['URI']['value']
#                                 entity[0] = {}
#                                 entity[0]['resource'] = uri
#                                 entity[0]['text'] = e['text']
#                                 entity[0]['annotator'] = e['annotator']
#                                 entity[0]['characterOffsetBegin'] = e['characterOffsetBegin']
#                                 entity[0]['characterOffsetEnd'] = e['characterOffsetEnd']
#                                 if 'ner' in e.keys():
#                                     entity[0]['ner'] = e['ner']
#                                 if 'Rule' in e.keys():
#                                     entity[0]['Rule'] = e['Rule']
#                             break
#                     if response != '' and len(response["results"]["bindings"]) != 0:  # if uri returned
#                         uri = response["results"]["bindings"][0]['URI']['value']
#                         t = queryDBpediaType(uri)
#                         if t == 'notPpo':
#                             response = 'notPpo'
#                             isNotPpo = True
#                             continue
#                         if str.count(e['text'],'('):  # if the longer annotation consists of ( for ex European Economic Area (EEA) contains accronym, then go for shorter annotation instead
#                             isNotPpo = True
#                             continue
#
#                         #for rule 6: z=3 ie. identified by spotlight
#                         if z==3 and len(entity)==1:
#                             isUriCorrect = 0
#                             uriLabel = str.rsplit(uri, '/', 1)
#                             uriLabel = str.replace(uriLabel[1], '_', ' ')
#                             disambiguateQuery = 'SELECT ?d WHERE {<' + uri + '> dbo:wikiPageDisambiguates ?d} LIMIT 10'
#                             sparql.setQuery(disambiguateQuery)
#                             responseDisambiguate = sparql.query().convert()
#                             if responseDisambiguate != '' and len(responseDisambiguate["results"]["bindings"]) != 0:  # if its a disambiguation page
#                                 isNotPpo = True
#                                 continue
#                             if str.startswith(uriLabel.lower(), str.lower(e['text'])):
#                                 isUriCorrect = 1
#                             if str.isupper(e['text']) and len(str.split(e['text'], ' ')) == 1:  # if e has all cap letters and has no spaces
#                                 if checkAcronym(uriLabel, e['text']):
#                                     isUriCorrect = 1
#                             if isUriCorrect == 0:
#                                 isNotPpo = True
#                                 continue
#                             else:
#                                 t = queryDBpediaType(uri)
#                                 if t == 'notPpo':
#                                     response = 'notPpo'
#                                     isNotPpo = True
#                                     continue
#                         #if uri returned is correct
#                         entity[0] = {}
#                         entity[0]['resource'] = uri
#                         entity[0]['text'] = e['text']
#                         entity[0]['annotator'] = e['annotator']
#                         entity[0]['characterOffsetBegin'] = e['characterOffsetBegin']
#                         entity[0]['characterOffsetEnd'] = e['characterOffsetEnd']
#                         if 'ner' in e.keys():
#                             entity[0]['ner'] = e['ner']
#                         if 'Rule' in e.keys():
#                             entity[0]['Rule'] = e['Rule']
#                         break
#                 except error.URLError as err:
#                     print('\n\n@@@@@@@@@@@')
#                     print(err)
#                     print(".. Re trying after 15 seconds.. \n@@@@@@@@@@@@@@\n")
#                     time.sleep(15)
#                     queryDBpedia(entity, z,flag)
#                 except error.HTTPError as err:
#                     print('\n\n@@@@@@@@@@@')
#                     print(err)
#                     print(".. Re trying after 15 seconds.. \n@@@@@@@@@@@@@@\n")
#                     time.sleep(15)
#                     queryDBpedia(entity, z,flag)
#                 except socket.timeout as err:
#                     print('\n\n@@@@@@@@@@@')
#                     print(err)
#                     print(".. Re trying after 15 seconds.. \n@@@@@@@@@@@@@@\n")
#                     time.sleep(15)
#                     queryDBpedia(entity, z,flag)
#                 except  client.HTTPException as err:
#                     print('\n\n@@@@@@@@@@@')
#                     print(err)
#                     print(".. Re trying after 30 seconds.. \n@@@@@@@@@@@@@@\n")
#                     time.sleep(30)
#                     queryDBpedia(entity, z,flag)
#                 except  client.RemoteDisconnected as err:
#                     print('\n\n@@@@@@@@@@@')
#                     print(err)
#                     print(".. Re trying after 30 seconds.. \n@@@@@@@@@@@@@@\n")
#                     time.sleep(30)
#                     queryDBpedia(entity, z,flag)
#                 except  ConnectionResetError as err:
#                     print('\n\n@@@@@@@@@@@')
#                     print(err)
#                     print(".. Re trying after 30 seconds.. \n@@@@@@@@@@@@@@\n")
#                     time.sleep(30)
#                     queryDBpedia(entity, z,flag)
#             if response != '' and len(response["results"]["bindings"]) != 0:  # if uri returned
#                 uri = response["results"]["bindings"][0]['URI']['value']
#                 t = queryDBpediaType(uri)
#                 if t == 'notPpo':
#                     response = 'notPpo'
#                     isNotPpo = True
#                     continue
#                 entity[0] = {}
#                 entity[0]['resource'] = uri
#                 entity[0]['text'] = e['text']
#                 entity[0]['annotator'] = e['annotator']
#                 entity[0]['characterOffsetBegin'] = e['characterOffsetBegin']
#                 entity[0]['characterOffsetEnd'] = e['characterOffsetEnd']
#                 if 'ner' in e.keys():
#                     entity[0]['ner'] = e['ner']
#                 if 'Rule' in e.keys():
#                     entity[0]['Rule'] = e['Rule']
#             else:#if no uri returned
#                 if z==3:#if from rule 6
#                     # response = 'notPpo'
#                     isNotPpo = True
#                     continue
#     elif flag==0:
#         text = entity[0]['text']
#         queryUnTyped = 'SELECT distinct (?s as ?URI) WHERE {{?s rdfs:label \"' + text + '\"@en. } FILTER (regex(?s, \'"+url+"\')).FILTER (!(regex(?s, \"Category\"))).} LIMIT 10'
#         sparql.setQuery(queryUnTyped)
#         response = sparql.query().convert()
#         if response != '' and len(response["results"]["bindings"]) != 0:  # if uri returned
#             uri = response["results"]["bindings"][0]['URI']['value']
#             entity[0]['resource']=uri
#     return response

#this function is called from inside the link function
# def queryDBpediaType(uri):
#     # return ''#REMOVE THIS LINE, add this line for debugging purpose
#
#     #this function returns either '' or 'notPpo' or types
#     query = "SELECT Distinct ?type WHERE { <"+uri+"> rdf:type ?type}"
#     # If dbp:orgType is dbr:Single_market
#     # query1 = "ASK { <"+uri+"> <http://dbpedia.org/property/orgType> <http://dbpedia.org/resource/Single_market>. }" #returns true or false
#     # If http://purl.org/linguistics/gold/hypernym is dbr:Bloc or dbr:Scheme
#     # query2= "SELECT Distinct ?hypernym WHERE  { <"+uri+"> <http://purl.org/linguistics/gold/hypernym> ?hypernym. }"
#
#     types=[]
#     response=''
#     sparql = SPARQLWrapper("http://dbpedia.org/sparql/")
#     sparql.setReturnFormat(JSON)
#     sparql.setTimeout(6000)#60 sec
#     # sparql.setQuery(query1)  # if dbp:orgType is dbr:Single_market
#     try:
#         # response = sparql.query().convert()
#         # if response!='' and response['boolean']==True:#if result is true
#         #     return 'notPpo'
#         # sparql.setQuery(query2)  # If http://purl.org/linguistics/gold/hypernym is dbr:Bloc
#         # response = sparql.query().convert()
#         # if response!='' and len(response["results"]["bindings"])!=0:#if result returned
#         #     for result in response["results"]["bindings"]:
#         #         tmp = result["hypernym"]["value"]
#         #         if tmp in hypernyms:
#         #             return 'notPpo'
#         sparql.setQuery(query)  # uri rdf:type ?type. the query as a literal string
#         response = sparql.query().convert()
#         l = len(response["results"]["bindings"])
#         print('Printing types from dbpedia for: '+uri)
#         print('No of types returned is ' + str(l))
#         if response!='' and l!=0:#if result returned
#             for result in response["results"]["bindings"]:
#                 tmp = result["type"]["value"]
#                 types.append(tmp)
#                 # print(tmp)
#         # print('Printed types from dbpedia for: ' + uri)
#     except error.URLError as err:
#         print('\n\n@@@@@@@@@@@')
#         print(err)
#         print(".. Re trying after 15 seconds.. \n@@@@@@@@@@@@@@\n")
#         time.sleep(15)
#         queryDBpediaType(uri)
#     except error.HTTPError as err:
#         print('\n\n@@@@@@@@@@@')
#         print(err)
#         print(".. Re trying after 15 seconds.. \n@@@@@@@@@@@@@@\n")
#         time.sleep(15)
#         queryDBpediaType(uri)
#     except socket.timeout as err:
#         print('\n\n@@@@@@@@@@@')
#         print(err)
#         print(".. Re trying after 15 seconds.. \n@@@@@@@@@@@@@@\n")
#         time.sleep(15)
#         queryDBpediaType(uri)
#     except  client.HTTPException as err:
#         print('\n\n@@@@@@@@@@@')
#         print(err)
#         print(".. Re trying after 30 seconds.. \n@@@@@@@@@@@@@@\n")
#         time.sleep(30)
#         queryDBpediaType(uri)
#     except  client.RemoteDisconnected as err:
#         print('\n\n@@@@@@@@@@@')
#         print(err)
#         print(".. Re trying after 30 seconds.. \n@@@@@@@@@@@@@@\n")
#         time.sleep(30)
#         queryDBpediaType(uri)
#     except  ConnectionResetError as err:
#         print('\n\n@@@@@@@@@@@')
#         print(err)
#         print(".. Re trying after 30 seconds.. \n@@@@@@@@@@@@@@\n")
#         time.sleep(30)
#         queryDBpediaType(uri)
#     return types

# def disambiguate(e,uris):#yet to write this function
#     return


# def link(entity,z,flag):#z=1 if acronym, z=0 if not, z=2 is for nationality/title , z=3 if recognized by only spotlight
#     # this function returns a uri/link if found, returns "notPpo" if the found link is not of person/loc/org
#     # returns -1 if not found
#     # this also sets value of e['ner']
#     # return  -1# REMOVE THIS LINE,  add this line for debugging purpose
#     e = entity[0]
#     response = -1
#     if 'uri' in e.keys():
#         spotUri = e['uri']
#         spotUriLabel = str.rsplit(spotUri, '/', 1)
#         spotUriLabel = str.replace(spotUriLabel[1], '_', ' ')
#         entityLabel = e['text']
#         # first check if spotlight's uri is correct, not need to query dbpedia in that case
#         if str.startswith(spotUriLabel.lower(), str.lower(entityLabel)) or (str.endswith(spotUriLabel.lower(), str.lower(entityLabel)) and str.rstrip(spotUriLabel.lower(), str.lower(entityLabel)) in Titles):
#             ner = getTypeSpot(e)
#             if (ner=="notPpo"):
#                 return "notPpo"
#             elif (ner!='' and ner!="notPpo"):
#                 entity[0]['ner'] = ner
#             else:
#                 ner = getTypeDBpedia(spotUri)
#                 if (ner == "notPpo"):
#                     return "notPpo"
#                 elif (ner != '' and ner != "notPpo"):
#                     entity[0]['ner'] = ner
#                     entity[0]['resource'] = spotUri
#             return spotUri
#
#     # if e['text']!='Prime Minister':
#     #     return -1
#     response = queryDBpedia(entity,z,flag=flag)
#     uris = []
#     uri = -1
#
#     print("linking..............."+e['text'])
#     if response=='notPpo':
#         return response
#     if response=='' or response==-1:
#         return response
#     for result in response["results"]["bindings"]:
#         uri = result["URI"]["value"]
#         uris.append(uri)
#         print(uri)
#
#     # with open(dbpediaOutFile,'a', encoding='utf-8') as outfile:
#     #     json.dump(response, outfile, sort_keys=True, indent=4, ensure_ascii=False)
#     # uri = disambiguate(e,uris) #uncomment this when disambiguate func is written
#     # comment this when disambiguate func is written
#     if len(uris)>0:
#         for uri in uris:
#             # uri = uris[0]
#             ner = getTypeDBpedia(uri)
#             if ner == 'notPpo':
#                 uri = "notPpo"
#             else:
#                 if ner != '':
#                     #if not ('ner' in e.keys()) or e['ner'] == 'MISC' or not (e['ner'] in T1):
#                     #should check here what to do it both types are different
#                     if not('ner' in e.keys()):
#                         entity[0]['ner'] = ner
#                     break
#     return uri

def applyRules(inputText, spotOutFile,nerOutFile,illinoisOutFile,entitiesOutFile,flag=1):#flag is 1 normally, 0 for conll exp4 i.e. highest preference to illinois
    tokenizedText = nltk.word_tokenize(inputText)
    posTaggedText = nltk.pos_tag(tokenizedText)
    hmariEntities = []  # entities recognized by hmaraNER. its going to be a list of dictionaries where each dict is an entity
    nHmariEntities = []  # entities NOT recognized by hmaraNER. its going to be a list of dictionaries where each dict is an entity
    pE=[]#its going to be a list of dictionaries where each dict is a potential entity
    nextToken={}

    with open(spotOutFile, 'r', encoding='utf-8',errors='ignore') as outfile:
        spotDict = json.load(outfile) #spotlight output file loaded as json dicts, contains entities identified by Illinois NER
    with open(illinoisOutFile, 'r', encoding='utf-8',errors='ignore') as outfile3:
        illinoisEntities = json.load(outfile3) #illinois output file loaded as json array of dictionaries
    illinoisEntities = sorted(illinoisEntities, key=lambda k: k['characterOffsetBegin'])  # sorting illinoisEntities on characterOffsetBegin
    with open(nerOutFile, 'r', encoding='utf-8',errors='ignore') as outfile2:
        nerDict = json.load(outfile2) #ner output file loaded as json dictionary
    # inputText = spotDict['@text'] #the input text
    spotResources = []
    if 'Resources' in spotDict.keys():  spotResources = spotDict['Resources']#list of dicts, contains entities identified by Spotlight
    spotResources = sorted(spotResources,key=lambda k: int(k['@offset']))#sorting spotResources on offset

    nerSentences = nerDict['sentences'] #list of dicts
    # spotIndex=0
    sentenceIndex=0
    allNerEntities = []
    #this for loop populates the list pE
    for sentence in nerSentences:
        nerEntities = sentence['entitymentions']#list of dicts, contains entities identified by NER in one sentence
        nerEntities = sorted(nerEntities, key=lambda k: k['characterOffsetBegin'])  # sorting nerEntities on offset
        allNerEntities += nerEntities

    for temp in spotResources:
        temp['characterOffsetBegin']=int(temp['@offset'])
        temp['characterOffsetEnd']= temp['characterOffsetBegin'] + len(temp['@surfaceForm'])
        # temp['text'] = temp['@surfaceForm']
    allNerEntities = sorted(allNerEntities, key=lambda k: k['characterOffsetBegin'])
    pE = illinoisEntities + allNerEntities + spotResources #concatenating
    pE = sorted(pE, key=lambda k: (k['characterOffsetBegin'],k['characterOffsetEnd']))
    pE2 = pE.copy()
    offsetRange2 = offsetRange1 = []
    prevE = {}
    mergedE = {}
    for e in pE: #merging duplicates in pE
        offsetRange2 = offsetRange1 #offset range of previous e
        offsetRange1 = range(e['characterOffsetBegin'], e['characterOffsetEnd'])#offset range of current e
        if prevE=={}:
            if 'label' in e.keys():
                if not('i-label' in mergedE.keys()):
                    for key in e.keys():
                        mergedE['i-' + key] = e[key]  # i for illinois
                else:
                    eTemp = {}
                    for key in e.keys():
                        eTemp['i-' + key] = e[key]  # i for illinois
                    pE2.append(eTemp)
            elif 'ner' in e.keys():
                for key in e.keys():
                    mergedE['n-' + key] = e[key]  # n for stanford NER
            elif '@URI' in e.keys():
                for key in e.keys():
                    mergedE['s-' + key] = e[key]  # s for spotlight
            pE2.append(mergedE)
            prevE = mergedE
            pE2.remove(e)
            continue
        offsetRangesOverlap = [i for i in offsetRange1 if i in offsetRange2]
        if len(offsetRangesOverlap)>0:
            #merge
            if 'label' in e.keys():
                if not('i-label' in mergedE.keys()):
                    for key in e.keys():
                        mergedE['i-' + key] = e[key]  # i for illinois
                else:
                    eTemp = {}
                    for key in e.keys():
                        eTemp['i-' + key] = e[key]  # i for illinois
                    pE2.append(eTemp)
            elif 'ner' in e.keys():
                for key in e.keys():
                    mergedE['n-' + key] = e[key]  # n for stanford NER
            elif '@URI' in e.keys():
                for key in e.keys():
                    mergedE['s-' + key] = e[key]  # s for spotlight

            pE2.remove(e)
            pE2.remove(prevE)
            pE2.append(mergedE)
            prevE = mergedE
        else:
            mergedE = {}
            if 'label' in e.keys():
                for key in e.keys():
                    mergedE['i-' + key] = e[key]  # i for illinois
            elif 'ner' in e.keys():
                for key in e.keys():
                    mergedE['n-' + key] = e[key]  # n for stanford NER
            elif '@URI' in e.keys():
                for key in e.keys():
                    mergedE['s-' + key] = e[key]  # s for spotlight
            pE2.append(mergedE)
            prevE = mergedE
            pE2.remove(e)
    # pE2 = sorted(pE2, key=lambda k: k['characterOffsetBegin'])#s i or n before 'characterOffsetBegin'
    pE = pE2.copy()
    # pre-processing
    pE3 = [] #list to hold lists of entities
    presRange = prevRange = []
    for e in pE:
        pEtemp = [] #list to hold annotations of one entity
        if 'i-tokens' in e.keys():
            illinoisE = {}
            illinoisE['annotator'] = 'ILLINOIS NER'
            illinoisE['text'] = e['i-tokens']
            illinoisE['characterOffsetBegin'] = e['i-characterOffsetBegin']
            illinoisE['characterOffsetEnd'] = e['i-characterOffsetEnd']
            illinoisE['ner'] = e['i-label']
            pEtemp.append(illinoisE)
        if 'n-ner' in e.keys():
            nerE = {}
            nerE['annotator'] = 'STANFORD NER'
            nerE['text'] = e['n-text']
            nerE['characterOffsetBegin'] = e['n-characterOffsetBegin']
            nerE['characterOffsetEnd'] = e['n-characterOffsetEnd']
            nerE['ner'] = e['n-ner']
            pEtemp.append(nerE)
        if 's-@URI' in e.keys():
            spotE = {}
            spotE['annotator'] = 'SPOTLIGHT'
            spotE['text'] = e['s-@surfaceForm']
            spotE['characterOffsetBegin'] = e['s-characterOffsetBegin']
            spotE['characterOffsetEnd'] = e['s-characterOffsetEnd']
            spotE['types'] = e['s-@types']
            spotE['uri'] = e['s-@URI']
            pEtemp.append(spotE)

        if flag==1:#flag is 1 normally, 0 for conll exp4 i.e. highest preference to illinois
            pEtemp = sorted(pEtemp, key=lambda k: len(k['text']),reverse=True)
            pEtemp = sorted(pEtemp, key=lambda k: k['characterOffsetBegin'])
        presE = pEtemp
        presRange = range(pEtemp[0]['characterOffsetBegin'], pEtemp[0]['characterOffsetEnd'])
        offsetRangesOverlap = [i for i in presRange if i in prevRange]

        # if e has no capital letter, remove it from list
        if not(str.islower(pEtemp[0]['text'])):
            if len(offsetRangesOverlap) > 0:
                pE3.remove(preE)
                pEtemp = preE + presE
                pEtemp = sorted(pEtemp, key=lambda k: len(k['text']), reverse=True)
                pEtemp = sorted(pEtemp, key=lambda k: k['characterOffsetBegin'])
                pE3.append(pEtemp)
            else:pE3.append(pEtemp)
            preE = pEtemp
            prevRange = range(pEtemp[0]['characterOffsetBegin'],pEtemp[0]['characterOffsetEnd'])
    pE = pE3.copy()

    #Rules for refinement of A_S and A_N
    tokens = nerSentences[sentenceIndex]['tokens']  # list of dics
    parseTree = Tree.fromstring(nerSentences[sentenceIndex]['parse'])
    # parseTree.pretty_print()

    for entity in pE:#e is a dict of different sizes
        print('Potential Entity: '+entity[0]['text'])
        # if entity[0]['text']!='Shen':
        #     continue
        if str.count(entity[0]['text'],'$')>0 or str.count(entity[0]['text'],'&')>0 or str.count(entity[0]['text'],'-')>0:
            nHmariEntities.append(entity)
            continue
        #pre requisites


        #If boundaries of all 3 are same, then use type that is common in 2.
        sameTexts = 0
        if len(entity)==3:
            annotators = []
            texts = []
            types = []
            for ent in entity:
                # if ent['annotator']=='STANFORD NER' or ent['annotator']=='ILLINOIS NER':
                annotators.append(ent['annotator'])
                texts.append(ent['text'])
                if 'ner' in ent.keys():
                    if ent['ner'].startswith('LOC') or ent['ner'].startswith('CITY') or ent['ner'].startswith('STATE_OR_PROVINCE') or ent['ner'].startswith('COUNTRY'):
                        ent['ner'] = 'LOCATION'
                    types.append(ent['ner'])
                else:
                    tt = getTypeDBpedia(ent['uri'])
                    if tt != '' and tt != 'notPpo':
                        types.append(tt)

            if texts[0]==texts[1]==texts[2]:#if texts of all 3 are same
                if (str.startswith(types[0],types[1]) or str.startswith(types[1],types[0])):#if 2 types are same
                    if types[0] in T1 and types[1] in T1:#and both types are in T1
                        sameTexts = 1
                        tt = types[0]
                elif tt != '' and tt != 'notPpo' and (str.startswith(types[1],types[2]) or str.startswith(types[2],types[1])):#if 2 types are same
                    if types[1] in T1 and types[2] in T1:#and both types are in T1
                        sameTexts = 1
                        tt = types[1]
                elif tt != '' and tt != 'notPpo' and (str.startswith(types[2],types[0]) or str.startswith(types[0],types[2])):#if 2 types are same
                    if types[2] in T1 and types[0] in T1:#and both types are in T1
                        sameTexts = 1
                        tt = types[2]

        uri = ''
        # uri = link(entity,-1,flag=flag) #first get the biggest annotation that is not classed notPpo at first position of entity list
        if uri=='notPpo':
            nHmariEntities.append(entity)
            continue
        if uri!='' and uri!=-1 and uri!='notPpo':
            entity[0]['resource'] = uri
        e = entity[0]
        if sameTexts==1:
            entity[0]['ner'] = tt

        resource = -1
        wordCount = len(str(e['text']).split())#no of words in this entity e
        token=next((token for token in tokens if token['characterOffsetBegin'] == e['characterOffsetBegin']),False) #token = the token whose characterOffsetBegin matches with e in this sentence, false if not matched in this sentence
        if token==False: # if characterOffsetBegin of e not matched with any token in this sentence, going to check if its offset range overlaps with any token e.g. token=3.8-tonne and e=tonne
            offsetRangeE = range(int(e['characterOffsetBegin']), int(e['characterOffsetEnd']))
            token = next((token for token in tokens if len([i for i in range(token['characterOffsetBegin'],token['characterOffsetEnd']) if i in offsetRangeE])>0),
                         False) #token = the token whose offset range overlaps with e in this sentence, false if no overlap in this sentence
        while token==False:#if token is not in this sentence
            sentenceIndex+=1
            if sentenceIndex>=len(nerSentences):
                print("CHECK: NER and Spot indices probably DO NOT match!!!")
                break
            tokens = nerSentences[sentenceIndex]['tokens']  # list of dics
            token = next((token for token in tokens if token['characterOffsetBegin'] == e['characterOffsetBegin']),False)
            if token == False:  # if characterOffsetBegin of e not matched with any token in this sentence, going to check if its offset range overlaps with any token e.g. token=3.8-tonne and e=tonne
                offsetRangeE = range(e['characterOffsetBegin'], e['characterOffsetEnd'])
                token = next((token for token in tokens if len(
                    [i for i in range(token['characterOffsetBegin'], token['characterOffsetEnd']) if
                     i in offsetRangeE]) > 0),
                             False)  # token = the token whose offset range overlaps with e in this sentence, false if no overlap in this sentence
            parseTree = Tree.fromstring(nerSentences[sentenceIndex]['parse'])
        tokenIndex = token['index']-1
        pos1=token['pos']#pos of first word of e
        # lemma1=token['lemma']#lemma of first word of e
        # #part of pre processing
        # if (wordCount == 1 and str.islower(lemma1)): # if single word and lemma form has no capital letter
        #     nHmariEntities.append(entity)
        #     continue

        # Rule 1 - elimination of some false positives
        if ((wordCount==1 and (not(str.startswith(pos1,'NN'))or e['text']in Titles)) or (wordCount>1 and (pos1=='CC' or pos1=='VBZ'))):#might have to add more cases with CC
            if 'ner' in e.keys() and e['ner'] != 'NATIONALITY' and e['ner'] != 'TITLE' and e['ner'] != 'MISC':
                nHmariEntities.append(entity)
                continue
            if not('ner' in e.keys()):
                nHmariEntities.append(entity)
                continue

        #Rule 2 – elimination of type T_2 mentions
        if ('ner' in e.keys() and e['ner'] in T2) or (len(entity)>1 and 'ner' in entity[1].keys() and entity[1]['ner'] in T2) or (len(entity)>2 and 'ner' in entity[2].keys() and entity[2]['ner'] in T2):
            nHmariEntities.append(entity)
            continue

        # Rule 5 – addition of mentions recognized by NER or illinois
        if e['annotator'] == 'STANFORD NER' or e['annotator'] == 'ILLINOIS NER':
            if 'ner' in e.keys() and e['ner'] in T1:
                if e['ner']=='LOC' or e['ner'] =='CITY' or e['ner'] =='STATE_OR_PROVINCE' or e['ner'] =='COUNTRY':
                    e['ner'] = 'LOCATION'
                e['Rule'] = '5 – addition of mentions recognized by NER: stanford or illinois'
                hmariEntities.append(entity)
                continue

        # Rule 3 and 4, type T3 = NATIONALITY or TITLE
        if 'ner' in e.keys():  # if e was identified by Stanford NER
            if e['ner'] in T3:  # if e has type T3 i.e. Tilte or Nationality
                expanded = False
                # if 'uri' in e.keys() and 'characterOffsetEnd' in e.keys() and e['n-characterOffsetEnd']<e['s-characterOffsetEnd']: #if e was identified by Spotlight and it was already expanded by spotlight
                #     expanded = True
                nextTokIndex = tokenIndex + wordCount  # index of token next to e
                if nextTokIndex < len(tokens):
                    nextToken = tokens[nextTokIndex]

                # Rule 3 – for type Title
                entityIndex = pE.index(entity)
                previousE, nextE = None, None
                if entityIndex > 0:
                    previousE = pE[entityIndex - 1][0]
                if entityIndex < (len(pE) - 1):
                    nextE = pE[entityIndex + 1][0]
                if e['ner'] == 'TITLE':
                    isEntity = 0
                    if nextTokIndex < len(tokens):
                        if nextToken['pos'] == ',':  # CHECKING TOWARDS RIGHT SIDE
                            nextTokIndex += 1
                            nextToken = tokens[nextTokIndex]
                        for y in hmariEntities:
                            x = y[0]
                            offsetRangesOverlap = [i for i in range(nextToken['characterOffsetBegin'],
                                                                    nextToken['characterOffsetEnd']) if
                                                   i in range(x['characterOffsetBegin'], x['characterOffsetEnd'])]
                            if len(offsetRangesOverlap) > 1:
                                break
                        if nextTokIndex < len(tokens) and str.startswith(nextToken['pos'], 'NNP') and not (
                                nextToken['ner'] in T2 and not (
                            offsetRangesOverlap > 1)):  # if token is a proper noun and is not in type T2. and is not already in hmariEntities
                            # if 'n-ner'==nextE['annotator'] and nextE['n-ner'] in T1:  # 2nd priority: if recognized by stanford ner
                            #     nextE['characterOffsetBegin'] = nextE['n-characterOffsetBegin']
                            #     nextE['characterOffsetEnd'] = nextE['n-characterOffsetEnd']
                            # elif 'i-tokens' in nextE.keys() and nextE['i-label'] in T1:  # 1st priority: if recognized by illinois ner
                            #     nextE['characterOffsetBegin'] = nextE['i-characterOffsetBegin']
                            #     nextE['characterOffsetEnd'] = nextE['i-characterOffsetEnd']
                            # elif 's-@URI' in nextE.keys():  # 3rd priority: if recognized by spotlight
                            #     nextE['characterOffsetBegin'] = nextE['s-characterOffsetBegin']
                            #     nextE['characterOffsetEnd'] = nextE['s-characterOffsetEnd']
                            # elif 'n-ner' in nextE.keys():  # 4th priority: if recognized in T3 by stanford ner
                            #     nextE['characterOffsetBegin'] = nextE['n-characterOffsetBegin']
                            #     nextE['characterOffsetEnd'] = nextE['n-characterOffsetEnd']
                            # elif 'i-tokens' in nextE.keys():  # 5th priority: if recognized by illinois ner
                            #     nextE['characterOffsetBegin'] = nextE['i-characterOffsetBegin']
                            #     nextE['characterOffsetEnd'] = nextE['i-characterOffsetEnd']

                            offsetRangesOverlap = [i for i in range(nextToken['characterOffsetBegin'],
                                                                    nextToken['characterOffsetEnd']) if
                                                   i in range(nextE['characterOffsetBegin'],
                                                              nextE['characterOffsetEnd'])]
                            if nextE != None and len(offsetRangesOverlap) > 1:
                                nHmariEntities.append(entity)
                                continue  # if next token is already an entity, don't expand e
                            while nextTokIndex < len(tokens) and str.startswith(nextToken['pos'], 'NNP') and not (
                                nextToken['ner'] in T2):  # if token is a proper noun n is not in type T2.
                                expandE(e, nextToken, {},'r')  # temp = temp+tokens[i]
                                isEntity = 1
                                nextTokIndex += 1
                                if nextTokIndex < len(tokens):
                                    nextToken = tokens[nextTokIndex]
                                else:
                                    break
                        else:  # checking towards left side now
                            nextTokIndex = tokenIndex - 1  # recheck this
                            if nextTokIndex > 0:
                                nextToken = tokens[nextTokIndex]  # this is actually previous token
                                if nextToken['pos'] == ',':
                                    nextTokIndex -= 1
                                    if nextTokIndex > 0:
                                        nextToken = tokens[nextTokIndex]
                            for y in hmariEntities:
                                x = y[0]
                                offsetRangesOverlap = [i for i in range(nextToken['characterOffsetBegin'],
                                                                        nextToken['characterOffsetEnd']) if
                                                       i in range(x['characterOffsetBegin'],
                                                                  x['characterOffsetEnd'])]
                                if len(offsetRangesOverlap) > 1:
                                    break
                            while nextTokIndex > 0 and str.startswith(nextToken['pos'], 'NNP') and not (
                                    nextToken['ner'] in T2 and not (
                                len(offsetRangesOverlap) > 1)):  # if token is a proper noun n is not in type T2
                                offsetRangesOverlap = [i for i in range(nextToken['characterOffsetBegin'],
                                                                        nextToken['characterOffsetEnd']) if
                                                       i in range(previousE['characterOffsetBegin'],
                                                                  previousE['characterOffsetEnd'])]
                                if previousE != None and len(offsetRangesOverlap) > 1:
                                    break  # if previous token is already an entity, don't expand e
                                expandE(e, nextToken,{}, 'l')  # temp = tokens[i] + temp #have to write a function to append these, they are not strings, but dictionaries
                                isEntity = 1
                                nextTokIndex -= 1
                                if nextTokIndex > 0:
                                    nextToken = tokens[nextTokIndex]
                                else:
                                    break
                    if isEntity == 1:
                        e['Rule'] = '3 – for type Title'
                        # resource = link(entity,2,flag) ## 2 is for nationality/title .. resource has URI if found, else -1, "notPpo" if found entity is not per/loc/org
                        e = entity[0]
                        if resource != -1:
                            if resource == "notPpo":
                                nHmariEntities.append(entity)
                                continue
                            if not ('ner' in e.keys()):
                                e['ner'] = 'PERSON'  # randomly giving type person to entities who are still not given any type
                            e['resource'] = resource
                            hmariEntities.append(entity)
                    elif isEntity == 0:
                        nHmariEntities.append(entity)
                    continue

                # RULE 4: if e has type Nationality
                elif e['ner'] == 'NATIONALITY':
                    if expanded==False:
                        adjToken = {}
                        nextTokIndex = tokenIndex + wordCount  # index of token next to e
                        if nextTokIndex < len(tokens):
                            nextToken = tokens[nextTokIndex]
                        while nextTokIndex < len(tokens) and nextToken['pos'] == 'JJ':
                            if len(adjToken)==0:#if its first iteration of while loop ie its first adjective
                                adjToken = nextToken
                            else:#if its not first iteration
                                adjToken['characterOffsetEnd'] = nextToken['characterOffsetEnd']
                                adjToken['originalText'] += ' ' + nextToken['originalText']
                            nextTokIndex += 1
                            if nextTokIndex < len(tokens):
                                nextToken = tokens[nextTokIndex]
                        while nextTokIndex < len(tokens) and str.startswith(nextToken['pos'], 'NN') and not (nextToken['ner'] in T1) and inSameNP(parseTree, e, nextToken):  # and are in same NP:
                            expandE(e, nextToken, adjToken, 'r')  # e = e+tokens[i]
                            expanded = True
                            nextTokIndex += 1
                            if nextTokIndex < len(tokens):
                                nextToken = tokens[nextTokIndex]
                            else:
                                break
                    resource = -1
                    if expanded:
                        # resource = link(entity, 2,flag)  # 2 is for nationality/title ... #resource has URI if found, else -1, "notPpo" if found entity is not per/loc/org
                        e = entity[0] #need to check if e and entity[0] are both same and both changed
                        if resource != -1:
                            if resource=="notPpo":
                                nHmariEntities.append(entity)
                                continue
                            if not('ner' in e.keys()):
                                e['ner'] = 'ORGANIZATION' #randomly giving type org to entities who are still not given any type
                            e['Rule'] = '4 – for type Nationality'
                            e['resource'] = resource
                            hmariEntities.append(entity)
                        else:
                            nHmariEntities.append(entity)
                    continue

        # Rule 6 – addition of mentions recognized by spotlight
        if e['annotator']=='SPOTLIGHT' and len(e['text'])>3 and str.startswith(pos1,'NNP'):
            # resource = link(entity,3,flag) #3 means recognized by only spotlight ... #resource has URI if found,  -1 if not found, "notPpo" if found link is not of per/loc/org
            e = entity[0]
            if resource != -1:
                if resource=="notPpo" or not('ner' in e.keys()) or e['ner']in T3: #if linking could not type it as ppo then drop it
                    nHmariEntities.append(entity)
                else:
                    e['resource'] = resource
                    e['Rule'] = '6 – addition of mentions recognized by spotlight'
                    hmariEntities.append(entity)
            else:
                nHmariEntities.append(entity)
            continue


    hmariEntities = sorted(hmariEntities, key=lambda k: k[0]['characterOffsetBegin'])
    hmariEntities2 = hmariEntities.copy()
    print("LINKING all hamari entities")
    for entity in hmariEntities:
        e = entity[0]
        if 'resource' in e.keys():
            resource = e['resource']
        else:
            resource = ''# resource = link(entity, 0,flag)  # 0 means e is not an acronym
        if resource != -1:
            if resource!="notPpo":
                entity[0]['resource'] = resource
            else:
                hmariEntities2.remove(entity)
                nHmariEntities.append(entity)
            # if not(str.startswith(e['ner'],type)) and type!='':#if type from dbpedia is different from ner type
            #     e['ner']=type #update ner to this new type by dbpedia # need to recheck this
        else:
            if not('ner' in e.keys()):
                print("\n\nTHIS ENTITY DOES NOT HAVE NER KEY "+e['text']+"\n\n")

    hmariEntities = hmariEntities2.copy()

    # Rules for Refinement of E_H i.e. hmariEntities
    for i, entity in enumerate(hmariEntities):
        # pre requisites
        e = entity[0]
        print(i)
        print(e['text'])
        wordCount = len(str(e['text']).split())  # no of words in this entity e
        sentenceIndex = 0
        tokens = nerSentences[sentenceIndex]['tokens']  # list of dics
        token = next((token for token in tokens if token['characterOffsetBegin'] == e['characterOffsetBegin']), False)
        while token == False:  # if token is not in this sentence
            sentenceIndex += 1
            if sentenceIndex>=len(nerSentences):
                print("CHECK: NER and Spot indices probably DO NOT match!!!")
                break
            # print(e['text'])
            tokens = nerSentences[sentenceIndex]['tokens']  # list of dics
            token = next((token for token in tokens if token['characterOffsetBegin'] == int(e['characterOffsetBegin'])), False)
        tokenIndex = token['index']-1
        nextTokIndex = tokenIndex + wordCount
        if nextTokIndex < len(tokens):
            nextToken = tokens[nextTokIndex]

        # Rule 7 – for recognizing acronyms
        if (wordCount>1  and not(str.isupper(e['text'])) and (str.startswith(nextToken['originalText'],'(') or str.startswith(nextToken['originalText'],',') or str.isupper(nextToken['originalText']))):#if token next to e is ( or , or is all caps
            maybAcronym = ''
            found=-1
            s=0
            if nextToken['originalText']=='(' or nextToken['originalText']==',':
                if nextTokIndex+1<len(tokens):
                    nextToken = tokens[nextTokIndex+1]
            if str.isupper(nextToken['originalText']):
                maybAcronym = nextToken
                for y in hmariEntities: #checking if maybAcronym already exists in hmariEntities
                    x = y[0]
                    offsetRangesOverlap = [i for i in range(maybAcronym['characterOffsetBegin'], maybAcronym['characterOffsetEnd']) if i in range(x['characterOffsetBegin'], x['characterOffsetEnd'])]
                    if len(offsetRangesOverlap) > 1: # already exists
                        found=1
                        break
            if maybAcronym != '' and found==-1: #and maybAcronym does not already exist in hmariEntities
                s = checkAcronym(e['text'],maybAcronym['originalText'])
                pos = maybAcronym['pos']
            if s!=0 and str.startswith(pos,'NNP'):
                temp3 = {}
                temp3['text'] = maybAcronym['originalText']
                temp3['characterOffsetBegin'] = maybAcronym['characterOffsetBegin']
                temp3['characterOffsetEnd'] = temp3['characterOffsetBegin']+len(temp3['text'])
                temp3['ner'] = e['ner']
                temp3['Rule'] = '7 – addition of acronyms'
                if 'resource' in e.keys():
                    temp3['resource']=e['resource']
                hmariEntities2.append([temp3])
                continue
            else:continue

        # # Rule 8 – for merging consecutive places
        # if e['ner']=='LOCATION':
        #     while str.startswith(nextToken['originalText'],','):# while token next to e is a comma. checking if there are > 1 consecutive place entities
        #         if i+1 < len(hmariEntities):
        #             if hmariEntities[i+1]['ner']=='LOCATION':# if next e has type location
        #                 if hmariEntities[i+1]['tokenBegin']==e['tokenEnd']+1:
        #
        #         token = next((token for token in tokens if token['characterOffsetBegin'] == e['characterOffsetBegin']),False)
        #         tokenIndex = token['index']-1
        #         nextTokIndex = tokenIndex + e['tokenEnd'] - e['tokenBegin'] #update nextTokenIndex at the end
    hmariEntities2 = sorted(hmariEntities2, key=lambda k: k[0]['characterOffsetBegin'])
    hmariEntities = hmariEntities2.copy()

    # Rule 9 – for adding re-occurrences of added entities
    for entity in hmariEntities:
        e = entity[0]
        start = 0
        temp2=temp1=-1
        while True:
            offsetRangesOverlap = []
            entityFound = -1
            index=-1
            temp1 = str.find(inputText,e['text'],start)#temp1 contains the lowest index where e['text'] is found, -1 if not found
            temp2 = temp1+len(e['text'])
            if temp1 == -1:#if e is not found in text
                break
            nextChar = prevChar = ''
            if temp2 < len(inputText):
                nextChar = inputText[temp2]
            if temp1>0:
                prevChar = inputText[temp1-1]
            if str.isalnum(nextChar) or str.isalnum(prevChar):#if the instance of e found in text is not an exact match
                start = temp2  # update start
                continue
            for y in hmariEntities2: #checking if the entity occurence found already exist in hmariEntities
                x = y[0]
                offsetRangesOverlap = [i for i in range(temp1, temp2) if i in range(x['characterOffsetBegin'], x['characterOffsetEnd'])]
                if len(offsetRangesOverlap) > 1: # entity occurence found
                    entityFound = x
                    index = hmariEntities2.index(y)
                    break
            if entityFound==-1:#if temp entity does not already exist in hmariEntities
                # make entity and add to hmariEntities
                t = {}
                t['characterOffsetBegin'] = temp1
                t['text'] = e['text']
                t['characterOffsetEnd'] = temp2
                if 'ner' in e.keys():
                    t['ner'] = e['ner']
                if 'resource' in e.keys():
                    t['resource'] = e['resource']
                if 'types' in e.keys():
                    t['types']=e['types']
                t['Rule'] = '9 – addition of re-occurrences of added entities'
                hmariEntities2.append([t])
            else: #if the occurence found already exists in hmariEntities
                if entityFound['text']==e['text']: #if its an exact match
                    if 'ner' in e.keys() and e['ner'] in T1:
                        hmariEntities2[index][0]['ner'] = e['ner']
                    if 'resource' in e.keys():
                        hmariEntities2[index][0]['resource'] = e['resource']
                    if 'types' in e.keys():
                        hmariEntities2[index][0]['types'] = e['types']
                t = entityFound
            start = t['characterOffsetEnd']  # update start at the end
        #for type person
        if 'ner' in e.keys() and (e['ner']=='PERSON' or e['ner']=='PER') and str.count(e['text']," ")>0:#if type is person and has more than one words
            for word in str.split(e['text']):
                if word=='der':
                    print('**************************************************************************************************')
                start = 0
                temp2=temp1=-1
                pos = [i[1] for i in posTaggedText if i[0] == word]
                if pos==[]: break
                pos = pos[0]
                while True:
                    if word.lower() in (title.lower() for title in Titles):
                        break
                    if len(word)<3:
                        break
                    if word.islower():#if all letters all small
                        break
                    if not(str.startswith(pos,'NNP')):#if pos of word is not NNP=properNoun
                        break
                    offsetRangesOverlap = []
                    entityFound = -1
                    index=-1
                    temp1 = str.find(inputText, word,start)  # temp1 contains the lowest index where e['text'] is found, -1 if not found
                    if temp1 == -1:
                        break
                    temp2 = temp1 + len(word)
                    nextChar = prevChar = ''
                    if temp2 < len(inputText):
                        nextChar = inputText[temp2]
                    if temp1 > 0:
                        prevChar = inputText[temp1 - 1]
                    if str.isalnum(nextChar) or str.isalnum(prevChar):  # if the instance of e found in text is not an exact match
                        start = temp2  # update start
                        continue
                    for y in hmariEntities2: #searching if the entity occurence found is already in hmariEntities, do not have to add again if yes
                        x = y[0]
                        offsetRangesOverlap = [i for i in range(temp1, temp2) if i in range(x['characterOffsetBegin'], x['characterOffsetEnd'])]
                        if len(offsetRangesOverlap) > 1:  # entity occurence found
                            entityFound = x
                            index = hmariEntities2.index(y)
                            break
                    if entityFound == -1: # if temp entity does not already exist in hmariEntities
                        # make entity and add to hmariEntities
                        t = {}
                        t['characterOffsetBegin'] = temp1
                        t['text'] = word
                        t['characterOffsetEnd'] = temp2
                        if 'ner' in e.keys():
                            t['ner'] = e['ner']
                        if 'resource' in e.keys():
                            t['resource'] = e['resource']
                        if 'types' in e.keys():
                            t['types'] = e['types']
                        t['Rule'] = '9 – addition of re-occurrences of added entities: Person'
                        print(posTaggedText)
                        hmariEntities2.append([t])
                        break
                    else:
                        if entityFound['text'] == word:  # if its an exact match
                            if 'ner' in e.keys() and e['ner'] in T1:
                                hmariEntities2[index][0]['ner'] = e['ner']
                            if 'resource' in e.keys():
                                hmariEntities2[index][0]['resource'] = e['resource']
                            if 'types' in e.keys():
                                hmariEntities2[index][0]['types'] = e['types']
                        t = entityFound
                    start = t['characterOffsetEnd']  # update start at the end
    # END_OF Rule 9 – for adding re-occurrences of added entities

    hmariEntities2 = sorted(hmariEntities2, key=lambda k: k[0]['characterOffsetBegin'])
    hmariEntities = hmariEntities2.copy()
    hmariEntities = list(filter(lambda k: (k[0]['ner'].startswith('PER') or k[0]['ner'].startswith('ORG') or k[0]['ner'].startswith('LOC')),hmariEntities)) #this line filters out any non-per/loc/org entities from hmariEntities

    # with open(entitiesOutFile, 'w', encoding='utf-8') as outfile:
    #     # outfile.write("No of entities: "+str(len(hmariEntities))+"\n")
    #     json.dump(hmariEntities,outfile, sort_keys=True, indent=4, ensure_ascii=False)

#END of function applyRules()


# this function checks if the passed tokenText is an acronym of the passed entity eText
def checkAcronym(eText,maybAcronymText):
    if maybAcronymText=='':
        return -1
    j = 0
    tokText = maybAcronymText
    for word in eText.split():
        if str.isupper(word[0]):#is a Capital Letter
            if j<len(tokText) and (tokText[j]=='.' or tokText[j]=='-'):
                j+=1
            if j<len(tokText) and word[0]==tokText[j]:
                j+=1
    if j==len(tokText):
        isAcronym = 1
    else:
        isAcronym = 0
    return isAcronym

# this function checks if the passed entity and token are in same noun phrase
# returns true if yes, false otherwise
# this function is intended for NATIONALITY entity
def inSameNP(parseTree,e,tok):
    sameNP = False
    wordsInE = str.split(e['text'],' ')
    for i in parseTree.subtrees(filter=lambda x: x.label() == 'NP'):
        if (x in i.leaves() for x in wordsInE) and tok['originalText'] in i.leaves():
            sameNP = True
            return sameNP
    return sameNP

# this function is only for entities that are recognized by NER, NOT FOR THOSE RECOGNIZED ONLY BY SPOTLIGHT
# this function is for expanding TITLE and NATIONALITY entities
def expandE (e,tok,adjTok,dir):#entity,token,direction='r'or'l'
    if len(adjTok) > 0:
        noOfToks = len(str.split(adjTok['originalText'],' '))+len(str.split(tok['originalText'],' '))
    else:
        noOfToks = len(str.split(tok['originalText'], ' '))
    if dir=='r':
        if len(adjTok)>0:
            e['text'] = e['text'] + ' ' +adjTok['originalText']+' '+ tok['originalText']
        else:
            e['text'] = e['text'] + ' ' +tok['originalText']
        e['characterOffsetEnd'] = tok['characterOffsetEnd']
    elif dir=='l':
        e['text'] = tok['originalText'] + ' ' + e['text']
        e['characterOffsetBegin'] = tok['characterOffsetBegin']


#this function returns per/loc/org/notPpo type for the passed entity based on its Spotlight types
def getTypeSpot(e):
    ner = ''
    if 'types' in e.keys():
        count = 0
        types = str.split(e['types'],',')
        commonTypesPerson = [x for x in types if x in personTypes]
        commonTypesPlace = [x for x in types if x in placeTypes]
        commonTypesOrg = [x for x in types if x in orgTypes]
        commonTypesNotPpo = [x for x in types if x in notPpoTypes]

        if len(commonTypesNotPpo) > 0:
            return 'notPpo'

        if len(commonTypesPerson) > 0:
            ner = 'PER'
            count += 1
        if len(commonTypesPlace) > 0:
            ner = 'LOC'
            count += 1
        if len(commonTypesOrg) > 0:
            ner = 'ORG'
            count += 1

        if count > 1:
            Max = len(commonTypesPerson)
            ner = 'PER'
            if len(commonTypesPlace) > Max:
                Max = len(commonTypesPlace)
                ner = 'LOC'
            if len(commonTypesOrg) > Max:
                Max = len(commonTypesOrg)
                ner = 'ORG'
    x = getTypeDBpedia(e['uri'])
    if x == 'notPpo':
        return "notPpo"
    else:
        if x!='':
            ner = x
    return ner

#this function returns per/loc/org/notPpo type or '' for the passed URI (of entity) based on its DBpedia types
def getTypeDBpedia(uri):
    count = 0
    type=''
    # label = str.rsplit(uri,'/',1)
    # label = str.replace(label[1],'_',' ')
    types = ''# types = queryDBpediaType(uri)
    if types=='notPpo':
        print('Type is: ' + type)
        return types
    commonTypesPerson = [x for x in types if x in personTypes]
    commonTypesPlace = [x for x in types if x in placeTypes]
    commonTypesOrg = [x for x in types if x in orgTypes]
    commonTypesNotPpo = [x for x in types if x in notPpoTypes]

    if len(commonTypesNotPpo) > 0:
        type = 'notPpo'
        print('Type is: '+type)
        return type

    if len(commonTypesPerson) > 0:
        type = 'PER'
        count+=1
    if len(commonTypesPlace) > 0:
        type = 'LOC'
        count += 1
    if len(commonTypesOrg) > 0:
        type = 'ORG'
        count += 1

    if count>1:
        Max = len(commonTypesPerson)
        type = 'PER'
        if len(commonTypesPlace) > Max:
            Max = len(commonTypesPlace)
            type = 'LOC'
        if len(commonTypesOrg) > Max:
            Max = len(commonTypesOrg)
            type = 'ORG'

    print('Type is: ' + type)
    return type

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time After Running CustNER on test dataset =", current_time)