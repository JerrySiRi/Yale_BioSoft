# -*- encoding: utf-8 -*-

# ------ Lann转换成可处理的格式！------ #

"""
@File    : generator.py
@Time    : 2020/10/19 21:48
@Author  : renyk
@Software: PyCharm
"""
import codecs
import json
import os

class Entity():
    def __init__(self,index,begin,end,sem,text):
        self.index=index
        self.begin=begin
        self.end=end
        self.sem=sem
        self.text=text

    def __str__(self):
        # return "T%d\t%s\t%d %d\t%s\n"%(self.index,self.sem,self.begin,self.end,self.text)
        return "T%d\t%s %d %d\t%s\n" % (self.index, self.sem, self.begin, self.end, self.text)

class Relation():
    #R1    att_NI Arg1:T6 Arg2:T5
    def __init__(self,index,sem,fromindex,toindex):
        self.index=index
        self.sem=sem
        self.fromindex=fromindex
        self.toindex=toindex

    def __str__(self):
        return "R%d\t%s\tArg1:T%d\tArg2:T%d\n"%(self.index,
                                               self.sem,self.fromindex,
                                               self.toindex)



def parseJson(path):

    for file in os.listdir(path):
        entityIdx={}
        entitylst={}
        relationlst={}
        content=''
        index=1
        reindex=1
        if file.endswith(".json"):
            print("Filename",file)
            f_json=os.path.join(path,file)
            f_text=os.path.join(path,file.replace(".json",".txt"))
            f_ann=os.path.join(path,file.replace(".json",".ann"))
            with codecs.open(f_json,'r','utf-8') as f \
            ,codecs.open(f_text,'w','utf-8') as w1 \
            ,codecs.open(f_ann,'w','utf-8') as w2 :
                jsonObj=json.load(f)
                for k,v in jsonObj.items():
                    # print(k)
                    if k=='content':
                        content=v
                        w1.write(v)
                    elif k=='indexes':
                        for i,j in v.items():
                            # print(j)
                            if "Entity" in j :
                                # for entitydict in j['Entity'].values():
                                for entitydict in j['Entity']:
                                    # print(entitydict)
                                    begin = int(entitydict['begin'])
                                    end = int(entitydict['end'])
                                    sem = entitydict['semantic']
                                    text = content[begin:end]
                                    if "%d_%d" % (begin, end) not in entitylst:
                                        entity = Entity(index, begin, end, sem, text)
                                        entitylst["%d_%d" % (begin, end)] = entity
                                        entityIdx[index] = "%d_%d" % (begin, end)
                                        index += 1

                for k,v in jsonObj.items():
                    # print(k)
                    if k=='indexes':
                        for i,j in v.items():
                            # print(j)
                            if "Relation" in j:
                                fromindex=-1
                                toindex=-1
                                # for relationitem in j['Relation'].values():
                                for relationitem in j['Relation']:
                                    fromEntdict = relationitem['fromEnt']
                                    toEntdict = relationitem['toEnt']
                                    frombegin = int(fromEntdict['begin'])
                                    fromend = int(fromEntdict['end'])
                                    fromsem = fromEntdict['semantic']
                                    fromtext = content[frombegin:fromend]
                                    if "%d_%d" % (frombegin, fromend) not in entitylst:
                                        continue
                                    else:
                                        fromindex = entitylst.get("%d_%d" % (frombegin, fromend)).index

                                    tobegin = int(toEntdict['begin'])
                                    toend = int(toEntdict['end'])
                                    tosem = toEntdict['semantic']
                                    totext = content[tobegin:toend]
                                    if "%d_%d" % (tobegin, toend) not in entitylst:
                                        continue
                                    else:
                                        toindex = entitylst.get("%d_%d" % (tobegin, toend)).index

                                    relation = Relation(reindex, relationitem['semantic'], fromindex, toindex)
                                    relationlst[reindex] = relation
                                    reindex += 1

                for i in entitylst.values():
                    w2.write(str(i))
                for i in relationlst.values():
                    w2.write(str(i))



if __name__=="__main__":
    path=r"/home/rs2926/workspace/Softname_Extraction/datasets/unconverted_files/gold_standard" ### YOUR JSON DIR
    parseJson(path)
