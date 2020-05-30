# @article{leskovec2016snap,
#   title={SNAP: A General-Purpose Network Analysis and Graph-Mining Library},
#   author={Leskovec, Jure and Sosi{\v{c}}, Rok},
#   journal={ACM Transactions on Intelligent Systems and Technology (TIST)},
#   volume={8},
#   number={1},
#   pages={1},
#   year={2016},
#   publisher={ACM}
# }
# @misc{snapnets,
#   author       = {Jure Leskovec and Andrej Krevl},
#   title        = {{SNAP Datasets}: {Stanford} Large Network Dataset Collection},
#   howpublished = {\url{http://snap.stanford.edu/data}},
#   month        = jun,
#   year         = 2014
# }

############################## START ##############################
#remark for TA:
#   running time is quite long (may need a whole day)
#   this snap coding is the sanme as the snap.py with addition on the plotting of graph using numpy and matplotlib library
#   need matplotlib and numpy
import snap
import random
import matplotlib.pyplot as plt
import numpy as np
LG = snap.LoadEdgeList(snap.PUNGraph , "soc-Slashdot0902.txt", 0, 1, '\t')
snap.DelSelfEdges(LG)

#initialize
nodeL, edgeL = [],[]
for i in range(LG.GetNodes()):
    nodeL.append({'id':i, 'outdeg':None, 'indeg':None, 'class':0})
    edgeL.append([])

#fill nodeL
for node in LG.Nodes():
    if(node.GetId()>nodeL[-1]['id']):
        print(node.GetId())
        break
    nodeL[node.GetId()]['outdeg']:node.GetOutDeg()
    nodeL[node.GetId()]['outdeg']:node.GetOutDeg()

#storing the connected nodeid w.r.t every node in edgeL, can access the node by node(edge[i][j])
for E in LG.Edges():
    edgeL[E.GetSrcNId()].append(E.GetDstNId())
    edgeL[E.GetDstNId()].append(E.GetSrcNId())

#get dataset info
snap.PrintInfo(LG)
snap.PlotInDegDistr(LG, "example", "Undirected graph - in-degree Distribution")
snap.PlotOutDegDistr(LG, "example", "Undirected graph - out-degree Distribution")
print('max deg:',len(edgeL[snap.GetMxDegNId(LG)]))

min = len(edgeL[0])
for i,val in enumerate(edgeL):
    if(len(edgeL[i])<min):
        min = len(edgeL[i])
print('min deg:', min)

avg = 0
for i,val in enumerate(edgeL):
    avg+=len(edgeL[i])
avg = avg/len(edgeL)
print('avg deg:',avg)

DegToCntV = snap.TIntPrV()
snap.GetOutDegCnt(LG, DegToCntV)
for item in DegToCntV:
    print("%d nodes with out-degree %d" % (item.GetVal2(), item.GetVal1()))
################################################################
################################################################
############################ CASE 1 ############################
################################################################
################################################################

#case1: simpliest case with 
#     A       B
# A   a,a     c,c
# B   c,c     b,b

#random initialize
def rand_adopter(nodeList,x):
    list = []
    for i in range(x):
        node = random.randint(0,nodeList[-1]['id'])
        while(nodeList[node]['class']!=0):
             node = random.randint(0,nodeList[-1]['id'])
        list.append(node)
        print(node)
        nodeList[node]['class']=1
    return list

#check correct
def calc_class(nodeList):
    c0, c1, cn = 0,0,0
    for n in nodeList:
        if (n['class']==0): c0+=1
        elif (n['class']==1): c1+=1
        else: cn+=1
    print('0:',c0)
    print('1:',c1)
    return [c0,c1]
    #print('number of class NULL instances:',cn)

#reset class default
def reset(nodeList):
    for n in nodeList:
        n['class'] = 0

#change adopt new product or not
def change(nodeL,edgeL,nid,q):
    if(nodeL[nid]['class']!=1):
        c0, c1 = 0,0
        for i in edgeL[nid]:
            if(nodeL[i]['class']==0): c0+=1
            elif(nodeL[i]['class']==1): c1+=1
        if (c1/(c0+c1-0.0001) >= q): #prevent / by 0
            nodeL[nid]['class']=1
            return True
    return False

#find neigbor
def find_neigbor(edgeList, alt):
    listset = set()
    for nid in alt:
        for n in edgeList[nid]:
            listset.add(n)
    return listset

def update_neigbor(listset,edgeList, nid):
    for n in edgeList[nid]:
        listset.add(n)
    return listset

#check from neighbour to neigbour every itr
def cascading(nodeL, edgeL, checklist, q, itr):
    for t in range(itr):
        update_ls = []
        for n in checklist:
            if(change(nodeL,edgeL,n,q)):
                update_ls.append(n)
        for n in update_ls:
            checklist.discard(n)
            update_neigbor(checklist,edgeL,n)
        calc_class(nodeL)

#check all nodes every itr
def cascading_slow(nodeL, edgeL, q, adt):
    res = []
    while(True):
        pres = res.copy()
        for n in range(nodeL[-1]['id']):
            if(change(nodeL,edgeL,n,q)):
                adt.append(n)
        res = calc_class(nodeL)
        if (pres!=[] and res == pres): break
    return res

def reverseid(nodeL, ls):
    rev = []
    for node in nodeL:
        if(node['id'] not in ls):
            rev.append(node['id'])
    return rev

#H = snap.TIntStrH()
#snap.SaveGViz(G, "Graph1.dot", "Undirected Random Graph", True, H)
#possible q
q = [0,0.05, 0.1, 0.15, 0.2, 0.25, 0.30, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.70, 0.75, 0.8, 0.85, 0.9, 0.95,1]

######### cascading #########
#cascading and store original distributiom and result to list
ls,ls_ori = [],[]
cluster = []
for i in range(21):
    ls.append([])
    ls_ori.append([])

for j in range(21):
    for i in range(21):
        reset(nodeL)
        x = int(nodeL[-1]['id']*q[j])
        adt = rand_adopter(nodeL,x)
        ls_ori[j].append(calc_class(nodeL))
        ls[j].append(cascading_slow(nodeL, edgeL, q[i], adt))

######### plot result #########
#avg of q val
lsn = ls.copy()
ls_orin = ls_ori.copy()
for i in range(21):
    lsn[i]=np.asarray(lsn[i])
    ls_orin[i]=np.asarray(ls_orin[i])
for i in range(21):
    lsn[i] = lsn[i].T[1]
    lsn[i] = sum(lsn[i])/21
    ls_orin[i] = ls_orin[i].T[1]
    ls_orin[i] = sum(ls_orin[i])/21
plt.plot(q,lsn,label='Random after cascading')
plt.plot(q,ls_orin,label='before cascading')
plt.title('Average Result of Cascading in Adopting New Product [2 class]')
plt.ylabel('Individuals that adopted new product')
plt.xlabel('Proportion of initial adopters')
plt.legend()
plt.savefig('avg_qval.png',bbox_inches='tight')
plt.show()
#make a copy for comparison w/ other cases
lsn1 = lsn.copy()
ls_orin1 = ls_orin.copy()

#plot result for diff q val and diff portion of initial adopter
for i in range(21):
    ls[i]=np.asarray(ls[i])
    ls_ori[i]=np.asarray(ls_ori[i])
for i in range(21):
    ls[i] = ls[i].T[1]
    ls_ori[i] = ls_ori[i].T[1]
#make a copy for comparison w/ other cases
ls1 = ls.copy()
ls_ori1 = ls_ori.copy()
fig = plt.figure(figsize=(10,13))
axes= fig.add_axes([0.1,0.1,0.8,0.8])
for i in range(1,20):
    lb = str(q[i])+' portion as initial adopters'
    axes.plot(q,ls[i],label = lb)
plt.title('Result of Cascading in Adopting New Product [2 class]')
plt.ylabel('Individuals that adopted new product')
plt.xlabel('q value')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.savefig('result_multiplecurve.png',bbox_inches='tight')
plt.show()

#plot result separately w.r.t the distribution before cascading
fig, ax = plt.subplots(7, 3, figsize=(13, 13))
fig.subplots_adjust(wspace = 0.3,hspace = 0.7)
for n in range(7):
    for k in range(3):
        ax[n,k].set_ylim([0, 90000])
        ax[n,k].set_title(str(q[n*3+k])+' portion as initial adopters')
        ax[n,k].plot(q,ls[n*3+k],label='after cascading')
        ax[n,k].plot(q,ls_ori[n*3+k],label='initial distribution')
        ax[n,k].legend(fancybox=True, framealpha=0.4)
plt.savefig('result_multiplot.png',bbox_inches='tight')
plt.show()

#average of initial portion distribution
sumls = sum(ls)/21
sumls_ori = sum(ls_ori)/21
plt.plot(q,sumls,label='Random after cascading')
plt.plot(q,sumls_ori,label='before cascading')
plt.title('Average Result of Cascading in Adopting New Product [2 class]')
plt.ylabel('Individuals that adopted new product')
plt.xlabel('q value')
plt.legend()
plt.savefig('avg_portion.png',bbox_inches='tight')
plt.show()

################################################################
################################################################
############################ CASE 2 ############################
################################################################
################################################################

#case1.1 KOL (topN leader)
def changeTopNKOL(nodeL,Graph,n):
    ls = []
    G = snap.ConvertGraph(type(Graph), Graph)
    for i in range(n):
        ls.append(snap.GetMxDegNId(G))
        nodeL[ls[-1]]['class']=1
        V = snap.TIntV()
        V.Add(ls[-1])
        snap.DelNodes(G, V)
    return ls

ls,ls_ori = [],[]
for i in range(21):
    ls.append([])
    ls_ori.append([])

for j in range(21):
    for i in range(21):
        reset(nodeL)
        x = int(nodeL[-1]['id']*q[j])
        adt = changeTopNKOL(nodeL,LG,x)
        ls_ori[j].append(calc_class(nodeL))
        ls[j].append(cascading_slow(nodeL, edgeL, q[i], adt))

#then repeat plotting and compare to previous plot
lsn = ls.copy()
ls_orin = ls_ori.copy()
for i in range(21):
    lsn[i]=np.asarray(lsn[i])
    ls_orin[i]=np.asarray(ls_orin[i])
for i in range(21):
    lsn[i] = lsn[i].T[1]
    lsn[i] = sum(lsn[i])/21
    ls_orin[i] = ls_orin[i].T[1]
    ls_orin[i] = sum(ls_orin[i])/21

#avg of q val
plt.plot(q,lsn1,label='Random after cascading')
plt.plot(q,lsn,label='TopN after cascading')
plt.plot(q,ls_orin,label='before cascading')
plt.title('Average Result of Cascading in Adopting New Product [2 class]')
plt.ylabel('Individuals that adopted new product')
plt.xlabel('Proportion of initial adopters')
plt.legend()
plt.savefig('avg_qval2.png',bbox_inches='tight')
plt.show()

#plot result for diff q val and diff portion of initial adopter
for i in range(21):
    ls[i]=np.asarray(ls[i])
    ls_ori[i]=np.asarray(ls_ori[i])
for i in range(21):
    ls[i] = ls[i].T[1]
    ls_ori[i] = ls_ori[i].T[1]
fig = plt.figure(figsize=(10,13))
axes= fig.add_axes([0.1,0.1,0.8,0.8])
for i in range(1,20):
    lb = str(q[i])+' portion as initial adopters'
    axes.plot(q,ls[i],label = lb)
plt.title('Result of Cascading in Adopting New Product [2 class]')
plt.ylabel('Individuals that adopted new product')
plt.xlabel('q value')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.savefig('result_multiplecurve2.png',bbox_inches='tight')
plt.show()

#plot result separately w.r.t the distribution before cascading
fig, ax = plt.subplots(7, 3, figsize=(13, 17))
fig.subplots_adjust(wspace = 1.3,hspace = 0.7)
for n in range(7):
    for k in range(3):
        ax[n,k].set_ylim([0, 90000])
        ax[n,k].set_title(str(q[n*3+k])+' portion as initial adopters')
        ax[n,k].plot(q,ls1[n*3+k],label='Rand cascaded')
        ax[n,k].plot(q,ls[n*3+k],label='TopN cascaded')
        ax[n,k].plot(q,ls_ori[n*3+k],label='Initial')
        ax[n,k].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fancybox=True, framealpha=0.4)
plt.savefig('result_multiplot2.png',bbox_inches='tight')
plt.show()

#average of initial portion distribution
sumls1 = sum(ls1)/21
sumls = sum(ls)/21
sumls_ori = sum(ls_ori)/21
plt.plot(q,sumls1,label='Random after cascading')
plt.plot(q,sumls,label='TopN after cascading')
plt.plot(q,sumls_ori,label='before cascading')
plt.title('Average Result of Cascading in Adopting New Product [2 class]')
plt.ylabel('Individuals that adopted new product')
plt.xlabel('q value')
plt.legend()
plt.savefig('avg_portion2.png',bbox_inches='tight')
plt.show()

################################################################
################################################################
############################ CASE 3 ############################
################################################################
################################################################
#find cluster relation
def findMinCC(Graph, rmls):
    G = snap.ConvertGraph(type(Graph), Graph)
    V = snap.TIntV()
    for i in rmls:
        V.Add(i)
    snap.DelNodes(G, V)
    NIdCCfH = snap.TIntFltH()
    snap.GetNodeClustCf(G, NIdCCfH)
    minid, minval = -1, 100
    for item in NIdCCfH:
        if(minval > NIdCCfH[item]):
            minid, minval = item, NIdCCfH[item]
    return [minid, minval]

def findAvgCC(Graph, rmls):
    G = snap.ConvertGraph(type(Graph), Graph)
    V = snap.TIntV()
    for i in rmls:
        V.Add(i)
    snap.DelNodes(G, V)
    return snap.GetClustCf(G, -1)

def DelNodeSmallDeg(deg,Graph,nodeL,edgeL):
    G = snap.ConvertGraph(type(Graph), Graph)
    V = snap.TIntV()
    for n in nodeL:
        if (len(edgeL[n['id']])<deg):
            V.Add(n['id'])
    snap.DelNodes(G, V)
    return G

def clusden(edgeL, adt):
    min = 1 
    for i in adt: 
        print('#',i)
        if (len(edgeL)==1 or len(edgeL)==0):
            return 0
        cnt = 0
        for j in edgeL[i]:
            if j in adt:
                cnt+=1
        tmp = cnt/len(edgeL[i])+0.0001
        if abs(tmp)>0.01 and tmp < min: #prevent len(edgeL) = 0
            min = tmp
    return min

G = DelNodeSmallDeg(10,LG,nodeL,edgeL)

q = [0.1, 0.2, 0.30, 0.4, 0.5, 0.6,0.70, 0.8, 0.9]
inadls,restls,ls,ls_ori  = [],[],[],[]
for i in range(9):
    inadls.append([])
    restls.append([])
    ls.append([])
    ls_ori.append([])

for j in range(9):
    for i in range(9):
        reset(nodeL)
        x = int(nodeL[-1]['id']*q[j])
        adt = rand_adopter(nodeL,x)
        ls_ori[j].append(calc_class(nodeL))
        ls[j].append(cascading_slow(nodeL, edgeL, q[i], adt))
        inadls[j].append(clusden(edgeL,adt))
        restls[j].append(clusden(edgeL,reverseid(nodeL, adt)))

#then start plotting the cluster coefficient
restls=np.asarray(restls)

#one plot with different q value and proportion of initial adopters
fig = plt.figure(figsize=(10,7))
axes= fig.add_axes([0.1,0.1,0.8,0.8])
for i in range(9):
    lb = str(q[i])+' portion as initial adopters'
    axes.plot(q,restls[i],label = lb)
plt.title('Cluster Density After Cascading')
plt.ylabel('Cluster Density')
plt.xlabel('q value')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.savefig('ClusDen.png',bbox_inches='tight')
plt.show()

#9 plot corresponse to 9 different proportion of initial adopters
q=np.asarray(q)
fig, ax = plt.subplots(3, 3, figsize=(13, 7))
fig.subplots_adjust(wspace = 1.3,hspace = 0.7)
for n in range(3):
    for k in range(3):
        ax[n,k].set_title(str(q[n*3+k])+' portion as initial adopters')
        ax[n,k].plot(q,restls[n*3+k],label='remaining network')
        ax[n,k].plot(q,1-q,label='1-q')
        ax[n,k].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fancybox=True, framealpha=0.4)
plt.savefig('result_multiplot.png',bbox_inches='tight')
plt.show()