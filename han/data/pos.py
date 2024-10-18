import numpy as np
import scipy.sparse as sp
from collections import Counter

####################################################
# This tool is to generate positive set with a thre-
# shold "pos_num".
# dataset  pos_num
# acm      5
# dblp     1000
# aminer   15
# freebase 80
#
#
# Notice: The best pos_num of acm is 7 reported in 
# paper, but we find there is no much difference 
# between 5 and 7 in practice.
####################################################

pos_num = 1000
p = 4019
apa = sp.load_npz("./dblp/apa.npz")
apa = apa / apa.sum(axis=-1).reshape(-1,1)
apcpa = sp.load_npz("./dblp/apcpa.npz")
apcpa = apcpa / apcpa.sum(axis=-1).reshape(-1,1)
aptpa = sp.load_npz("./dblp/aptpa.npz")
aptpa = aptpa / aptpa.sum(axis=-1).reshape(-1,1)
all = (apa + apcpa + apcpa).A.astype("float32")
all_ = (all>0).sum(-1)
print(all_.max(),all_.min(),all_.mean())

pos = np.zeros((p,p))
k=0
for i in range(len(all)):
  one = all[i].nonzero()[0]
  if len(one) > pos_num:
    oo = np.argsort(-all[i, one])
    sele = one[oo[:pos_num]]
    pos[i, sele] = 1
    k+=1
  else:
    pos[i, one] = 1
pos = sp.coo_matrix(pos)
sp.save_npz("pos.npz", pos)
