
from sklearn import metrics
import operator
from munkres import Munkres
import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score


class clustering_metrics():
    def __init__(self, true_label, predict_label):
        self.true_label = true_label
        self.pred_label = predict_label

    def clusteringAcc(self):
        print(len(self.true_label), len(self.pred_label))
        # best mapping between true_label and predict label
        l1 = list(set(self.true_label))
        numclass1 = len(l1)

        l2 = list(set(self.pred_label))
        numclass2 = len(l2)
        if numclass1 != numclass2:
            print('Class Not equal!!!!')
            c1_clusters = {c: set() for c in set(l1)}
            c2_clusters = {c: set() for c in set(l2)}
            
            for i in range(len(self.true_label)):
                c1 = self.true_label[i]
                c2 = self.pred_label[i]
                c1_clusters[c1].add(i)
                c2_clusters[c2].add(i)

            c2_c1 = {}
            for c2 in set(l2):
                for c1 in set(l1):
                    c2_c1[str(c2)+","+str(c1)]=0


            for (c1, s1) in c1_clusters.items():
                for (c2, s2) in c2_clusters.items():
                    num_com_s1s2 = len(s1.intersection(s2))
                    c2_c1[str(c2)+","+str(c1)]=num_com_s1s2

            sorted_x = sorted(c2_c1.items(), key=operator.itemgetter(1), reverse=True)
            
            c2_c1_map = {}
            c1_flag = {c: True for c in set(l1)}
            c2_flag = {c: True for c in set(l2)}
            for (k, v) in sorted_x:
                if len(c2_c1_map.keys())==numclass1:
                    break
                c2, c1 = k.split(',')
                c2, c1 = int(c2), int(c1)
                #print(c2, c1, v)
                if c1_flag[c1] and c2_flag[c2]:
                    c2_c1_map[c2]=c1

                c1_flag[c1] = False
                c2_flag[c2] = False
            
            new_predict = np.zeros(len(self.pred_label))
            for i in range(len(l2)):
                new_predict[i] = c2_c1_map[self.pred_label[i]]
                
        else:
            cost = np.zeros((numclass1, numclass2), dtype=int)
            for i, c1 in enumerate(l1):
                mps = [i1 for i1, e1 in enumerate(self.true_label) if e1 == c1]
                for j, c2 in enumerate(l2):
                    mps_d = [i1 for i1 in mps if self.pred_label[i1] == c2]

                    cost[i][j] = len(mps_d)

            # match two clustering results by Munkres algorithm
            m = Munkres()
            cost = cost.__neg__().tolist()

            indexes = m.compute(cost)

            # get the match results
            new_predict = np.zeros(len(self.pred_label))
            for i, c in enumerate(l1):
                # correponding label in l2:
                c2 = l2[indexes[i][1]]

                # ai is the index with label==c2 in the pred_label list
                ai = [ind for ind, elm in enumerate(self.pred_label) if elm == c2]
                new_predict[ai] = c

        acc = metrics.accuracy_score(self.true_label, new_predict)
        f1_macro = metrics.f1_score(self.true_label, new_predict, average='macro')
        precision_macro = metrics.precision_score(self.true_label, new_predict, average='macro')
        recall_macro = metrics.recall_score(self.true_label, new_predict, average='macro')
        f1_micro = metrics.f1_score(self.true_label, new_predict, average='micro')
        precision_micro = metrics.precision_score(self.true_label, new_predict, average='micro')
        recall_micro = metrics.recall_score(self.true_label, new_predict, average='micro')
        return acc

    def evaluationClusterModelFromLabel(self):
        nmi = metrics.normalized_mutual_info_score(self.true_label, self.pred_label)
        adjscore = metrics.adjusted_rand_score(self.true_label, self.pred_label)
        acc = self.clusteringAcc()
        ari = adjusted_rand_score(self.true_label, self.pred_label)

        return acc, nmi, adjscore, ari