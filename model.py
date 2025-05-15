import numpy as np




def gini_impurity(class_vector):

    values, counts=np.unique(class_vector, return_counts=True)
    cardinal=np.sum(counts)
    pi=[counts[i]/cardinal for i in range(len(counts))]
    return 1-np.sum(np.power(pi,2))

def gini_gain(previous_classes, current_classes):

    values, counts_previous=np.unique(previous_classes, return_counts=True)
    cardinal_previous=np.sum(counts_previous)
    remainder=0
    for i in range(len(current_classes)):
        values, counts=np.unique(current_classes[i], return_counts=True)
        cardinal=np.sum(counts)
        remainder+=(cardinal/cardinal_previous)*gini_impurity(current_classes[i])
    return gini_impurity(previous_classes)-remainder

class DTLearner():  		  	   		  	  			  		 			     			  	 
		  	   		  	  			  		 			     			  	 
    def __init__(self, depth, **kwargs):  		  	   		  	  			  		 			     			  	 
        self.depth=depth
        self.tree=kwargs.get('tree')	  	   		  	  			  		 			     			  	 	  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
    def add_evidence(self, X_train, y_train):
        X=np.copy(X_train)
        y=np.copy(y_train)
        node=0
        self.tree=self.build_tree(X,y,node)

    def build_tree(self, X, y, node):

        if X.shape[0]==1 or np.unique(y).size==1 or node>self.depth:
            node+=1
            values, counts=np.unique(y, return_counts=True)
            y_pred=values[np.argmax(counts)]
            return np.asarray([[-1,y_pred,-1,-1]])

        else:
            quantiles=np.quantile(X,0.5,axis=0)
            sub=[np.where(np.ma.masked_where(X[:,j]<=quantiles[j],X[:,j]).mask) for j in range(X.shape[1])]
            up=[np.where(np.ma.masked_where(X[:,j]>=quantiles[j],X[:,j]).mask) for j in range(X.shape[1])]
            classes_fifth=[[y[sub[j]],y[up[j]]] for j in range(X.shape[1])]  	  			  		 			     			  	  		  	   		  	  			  		 			     			  	 
            gini_gains=[gini_gain(y,classes_fifth[j]) for j in range(X.shape[1])]
            best_f=np.argmax(gini_gains)
            split=np.quantile(X[:,best_f],0.5)

            sub_split_indexes=[i for i,x in enumerate(X.T[best_f]) if x<=split]
            up_split_indexes=[i for i,x in enumerate(X.T[best_f]) if x>split]

            if not(up_split_indexes):
                values, counts=np.unique(y, return_counts=True)
                y_pred=values[np.argmax(counts)]
                return np.asarray([[-1,y_pred,-1,-1]])

            node+=1
            left=self.build_tree(X[sub_split_indexes],y[sub_split_indexes],node)
            right=self.build_tree(X[up_split_indexes],y[up_split_indexes],node)
            root=np.asarray([[best_f,split,1,left.shape[0]+1]])
            return np.concatenate((root,left,right),axis=0)

    def query(self, X_test):

        y_predict=np.zeros(X_test.shape[0])
        for i in range(X_test.shape[0]):
            k=0
            while k!=-1:
                if int(self.tree[k,2])==-1 or int(self.tree[k,3])==-1: break
                elif X_test[i].T[int(self.tree[k,0])] <= self.tree[k,1]: k=k+int(self.tree[k,2])
                else: k=k+int(self.tree[k,3])
                
            y_predict[i]=self.tree[k,1]
        return y_predict
    

class BagLearner():

    def __init__(self, learner, bags, kwargs):  		  	   		  	  			  		 			     			  	 
        self.bags=bags
        self.learners=[learner(**kwargs) for _ in range(bags)]

    def add_evidence(self, X_train, y_train):
        X=np.copy(X_train)
        y=np.copy(y_train)
        bags_indexes=[np.random.choice(X_train.shape[0],X_train.shape[0],replace=False) for _ in range(self.bags)]
        for i,l in enumerate(self.learners):
            l.add_evidence(X[bags_indexes[i]],y[bags_indexes[i]])

    def query(self, X_test):
        bag_predict=[l.query(X_test) for l in self.learners]
        samples_trees=np.array(bag_predict).T
        samples_values_counts=[np.unique(samples_trees[s], return_counts=True) for s in range(samples_trees.shape[0])]
        y_pred=[samples_values_counts[s][0][np.argmax(samples_values_counts[s][1])] for s in range(samples_trees.shape[0])]
        return np.array(y_pred, dtype=np.int64)