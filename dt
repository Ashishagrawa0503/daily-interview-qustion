class DecisionTree:
  def __init__(self,max_depth=None):#ye depth aapan khud dalte hai ke kitne depth ka decision tree baanana hai 2,4,ect
    self.max_depth=max_depth

   def fit(self,X,y):
    self.n_classes=len(set(y))#y ke jitne be unique classes hai  unke length
    self.n_feature=X.shape[1]# number of colums in data
    self.tree=self._grow_tree(X,y)

   def _grow_tree(self,X,y,depth=0):
    num_samples_per_class=[]
    for i in range(self.n_classes):
      count=np.sum(y == i) #y main kitne baar i aaya hai
      num_samples_per_class.append(count) 
    predicted_class=np.argmax(num_samples_per_calss)  #num_samples_per_calss ke subse bade value ke index ko deta hai
    node=Node(predicted_class=predicted_class)
    if depth < self.max_depth:
      idx,thr= self._best_split(X,y)
      if idx is not None:
        indices_left=X[:,idx]<thr# maano thr =0.5 hai to 0.5 se choti value true or bade value false ho jayenge 
        X_left,y_left=X[indices_left],y[indices_left]
        X_right,y_right=X[~indices_left],y[~indices_left]# "~" ye bitwise operater hai jo true ko false or false ko true kr dega
        node.feature_index=idx
        node.threshold=thr
        node.left=self._grow_tree(X_left,y_left,depth+1)
        node.right=self._grow_tree(X_right,y_right,depth+1)
    return node


class _best_split(self,X,y):
  m,n=X.shape
  if m<=1: # data ke number of rows
     return None, None
  num_parent= []
  for c in range(self.n_classes):
      cnt=np.sum(y == i) #y main kitne baar c aaya hai
      num_parent.append(cnt)
  best_gini = 1.0-sum((num/m)**2 for num in num_parent)
  best_idx,best_thr= None,None
  for idx in range(n):
    threshold,classes=zip(*sorted(zip(X[:,idx],y)))    
#x main jitne be row hai unke unique kitne baar aaye hai uske idx,y ko zip yane list ke andar bhar do 
     num_left=[0]*self.n_classes
     num_right=num_parent .copy()

     for i in range(1,m):
      c=classes[i-1]# upper jo classes or threshold nikala hai usme claases main i-1 wala indx
      num_left[c]+=1
      num_right[c]-=1#num_left or num_right be upper nikala hai
      gini_left=1.0#gini ka formula
      if i!=0:
        sum_sq=0.0
        for x in range(self.n_classes):
          p=num_left[X]/i # xth class ka proportion
          sum_sq += p**2 #squre
        gini_left -= sum_sq  
      gini_right=1.0 - sum((num_right[X] /(m-i))**2 for x in range(self.n_classes) if (m-i) !=0)
      gini =(i * gini_left + (m-i)*gini_right) / m

       if thresholds[i] == thresholds[i - 1]:
                    continue

                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2 

        return best_idx, best_thr  
