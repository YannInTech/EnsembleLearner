import numpy as np, pandas as pd
from sklearn.preprocessing import LabelEncoder
from model import BagLearner, DTLearner



def process(dataPath, testPath):
    data=pd.read_csv(dataPath,usecols=range(1,38)).to_numpy()
    X=data[:,:-1]
    y=data[:,-1]
    le = LabelEncoder()
    le.fit(y)
    encoded_y=le.transform(y)
    encoded_y=encoded_y.astype(np.int64)
    L=BagLearner(learner=DTLearner,kwargs={'depth':7},bags=10)
    L.add_evidence(X,encoded_y)
    X_test=pd.read_csv(testPath,usecols=range(1,37)).to_numpy()
    y_pred=L.query(X_test[:])
    decoded_y=le.inverse_transform(y_pred)
    decoded_y=decoded_y.reshape((-1,1))
    test_ids=pd.read_csv(testPath,usecols=range(1)).to_numpy(dtype=np.int64)
    submission=np.concat((test_ids[:],decoded_y),axis=1)
    submission=pd.DataFrame(submission,columns=['id','Target'])
    submission.to_csv('submission.csv',index=False)


    
process(dataPath='./data/train.csv',testPath='./data/test.csv')