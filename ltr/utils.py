import pandas as pd
from sklearn.metrics import ndcg_score

import numpy as np

def get_ndcg(model, df, true_score, query_id, k=None):
    predicted_score = model.predict(df)
    
    ndcg_df = pd.DataFrame({'query_id': query_id, 'true_score': true_score, 'predicted_score': predicted_score})
    
    true_score_test = ndcg_df.groupby(['query_id'])['true_score'].apply(list).tolist()
    predicted_score_test = ndcg_df.groupby(['query_id'])['predicted_score'].apply(list).tolist()

    return np.mean([ndcg_score([_true], [_predicted], k=k) for _true, _predicted in zip(true_score_test, predicted_score_test) if len(_true) > 1])

def read_data(fold, filename):
    df = pd.read_csv(f'/mnt/d/datasets/MSLR-WEB10K/{fold}/{filename}.txt', delimiter=" ", header=None)
    df = df.applymap(lambda x: x.split(":", 1)[-1] if type(x) == str else x)

    y = df[0]
    query_id = df[1]

    group = df.groupby(1) # Group by based on query id
    group_size = group.size().to_list()

    df = df.drop([0, 1], axis=1) # Drop label and query id column
    df = df.dropna(axis=1, how='all')
    df = df.astype(float) # Turn all to float
    

    return df, y, group_size, query_id