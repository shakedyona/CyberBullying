import pandas as pd


def write_performances(folder_name,auc_list,performances_list,datetime_object):
    performances_df = pd.DataFrame(columns=['name', 'date', 'f-score', 'precision', 'recall', 'auc'])
    for name, per in performances_list.items():
        performances_df = performances_df.append({'name': name, 'date': datetime_object, 'f-score': per['f-score'], 'precision': per['precision'], 'recall': per['recall'], 'auc': auc_list[name]}, ignore_index=True)
    performances_df.to_csv(folder_name+r'\logger.csv', sep=',', mode='a')


def write_features(folder_name,features):
    my_features = open(folder_name+r'\features.txt', 'w')
    for f in features:
        my_features.write(str(f)+'\n')
    my_features.close()


