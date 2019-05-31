from source import api


api.train_file('source/dataNew.csv')
# pred = api.predict('היי מה שלומך?', True)
# print(pred)
performances = api.get_performances('source/dataNew.csv')
print(performances)
