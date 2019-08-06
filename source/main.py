from source import api
import os.path
SOURCE = os.path.join(__file__, '../')

api.train(os.path.join(SOURCE, 'dataNew.csv'))
pred = api.get_classification('וואי את ילדה מאוד מטומטמת', True)
print(pred)
performances = api.get_performance(os.path.join(SOURCE, 'dataNew.csv'))
print(performances)

