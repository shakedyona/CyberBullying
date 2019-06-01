from source import api
import os.path
SOURCE = os.path.join(__file__, '../')

api.train_file(os.path.join(SOURCE, 'dataNew.csv'))
pred = api.predict('וואי שמלה בת זונה', True)
print(pred)
performances = api.get_performances(os.path.join(SOURCE, 'dataNew.csv'))
print(performances)

