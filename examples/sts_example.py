import sys
sys.path.append('..')
import SequenceClassificationUtils as U

results = U.glue('config.json')
print(results)
