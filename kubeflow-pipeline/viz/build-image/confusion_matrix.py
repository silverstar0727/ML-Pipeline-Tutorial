import json
import os

import pandas as pd
from sklearn.metrics import confusion_matrix

y_target = [2, 0, 2, 2, 0, 1]
y_pred = [0, 0, 2, 2, 0, 2]

vocab = [0, 1, 2]

cm = confusion_matrix(y_target, y_pred, labels=vocab)

data = []
for target_index, target_row in enumerate(cm):
    for predicted_index, count in enumerate(target_row):
        data.append((vocab[target_index], vocab[predicted_index], count))

df_cm = pd.DataFrame(data, columns=['target', 'predicted', 'count'])

output = '.'
cm_file = os.path.join(output, 'confusion_matrix.csv')
with open(cm_file, 'w') as f:
    df_cm.to_csv(f, columns=['target', 'predicted', 'count'], header=False, index=False)

lines = ''
with open(cm_file, 'r') as f:
    lines = f.read()

metadata = {
    'outputs': [{
            'type': 'confusion_matrix',
            'format': 'csv',
            'schema': [
                {'name': 'target', 'type': 'CATEGORY'},
                {'name': 'predicted', 'type': 'CATEGORY'},
                {'name': 'count', 'type': 'NUMBER'},
            ],
            'source': lines,
            'storage': 'inline',
            'labels': list(map(str, vocab)),
        }]
}

with open('/mlpipeline-ui-metadata.json', 'w') as f:
    json.dump(metadata, f)