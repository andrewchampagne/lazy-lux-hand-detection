import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


dict = pickle.load(open('./data.pickle', 'rb'))

data = np.asarray(dict['features'])
labels = np.array(dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.15, train_size=0.85, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print(accuracy_score(y_pred, y_test))

f = open('model.p', 'wb')
pickle.dump({'model':model}, f)
f.close()