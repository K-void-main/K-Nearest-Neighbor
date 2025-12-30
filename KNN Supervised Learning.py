# %% Importing Libraries
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# %% KNN Supervised Learning Example
X = np.array([[1],[2],[3],[4],[5]])
y = np.array([0, 0, 0, 1, 1])

# %% Training the KNN Classifier
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

# %% Making Predictions
print(model.predict([[7]]))
print(model.predict([[3]]))
print(model.predict([[5]]))
# %%
