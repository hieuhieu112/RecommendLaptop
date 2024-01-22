import pickle
import pandas as pd

with open ('knn_model_laptop.pickle','rb') as f:
  knn_new = pickle.load(f)
with open ('scaler_laptop.pickle','rb') as f:
  scaler_new = pickle.load(f)

new_df = pd.read_csv("hihi.csv")
print (new_df)

x_new = new_df.to_numpy()
x_new_scale2 = scaler_new.fit_transform (x_new)
y_new_pred = knn_new.predict(x_new_scale2)

new_df['will_purchase'] = y_new_pred
new_df = new_df[new_df['will_purchase'] == 1]
print(new_df)