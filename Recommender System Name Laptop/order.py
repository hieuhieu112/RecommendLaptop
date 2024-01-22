import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv('oderdetails.csv',encoding= 'latin-1')
print(data)
#dataset
data[['OrderID', 'ProductID']] = data[['OrderID', 'ProductID']].astype(str)
subset = data[['OrderID','ProductID']] #tạo danh sách mã hóa đơn và mã sản phẩm
cart_list = []
#cập nhật dataset với cart_list
def add_new(cart_list, subset):
    for item in cart_list:
        subset=subset._append({'OrderID':'cart_list',
                             'ProductID': item}, ignore_index = True)
    return subset

#Chuyển dataset thành matrix
#Tính luôn cosine_sim
def get_cosine_sim(subset):
    pivot_tb = pd.crosstab(index = subset.OrderID , columns= subset.ProductID)
    print(pivot_tb)
    cosine_sim = cosine_similarity(pivot_tb.to_numpy())
    # taạo 1 biến indices tương ứng với inVOiceNo
    indices = pd.Series(pivot_tb.index.drop_duplicates().sort_values())
    cart_list_loc  = pivot_tb.index.get_loc('cart_list')
    return  cosine_sim, indices,cart_list_loc
def recommender (cosine_sim, indices,cart_list_loc):
    #lọc ra n  biggest cosine_sim
    item_in_cart = cosine_sim[cart_list_loc]
    with_i = list(enumerate(item_in_cart))
    with_i = sorted(with_i, key = lambda  x:x[1], reverse= True)
    top5 = with_i[1:6]
    results = []
    for i,j in top5:
        results.append([indices[i],j])
    results = pd.DataFrame(results)
    results.columns = ['OrderID','Similarity']
    return results

cart_list=['20','25','27']
subset = add_new(cart_list=cart_list, subset=subset)
cosine_sim, indices,cart_list_loc = get_cosine_sim(subset=subset)
results =recommender(cosine_sim,indices,cart_list_loc)
print(results)