from flask import Flask, request, jsonify
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas.core.series
import pickle

app = Flask(__name__)
@app.route("/recommend/nameproduct/<nameProduct>")
def get_recommendationName(nameProduct):
    df = pd.read_csv('namproducts.csv')

    df['NameProduct'] = df['NameProduct'].str.lower().replace(['^a-zA-Z0-9'], " ", regex=True)

    stemr = SnowballStemmer('english')  # chuyển từ tiếng anh bỏ bớt các hậu tố

    def tokenization(txt):
        tokens = nltk.word_tokenize(txt)
        stemming = [stemr.stem(w) for w in tokens if w not in stopwords.words('english')]
        return " ".join(stemming)

    df['NameProduct'] = df['NameProduct'].apply(lambda x: tokenization(x))

    def cosine_sim(txt1, txt2):
        obj_ifid = TfidfVectorizer(tokenizer=tokenization)
        matrix = obj_ifid.fit_transform([txt1, txt2])
        similarity = cosine_similarity(matrix)[0][1]
        return similarity

    def recommendation(query):
        tokenized_query = tokenization(query)
        df['similarity'] = df['NameProduct'].apply(lambda x: cosine_sim(tokenized_query, x))
        final_df = df.sort_values(by=['similarity'], ascending=False)
        # Lấy 4 phần tử đầu tiên
        final_df = final_df.iloc[1:5]
        return final_df
    results = recommendation(nameProduct)
    results_json = results.to_json(orient='records')  # Chuyển đổi DataFrame thành chuỗi JSON
    return results_json

#Recommender System Orderdetails

@app.route("/recommend/order/<cart_list>")
def get_recommendationsOrder(cart_list):
    # Chuyển đối số từ URL thành danh sách
    cart_list = cart_list.split(',')

    data = pd.read_csv('oderdetails.csv', encoding='latin-1')
    # dataset
    data[['OrderID', 'ProductID']] = data[['OrderID', 'ProductID']].astype(str)
    subset = data[['OrderID', 'ProductID']]  # tạo danh sách mã hóa đơn và mã sản phẩm

    # cập nhật dataset với cart_list
    def add_new(cart_list, subset):
        for item in cart_list:
            subset = subset._append({'OrderID': 'cart_list', 'ProductID': item}, ignore_index=True)
        return subset

    # Chuyển dataset thành matrix
    # Tính luôn cosine_sim
    def get_cosine_sim(subset):
        pivot_tb = pd.crosstab(index=subset.OrderID, columns=subset.ProductID)
        cosine_sim = cosine_similarity(pivot_tb.to_numpy())
        # taạo 1 biến indices tương ứng với inVOiceNo
        pivot_tb.index = pivot_tb.index.astype(str) #chuyển về string
        indices = pd.Series(pivot_tb.index.drop_duplicates().sort_values())
        cart_list_loc = pivot_tb.index.get_loc('cart_list')
        return cosine_sim, indices, cart_list_loc

    def recommender(cosine_sim, indices, cart_list_loc):
        # lọc ra n biggest cosine_sim
        item_in_cart = cosine_sim[cart_list_loc]
        with_i = list(enumerate(item_in_cart))
        with_i = sorted(with_i, key=lambda x: x[1], reverse=True)
        top5 = with_i[1:6]
        results = []
        for i, j in top5:
            results.append([indices[i], j])
        results = pd.DataFrame(results)
        results.columns = ['OrderID', 'Similarity']
        return results

    subset = add_new(cart_list=cart_list, subset=subset)
    cosine_sim, indices, cart_list_loc = get_cosine_sim(subset=subset)
    results = recommender(cosine_sim, indices, cart_list_loc)
    results_json = results.to_json(orient='records')  # Chuyển đổi DataFrame thành chuỗi JSON
    return results_json


#Dự đoán hành vi mua hàng của khách hàng
@app.route("/recommend/purchase")
def get_recommendationsPurchase():
    with open('knn_model_laptop.pickle', 'rb') as f:
        knn_new = pickle.load(f)
    with open('scaler_laptop.pickle', 'rb') as f:
        scaler_new = pickle.load(f)

    new_df = pd.read_csv("hihi.csv")
    x_new = new_df.to_numpy()
    x_new_scale2 = scaler_new.fit_transform(x_new)
    y_new_pred = knn_new.predict(x_new_scale2)

    new_df['will_purchase'] = y_new_pred
    new_df = new_df[['Productid','will_purchase']]
    new_df = new_df[new_df['will_purchase'] == 1]
    results_json = new_df.to_json(orient='records')  # Chuyển đổi DataFrame thành chuỗi JSON
    return results_json



if __name__ == '__main__':
    app.run(debug=True)