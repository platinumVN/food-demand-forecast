import streamlit as st
import pickle
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import r2_score,mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


def center_id(datacol):
    center_id_val_index_n = []
    for i in datacol:
        if i >= 10 and i <= 30:
            center_id_val_index_n.append("10-30")
        elif i >= 31 and i <=50:
            center_id_val_index_n.append("31-50")
        elif i >= 51 and i <=70:
            center_id_val_index_n.append("51-70")  
        elif i >= 71 and i <=90:
            center_id_val_index_n.append("71-90")
        elif i >= 91 and i <=110:
            center_id_val_index_n.append("91-110") 
        elif i >= 111 and i <=130:
            center_id_val_index_n.append("111-130")
        elif i >= 131 and i <=150:
            center_id_val_index_n.append("131-150")          
        else:
            center_id_val_index_n.append("151-190")
    
    return  center_id_val_index_n 

def meal_id(datacol):        
    meal_id_val_index_n = []
    for i in datacol:
        if i >= 1000 and i <= 1300:
            meal_id_val_index_n.append("1000-1300")
        elif i >= 1301 and i <=1600:
            meal_id_val_index_n.append("1301-1600")
        elif i >= 1601 and i <=1900:
            meal_id_val_index_n.append("1601-1900")  
        elif i >= 1901 and i <=2200:
            meal_id_val_index_n.append("1901-2200")
        elif i >= 2201 and i <=2500:
            meal_id_val_index_n.append("2201-2500") 
        elif i >= 2501 and i <=2800:
            meal_id_val_index_n.append("2501-2800")          
        else:
            meal_id_val_index_n.append("2801-3000") 
    return  meal_id_val_index_n

st.write("""
## Demand Forecasting Example
---
- Link to Slides: https://docs.google.com/presentation/d/1YDt9aR-OKCc5c8nbO74LVWCJ648hDRrs/edit?usp=sharing&ouid=100988340872896613643&rtpof=true&sd=true

    =============== Cách sử dụng ===============
    BƯỚC 1:
    - Upload file .csv input data 
    - hoặc nhập thủ công giá trị của từng biến đầu vào
    BƯỚC 2:
    - Nhấp chuột chọn "Predict"

###### Tải file test tại: https://drive.google.com/file/d/1vaHWY9PQApvjz8cgE8HlK8RGRPIb-QdI/view?usp=share_link
---
""")

st.sidebar.header('Input Data')

uploaded_file = st.sidebar.file_uploader("Cách 1: Upload your input CSV file", type=["csv"])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        st.sidebar.write('Cách 2: Manually input your data')
        week = st.sidebar.number_input('Week to predict', format='%d', min_value = 146, max_value=155,step = 1)
        center_id = st.sidebar.selectbox('center_id',(55,  24,  11,  83,  32,  13, 109,  52,  93, 186, 146,  57, 149,
                                        89, 124, 152,  97,  74, 108,  99,  66,  94,  91,  20,  34, 137,
                                        92, 126,  36, 162,  75, 177,  27, 157, 106,  64, 129,  14,  17,
                                        153, 139, 161,  81,  26,  73,  50, 104,  42, 113, 145,  53,  72,
                                        67, 174,  29,  77,  41,  30,  76,  59,  88, 143,  58,  10, 101,
                                        80,  43,  65,  39, 102, 110, 132,  23,  86,  68,  51,  61))
        meal_id = st.sidebar.selectbox('meal_id',(1885, 1993, 2539, 2139, 2631, 1248, 1778, 1062, 2707, 1207, 1230,
                                        2322, 2290, 1727, 1109, 2640, 2306, 2126, 2826, 1754, 1971, 1902,
                                        1311, 1803, 1558, 2581, 1962, 1445, 2444, 2867, 1525, 2704, 2304,
                                        2577, 1878, 1216, 1247, 1770, 1198, 1438, 2494, 1847, 2760, 2492,
                                        1543, 2664, 2569, 2490, 1571, 2956, 2104))

        emailer_for_promotion = st.sidebar.selectbox('emailer_for_promotion',(1, 0))
        homepage_featured = st.sidebar.selectbox('homepage_featured',(1, 0))  
        checkout_price = st.sidebar.number_input('Week to predict', format="%.2f", min_value = 2.97, max_value= 866.27)
        base_price = st.sidebar.number_input('Week to predict', format="%.2f", min_value = 55.35, max_value= 866.27)

        data = {'week' : week,
                'checkout_price' : checkout_price, 
                'base_price' : base_price,
                'center_id' : center_id,
                'meal_id' : meal_id,
                'emailer_for_promotion' : emailer_for_promotion,
                'homepage_featured' : homepage_featured
                }
        
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features() #input_df

exec_button = st.sidebar.button('Predict')
if exec_button:
    center_id_val_index_n = center_id(input_df.center_id) 
    input_df.center_id = center_id_val_index_n
    meal_id_val_index_n = meal_id(input_df.meal_id)
    input_df.meal_id = meal_id_val_index_n
    f_input = input_df.loc[:,['week','center_id','meal_id','checkout_price','base_price','emailer_for_promotion',
                    'homepage_featured']]
    final_input = pd.get_dummies(f_input)

    st.write('Your input:')
    st.write(input_df)

    RFRmodel = pickle.load(open('finalized_model.pkl', 'rb'))
    df_predict = RFRmodel.predict(final_input)
    input_df['num_orders'] = df_predict
    sample =  input_df.loc[:,['id','num_orders']]

    st.subheader('Prediction result:')
    sample =  input_df.loc[:,['id','num_orders']]
    st.write(sample)
