import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

model = joblib.load('svm_model2.pkl')
X_test = pd.read_csv ('X_test.csv')
y_test = pd.read_csv ('y_test.csv')
y_test_values = y_test['hua'].values

feature_names = ["bmi_z","sex","prealbumin","crea","age"]
st.title ("Prediction Model for Hyperuricemia in Pediatric Hypertension")
Age = st.number_input('Age (6-17 years):', min_value=6, max_value=18, value = 12)
age = (Age-12.51221)/2.235874
sex = st.selectbox("Sex:", options =[0, 1], format_func = lambda x:"Boys" if x==0 else "Girls")
Prealbumin = st.number_input('Prealbumin (mg/L):', min_value=1, max_value=1000, value = 41)
prealbumin = (Prealbumin-235.2251)/43.65179
Crea = st.number_input("Crea (μmol/L):", min_value = 1, max_value = 200, value = 41)
crea = (Crea-53.54569)/12.28562
BMI = st.number_input("BMI (kg/m^2):", min_value = 1, max_value = 100, value = 10)
# 定义LMS数据
boys_data = pd.DataFrame({
    'age': [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 
            10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0],
    'L': [-0.0293, -0.1935, -0.5351, -0.6372, -0.8185, -0.9278, -1.0161, -1.0837, -1.1325, -1.1647, -1.1822, -1.1871, 
          -1.1819, -1.1691, -1.1509, -1.1293, -1.1058, -1.0816, -1.0576, -1.0344, -1.0126, -0.9924, -0.9741, -0.9577, 
          -0.9430, -0.9301, -0.9187, -0.9085, -0.8993, -0.8908, -0.8830, -0.8756, -0.8685, -0.8617, -0.8552, -0.8487, -0.8425],
    'M': [13.0666, 17.9640, 17.1907, 16.4742, 16.3265, 15.9658, 15.6597, 15.4508, 15.3189, 15.2341, 15.2236, 15.2721, 
          15.3458, 15.4512, 15.5950, 15.7688, 15.9650, 16.1830, 16.4229, 16.6819, 16.9567, 17.2429, 17.5353, 17.8273, 
          18.1137, 18.3942, 18.6687, 18.9345, 19.1891, 19.4318, 19.6627, 19.8823, 20.0914, 20.2909, 20.4821, 20.6656, 20.8424],
    'S': [0.0837, 0.0875, 0.0814, 0.0787, 0.0761, 0.0748, 0.0745, 0.0752, 0.0767, 0.0790, 0.0823, 0.0866, 0.0916, 0.0971, 
          0.1027, 0.1084, 0.1140, 0.1192, 0.1240, 0.1283, 0.1320, 0.1352, 0.1379, 0.1401, 0.1418, 0.1432, 0.1443, 0.1451, 
          0.1456, 0.1460, 0.1463, 0.1464, 0.1465, 0.1465, 0.1465, 0.1464, 0.1464]
})

girls_data = pd.DataFrame({
    'age': [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 
            10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0],
    'L': [-0.4355, -0.4536, -0.5721, -0.6589, -0.8350, -0.8991, -0.9480, -0.9848, -1.0121, -1.0321, -1.0463, -1.0559, 
          -1.0621, -1.0657, -1.0673, -1.0674, -1.0665, -1.0649, -1.0629, -1.0606, -1.0582, -1.0559, -1.0537, -1.0517, 
          -1.0501, -1.0488, -1.0478, -1.0470, -1.0465, -1.0462, -1.0460, -1.0458, -1.0457, -1.0456, -1.0455, -1.0454, -1.0453],
    'M': [13.0027, 17.4082, 16.7427, 16.0306, 15.9459, 15.6371, 15.4167, 15.2686, 15.1544, 15.0632, 14.9936, 14.9624, 
          14.9602, 14.9712, 15.0154, 15.0963, 15.2142, 15.3730, 15.5748, 15.8167, 16.0946, 16.4034, 16.7377, 17.0885, 
          17.4471, 17.8038, 18.1500, 18.4784, 18.7831, 19.0605, 19.3083, 19.5273, 19.7197, 19.8898, 20.0427, 20.1834, 20.3165],
    'S': [0.0878, 0.0835, 0.0790, 0.0775, 0.0770, 0.0771, 0.0773, 0.0782, 0.0801, 0.0826, 0.0854, 0.0884, 0.0915, 0.0949, 
          0.0984, 0.1022, 0.1062, 0.1103, 0.1142, 0.1180, 0.1215, 0.1245, 0.1270, 0.1290, 0.1305, 0.1317, 0.1325, 0.1331, 
          0.1334, 0.1336, 0.1337, 0.1337, 0.1337, 0.1336, 0.1335, 0.1334, 0.1333]
})

# 创建LMS数据字典
lms_data = {'boys': boys_data, 'girls': girls_data}

def calculate_bmi_zscore(age, bmi, sex, lms_data):
    """
    计算BMI Z-score的函数
    
    参数:
    age: 年龄 (年)
    bmi: BMI值
    sex: 性别 (1=男童, 2=女童)
    lms_data: 包含男女童LMS数据的字典
    
    返回:
    z_score: BMI Z-score
    """
    # 根据性别选择对应的LMS数据
    if sex == 0:  # 男童
        data = lms_data['boys']
    elif sex == 1:  # 女童
        data = lms_data['girls']
    else:
        # 如果不是1或2，返回NaN
        return np.nan
    
    # 确保年龄是数值类型
    try:
        age = float(age)
    except ValueError:
        return np.nan
    
    # 找到对应年龄的索引
    # 由于年龄可能是小数，使用最接近的匹配
    age_idx = None
    min_diff = float('inf')
    
    for i, row_age in enumerate(data['age']):
        diff = abs(row_age - age)
        if diff < min_diff:
            min_diff = diff
            age_idx = i
    
    # 如果年龄超出范围，返回NaN
    if age_idx is None:
        return np.nan
    
    # 获取对应的L, M, S值
    L = data.loc[age_idx, 'L']
    M = data.loc[age_idx, 'M']
    S = data.loc[age_idx, 'S']
    
    # 计算Z-score (根据LMS方法)
    try:
        # 处理L接近0的情况
        if abs(L) < 1e-10:
            # 当L接近0时，使用极限形式
            z_score = (np.log(bmi / M)) / S
        else:
            z_score = ((bmi / M) ** L - 1) / (L * S)
    except:
        # 计算错误时返回NaN
        z_score = np.nan
    
    return z_score
bmi_z = calculate_bmi_zscore(Age, BMI, sex, lms_data)
feature_values = [bmi_z, sex, prealbumin, crea, age]
features = np.array([feature_values])
if st.button('Predict'):
    predict_class = model.predict(features)[0]
    predict_proba = model.predict_proba (features)[0]
    # 优化显示结果 - 加大、加粗、加颜色
    st.markdown(f"<h2 style='color: #1E88E5; font-weight: bold;'>Predicted Probabilities: {predict_proba[1]:.3f}</h2>", 
                unsafe_allow_html=True)
    
    # 根据预测结果设置不同颜色
    if predict_class == 1:
        st.markdown(f"<h1 style='color: #D32F2F; font-weight: bold; font-size: 32px;'>Predicted Class: {predict_class} (Hyperuricemia)</h1>", 
                    unsafe_allow_html=True)
    else:
        st.markdown(f"<h1 style='color: #388E3C; font-weight: bold; font-size: 32px;'>Predicted Class: {predict_class} (No hyperuricemia)</h1>", 
                    unsafe_allow_html=True)

