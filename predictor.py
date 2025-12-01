{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "7a5ae948-8a58-4d35-9162-6e23c9bdbb59",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import joblib\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import shap\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "5f66d54d-b9c2-4b01-8a2a-1a1b94d1adf6",
   "metadata": {},
   "outputs": [],
   "source": [
    "model = joblib.load('svm_model.pkl')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "46a9e028-7dc8-496d-a2e0-4008af6e41ab",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_test = pd.read_csv ('X_test.csv')\n",
    "y_test = pd.read_csv ('y_test.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "128ace9a-d9e9-46cf-94a1-7a257db7cc38",
   "metadata": {},
   "outputs": [],
   "source": [
    "feature_names = [\"bmi_z\",\"sex\",\"prealbumin\",\"crea\",\"age\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "fbb6b525-a3c2-4994-be42-4be86b1d1252",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-12-01 20:52:00.429 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-01 20:52:01.429 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run D:\\softwares\\Anaconda\\Lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n",
      "2025-12-01 20:52:01.429 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-01 20:52:01.430 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "DeltaGenerator()"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "st.title (\"Prediction Model for Hyperuricemia in Pediatric Hypertension\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "d962c167-08c3-4ce7-90be-9125caa84159",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-12-01 21:52:39.114 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-01 21:52:39.114 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-01 21:52:39.115 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-01 21:52:39.116 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-01 21:52:39.116 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-01 21:52:39.118 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-01 21:52:39.118 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "Age = st.number_input('Age (6-17 years):', min_value=6, max_value=18, value = 12)\n",
    "age = (Age-12.51221)/2.235874"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "ac9d6bfb-df29-4088-85dd-f53300a71ce0",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-12-01 21:52:46.005 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-01 21:52:46.005 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-01 21:52:46.006 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-01 21:52:46.007 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-01 21:52:46.008 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-01 21:52:46.008 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-01 21:52:46.009 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "sex = st.selectbox(\"Sex:\", options =[0, 1], format_func = lambda x:\"Boys\" if x==0 else \"Girls\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "5e7d7acc-51c9-44fa-92dc-7a53a891e943",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-12-01 21:52:46.256 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-01 21:52:46.257 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-01 21:52:46.258 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-01 21:52:46.258 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-01 21:52:46.259 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-01 21:52:46.260 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-01 21:52:46.261 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "Prealbumin = st.number_input('Prealbumin (mg/L):', min_value=1, max_value=500, value = 41)\n",
    "prealbumin = (Prealbumin-235.2251)/43.65179"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "bce468a6-3083-43d6-a8f2-58e922a84c44",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-12-01 21:52:46.400 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-01 21:52:46.402 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-01 21:52:46.403 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-01 21:52:46.403 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-01 21:52:46.404 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-01 21:52:46.405 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-01 21:52:46.405 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "Crea = st.number_input(\"Crea (μmol/L):\", min_value = 1, max_value = 120, value = 41)\n",
    "crea = (Crea-53.54569)/12.28562"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "6b2cbb2b-bbe8-40b7-9b41-8594411af94f",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-12-01 21:52:46.568 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-01 21:52:46.569 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-01 21:52:46.569 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-01 21:52:46.570 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-01 21:52:46.571 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-01 21:52:46.571 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-01 21:52:46.572 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "BMI = st.number_input(\"BMI (kg/m^2):\", min_value = 1, max_value = 100, value = 10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "22b3223b-2414-4166-a261-5ca53326b515",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 定义LMS数据\n",
    "boys_data = pd.DataFrame({\n",
    "    'age': [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, \n",
    "            10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0],\n",
    "    'L': [-0.0293, -0.1935, -0.5351, -0.6372, -0.8185, -0.9278, -1.0161, -1.0837, -1.1325, -1.1647, -1.1822, -1.1871, \n",
    "          -1.1819, -1.1691, -1.1509, -1.1293, -1.1058, -1.0816, -1.0576, -1.0344, -1.0126, -0.9924, -0.9741, -0.9577, \n",
    "          -0.9430, -0.9301, -0.9187, -0.9085, -0.8993, -0.8908, -0.8830, -0.8756, -0.8685, -0.8617, -0.8552, -0.8487, -0.8425],\n",
    "    'M': [13.0666, 17.9640, 17.1907, 16.4742, 16.3265, 15.9658, 15.6597, 15.4508, 15.3189, 15.2341, 15.2236, 15.2721, \n",
    "          15.3458, 15.4512, 15.5950, 15.7688, 15.9650, 16.1830, 16.4229, 16.6819, 16.9567, 17.2429, 17.5353, 17.8273, \n",
    "          18.1137, 18.3942, 18.6687, 18.9345, 19.1891, 19.4318, 19.6627, 19.8823, 20.0914, 20.2909, 20.4821, 20.6656, 20.8424],\n",
    "    'S': [0.0837, 0.0875, 0.0814, 0.0787, 0.0761, 0.0748, 0.0745, 0.0752, 0.0767, 0.0790, 0.0823, 0.0866, 0.0916, 0.0971, \n",
    "          0.1027, 0.1084, 0.1140, 0.1192, 0.1240, 0.1283, 0.1320, 0.1352, 0.1379, 0.1401, 0.1418, 0.1432, 0.1443, 0.1451, \n",
    "          0.1456, 0.1460, 0.1463, 0.1464, 0.1465, 0.1465, 0.1465, 0.1464, 0.1464]\n",
    "})\n",
    "\n",
    "girls_data = pd.DataFrame({\n",
    "    'age': [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, \n",
    "            10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0],\n",
    "    'L': [-0.4355, -0.4536, -0.5721, -0.6589, -0.8350, -0.8991, -0.9480, -0.9848, -1.0121, -1.0321, -1.0463, -1.0559, \n",
    "          -1.0621, -1.0657, -1.0673, -1.0674, -1.0665, -1.0649, -1.0629, -1.0606, -1.0582, -1.0559, -1.0537, -1.0517, \n",
    "          -1.0501, -1.0488, -1.0478, -1.0470, -1.0465, -1.0462, -1.0460, -1.0458, -1.0457, -1.0456, -1.0455, -1.0454, -1.0453],\n",
    "    'M': [13.0027, 17.4082, 16.7427, 16.0306, 15.9459, 15.6371, 15.4167, 15.2686, 15.1544, 15.0632, 14.9936, 14.9624, \n",
    "          14.9602, 14.9712, 15.0154, 15.0963, 15.2142, 15.3730, 15.5748, 15.8167, 16.0946, 16.4034, 16.7377, 17.0885, \n",
    "          17.4471, 17.8038, 18.1500, 18.4784, 18.7831, 19.0605, 19.3083, 19.5273, 19.7197, 19.8898, 20.0427, 20.1834, 20.3165],\n",
    "    'S': [0.0878, 0.0835, 0.0790, 0.0775, 0.0770, 0.0771, 0.0773, 0.0782, 0.0801, 0.0826, 0.0854, 0.0884, 0.0915, 0.0949, \n",
    "          0.0984, 0.1022, 0.1062, 0.1103, 0.1142, 0.1180, 0.1215, 0.1245, 0.1270, 0.1290, 0.1305, 0.1317, 0.1325, 0.1331, \n",
    "          0.1334, 0.1336, 0.1337, 0.1337, 0.1337, 0.1336, 0.1335, 0.1334, 0.1333]\n",
    "})\n",
    "\n",
    "# 创建LMS数据字典\n",
    "lms_data = {'boys': boys_data, 'girls': girls_data}\n",
    "\n",
    "def calculate_bmi_zscore(age, bmi, sex, lms_data):\n",
    "    \"\"\"\n",
    "    计算BMI Z-score的函数\n",
    "    \n",
    "    参数:\n",
    "    age: 年龄 (年)\n",
    "    bmi: BMI值\n",
    "    sex: 性别 (1=男童, 2=女童)\n",
    "    lms_data: 包含男女童LMS数据的字典\n",
    "    \n",
    "    返回:\n",
    "    z_score: BMI Z-score\n",
    "    \"\"\"\n",
    "    # 根据性别选择对应的LMS数据\n",
    "    if sex == 0:  # 男童\n",
    "        data = lms_data['boys']\n",
    "    elif sex == 1:  # 女童\n",
    "        data = lms_data['girls']\n",
    "    else:\n",
    "        # 如果不是1或2，返回NaN\n",
    "        return np.nan\n",
    "    \n",
    "    # 确保年龄是数值类型\n",
    "    try:\n",
    "        age = float(age)\n",
    "    except ValueError:\n",
    "        return np.nan\n",
    "    \n",
    "    # 找到对应年龄的索引\n",
    "    # 由于年龄可能是小数，使用最接近的匹配\n",
    "    age_idx = None\n",
    "    min_diff = float('inf')\n",
    "    \n",
    "    for i, row_age in enumerate(data['age']):\n",
    "        diff = abs(row_age - age)\n",
    "        if diff < min_diff:\n",
    "            min_diff = diff\n",
    "            age_idx = i\n",
    "    \n",
    "    # 如果年龄超出范围，返回NaN\n",
    "    if age_idx is None:\n",
    "        return np.nan\n",
    "    \n",
    "    # 获取对应的L, M, S值\n",
    "    L = data.loc[age_idx, 'L']\n",
    "    M = data.loc[age_idx, 'M']\n",
    "    S = data.loc[age_idx, 'S']\n",
    "    \n",
    "    # 计算Z-score (根据LMS方法)\n",
    "    try:\n",
    "        # 处理L接近0的情况\n",
    "        if abs(L) < 1e-10:\n",
    "            # 当L接近0时，使用极限形式\n",
    "            z_score = (np.log(bmi / M)) / S\n",
    "        else:\n",
    "            z_score = ((bmi / M) ** L - 1) / (L * S)\n",
    "    except:\n",
    "        # 计算错误时返回NaN\n",
    "        z_score = np.nan\n",
    "    \n",
    "    return z_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "8951a8d1-a44f-4c4e-94a4-20e62b0bf815",
   "metadata": {},
   "outputs": [],
   "source": [
    "bmi_z = calculate_bmi_zscore(Age, BMI, sex, lms_data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "31c22d17-39ae-4f00-bfcf-9745d66a1269",
   "metadata": {},
   "outputs": [],
   "source": [
    "feature_values = [bmi_z, sex, prealbumin, crea, age]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "c022ad5a-90b8-4b6f-904e-684aff2209ac",
   "metadata": {},
   "outputs": [],
   "source": [
    "features = np.array([feature_values])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "2b8e1f89-714b-42f0-93e5-4e49e19296f1",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-12-01 21:48:19.653 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-01 21:48:19.655 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-01 21:48:19.655 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-01 21:48:19.656 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-01 21:48:19.656 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-01 21:48:19.658 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "if st.button('Predict'):\n",
    "    predict_class = svm_model.predict(features)[1]\n",
    "    predict_proba = svm_model.predict_proba (features)[1]\n",
    "    st.write(f\"**Predicted Class:** {predict_class} (1:Hyperuricemia, 0:No hyperuricemia)\")\n",
    "    st.write(f\"**Predicted Probabilities:** {predict_proba}\"  )\n",
    "    st.subheader (\"SHAP Waterfall Plot Explanation\")\n",
    "    explainer = shap.KernelExplainer(svm_model)\n",
    "    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))\n",
    "    if predicted_class == 1:\n",
    "        shap.waterfall_plot(explainer.shap_values[1],shape_values[:,:,1],pd.DataFrame([feature_values], columns=feature_names),matplotlib=True)\n",
    "    else:\n",
    "        shap.waterfall_plot(explainer.shap_values[0],shape_values[:,:,0],pd.DataFrame([feature_values], columns=feature_names),matplotlib=True)\n",
    "\n",
    "    plt.savefig(\"shap_waterfall_plot.png\", bbox_inches = \"tight\", dpi=300)\n",
    "    st.image(\"shap_waterfall_plot.png\", caption = 'SHAP Waterfall Plot Explanation')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e3d2c911-a909-492f-b38a-9a60b436a889",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
