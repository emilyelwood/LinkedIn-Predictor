import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

s = pd.read_csv("social_media_usage.csv")

def clean_sm(x):
    x = np.where(x==1,1,0)
    return x

ss = pd.DataFrame({
    "linkedin": s["web1h"].apply(clean_sm),
    "income":np.where(s["income"] > 9, np.nan, s["income"]),
    "education": np.where(s["educ2"]>8,np.nan, s["educ2"]),
    "parent": np.where(s["par"] == 1, 1, 0),
    "married": np.where(s["marital"] == 1, 1, 0),
    "female": np.where(s["gender"] == 2, 1, 0),
    "age": np.where(s["age"] >97,np.nan, s["age"])})

ss = ss.dropna()

y = ss["linkedin"]
X = ss[["income", "education", "parent", "married", "female", "age"]]

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    stratify=y,       
                                                    test_size=0.2,   
                                                    random_state=5485) 

lr1 = LogisticRegression(class_weight='balanced')
lr1.fit(X_train, y_train)



import streamlit as st
st.title('LinkedIn User Prediction App')
st.caption('Configure demographics to predict if someone is likely to use LinkedIn:')
st.divider()


inc_options = ["Less than $10,000", "10 to under $20,000", "20 to under $30,000",
               "30 to under $40,000", "40 to under $50,000", "50 to under $75,000",
               "75 to under $100,000", "100 to under $150,000", "more than $150,000"]

income_mapping = {opt: idx + 1 for idx, opt in enumerate(inc_options)}

inc = st.selectbox("Select Income Level", options=inc_options)

income = income_mapping.get(inc, 0)
    

educ_options = ["Less than high school", "High school incomplete", "High school graduate",
                "Some college, no degree", "Two-year associate degree from a college or university",
                "Four-year college or university degree/Bachelor’s degree",
                "Some postgraduate or professional schooling, no postgraduate degree",
                "Postgraduate or professional degree, including master’s, doctorate, medical or law degree"]

education_mapping = {opt: idx + 1 for idx, opt in enumerate(educ_options)}

educ = st.selectbox("Select Education Level", options=educ_options)

education = education_mapping.get(educ, 0)


par = st.radio(
    "Select Parental Status",
    ["Parent", "Not a Parent"])
    
if par == "Parent":
    parent = 1
else:
    parent = 0
    
mar = st.radio(
    "Select Marital Status",
    ["Married", "Not Married"])
    
if mar == "Married":
    married = 1
else:
    married = 0
    
fem = st.radio(
    "Select Gender",
    ["Female", "Male"])
    
if fem == "Female":
    female = 1
else:
    female = 0    
    
age = st.slider(label="Select Age", 
          min_value=1,
          max_value=97,
          value=30)

    
input_data = pd.DataFrame([[income, education, parent, married, female, age]],
                           columns=["income", "education", "parent", "married", "female", "age"])

    
pred_result = lr1.predict(input_data)
pred_prob = lr1.predict_proba(input_data)[:,1]
pred_perc = pred_prob*100
    
if pred_result == 1:
    pred_label = "be a LinkedIn user"
else:
    pred_label = "not be a LinkedIn user"    
    
st.write(f"This person is predicted to {pred_label} with {pred_perc[0]}% probability")   

