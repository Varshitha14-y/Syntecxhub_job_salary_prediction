import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.title(" Salary Estimation System")
df = pd.read_csv("naukri_com-job_sample.csv")
st.subheader("Raw Dataset")
st.dataframe(df.head())
st.subheader("Data Cleaning")
drop_cols = ["jobdescription", "postdate", "company", "joblocation_address", "education"]
df = df.drop(columns=drop_cols, errors='ignore')
df["experience"] = df["experience"].astype(str).str.extract('(\d+)')
df["payrate"] = df["payrate"].astype(str).str.replace(",", "").str.extract('(\d+)')
df["experience"] = pd.to_numeric(df["experience"], errors='coerce')
df["payrate"] = pd.to_numeric(df["payrate"], errors='coerce')
df = df.dropna(subset=["experience", "payrate"])
df = df[np.isfinite(df["experience"])]
df = df[np.isfinite(df["payrate"])]
df["skills_count"] = df["skills"].apply(lambda x: len(str(x).split(",")))
df = df.drop("skills", axis=1)
st.write("Cleaned Data:")
st.dataframe(df.head())
st.subheader("Encoding Categorical Data")
label_encoders = {}

for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

st.dataframe(df.head())
st.subheader("Data Visualization")

fig1 = plt.figure()
plt.scatter(df["experience"], df["payrate"])
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.title("Experience vs Salary")
st.pyplot(fig1)

fig2 = plt.figure()
plt.hist(df["payrate"])
plt.title("Salary Distribution")
st.pyplot(fig2)
st.subheader("Correlation Matrix")

numeric_df = df.select_dtypes(include=['number'])
st.dataframe(numeric_df.corr())
st.subheader("Model Training")

numeric_df = df.select_dtypes(include=['number'])
numeric_df = numeric_df.dropna()

X = numeric_df[["experience", "industry", "jobtitle", "skills_count"]]
y = numeric_df["payrate"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

st.success("Model Trained Successfully!")
st.subheader("Model Evaluation")

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

st.write("MAE:", mae)
st.write("MSE:", mse)
st.subheader("Predict Salary")

experience = st.slider("Experience (years)", 0.0, 20.0, 2.0)
skills_count = st.slider("Number of Skills", 1, 20, 5)

industry_list = label_encoders["industry"].classes_
jobtitle_list = label_encoders["jobtitle"].classes_

industry = st.selectbox("Industry", industry_list)
jobtitle = st.selectbox("Job Title", jobtitle_list)

industry_enc = label_encoders["industry"].transform([industry])[0]
jobtitle_enc = label_encoders["jobtitle"].transform([jobtitle])[0]

if st.button("Predict Salary"):
    input_data = pd.DataFrame(
        [[experience, industry_enc, jobtitle_enc, skills_count]],
        columns=X.columns
    )

    prediction = model.predict(input_data)

    st.success(f"Predicted Salary: ₹{prediction[0]:,.2f}")