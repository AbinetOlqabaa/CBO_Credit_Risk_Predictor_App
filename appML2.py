import streamlit as st
from PIL import Image
import time
import pickle
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestClassifier
####


image = Image.open("coop.jpg")
st.image(image, caption='Jijjiirama Guddaaf Kan Onnate!', use_column_width=True)
st.title("CREDIT RISK PREDICTION ")
html_temp = """
	<div style="background-color:tomato;padding:10px">
	<h2 style="color:white;text-align:center;">AN ONLINE SOFTWARE (DEMO)</h2>
	</div>
	"""
st.markdown(html_temp, unsafe_allow_html=True)

df = pd.read_csv("C:\\Users\\User\\Desktop\\ANNchurn\\Credit-Risk-Analyzer-master\\credit_data.csv")

pickle_in = open("Abinet.pkl", "rb")
classifier = pickle.load(pickle_in)

st.set_option('deprecation.showPyplotGlobalUse', False)

from collections import Counter
import warnings
warnings.filterwarnings('ignore')

html_temp2 = """
<div style="background-color:tomato;padding:10px">
<h2 style="color:white;text-align:center;">READ THE INSTRUCTIONS AND FILL YOUR VALUES OF FEATURES</h2>
</div>
"""
st.sidebar.markdown(html_temp2,unsafe_allow_html=True)

def get_user_Select():
    age = st.sidebar.number_input('Age', 0.0, 100.0, 18.0)

    gender=st.sidebar.radio('What is your Gender?', ["Male","Female"])
    if gender == 'Female':
        gender = 0
    else:
        gender = 1
    
    education = st.sidebar.radio('Your education level', ['Under Graduate','Graduate', 'Post Graduate', 'Other'])
    if education == 'Graduate':
        education = 0
    elif education == 'Post Graduate':
        education = 1
    elif education == 'Other':
        education = 2
    else:
        education = 3

    occupation = st.sidebar.radio('Your occupation', [ 'Salaried','Business', 'Professional', 'Student'])
    if occupation == 'Salaried':
        occupation = 0
    elif occupation == 'Business':
        occupation = 1
    elif occupation == 'Professional':
        occupation = 2
    else:
        occupation = 3
    organization_type = st.sidebar.radio('Type of your organization', ['Tier 1','Tier 2', 'Tier 3', 'None'])
    if organization_type == 'Tier 1':
        organization_type = 0
    elif organization_type == 'Tier 2':
        organization_type = 1
    elif organization_type == 'Tier 3':
        organization_type = 2
    else:
        organization_type = 3
    seniority = st.sidebar.radio('Your seniority', ['Entry','Junior','Mid-level 1','Mid-level 2','Senior','None'])
    if seniority == 'Mid-level 2':
        seniority = 0
    elif seniority == 'Senior':
        seniority = 1
    elif seniority == 'Mid-level 1':
        seniority = 2
    elif seniority == 'Entry':
        seniority = 3
    elif seniority == 'None':
        seniority = 4
    else:
        seniority = 5

    annual_income = st.sidebar.number_input('Your annual income', 0.0, 500000.0)

    disposable_income = st.sidebar.number_input('Your disposable income', 0.0, 500000.0)

    house_type = st.sidebar.radio('Type of your House', ['Family', 'Rented', 'Company provided', 'Owned'])
    if house_type == 'Owned':
        house_type = 0
    elif house_type == 'Company provided':
        house_type = 1
    elif house_type == 'Rented':
        house_type = 2
    else:
        house_type = 3
    
    vehicle_type = st.sidebar.radio('Type of your vehicle', ['Two Wheeler', 'Four Wheeler','None'])
    if vehicle_type == 'Four Wheeler':
        vehicle_type = 0
    elif vehicle_type == 'Two Wheeler':
        vehicle_type = 1
    else:
        vehicle_type = 2

    marital_status = st.sidebar.radio('Marital status', ['Married', 'Single', 'Other'])
    if marital_status == 'Married':
        marital_status = 0
    elif marital_status == 'Other':
        marital_status = 1
    else:
        marital_status = 2
    
    no_card = st.sidebar.number_input('Number of accounts in different banks', 0.0, 100.0)
    #default= st.sidebar.number_input('Default Bias', 0.0, 10.0)


    user_data = {'age': age,
                 'gender': gender,
                 'education': education,
                 'occupation': occupation,
                 'organization_type': organization_type,
                 'seniority': seniority,
                 'annual_income': annual_income,
                 'disposable_income': disposable_income,
                 'house_type': house_type,
                 'vehicle_type': vehicle_type,
                 'marital_status': marital_status,
                 'no_card': no_card
                 }
    features = pd.DataFrame(user_data, index= [0])
    return features

user_Select = get_user_Select()

prediction = classifier.predict(user_Select)

#####

with st.spinner('Processing....'):
    time.sleep(5)
st.success('Process completed')

#@st.cache(allow_output_mutation = True)

def main():
    result = ""
    if st.button("predict"):
        result = round(prediction[0],2)
        st.success("The result of a classifier is {}".format(result))
        if result >= 0.5:
            st.warning("Customer will default")
        else:
            st.success("Customer will pay the loan")
            st.info("You can continue to the loan application process.")
            st.write("SignUp on  https://www.example.com")
            st.write("Login on  https://www.example.com")

if __name__ == '__main__':
    main()

if st.button("About CBO"):
    st.info("Cooperative Bank of Oromia is....")

html_temp3 = """
<div style="background-color:tomato;padding:10px">
<h2 style="color:white;text-align:center;">Additional features</h2>
</div>
"""
st.markdown(html_temp3,unsafe_allow_html=True)
st.subheader("Informations")
st.info("A.	General to specific credit product Introduction such as customer requirement,how to get the service, service advantages, service charges, policy and procedures will be presented for customers.")
st.info("B.	Available credit products and detail descriptions explaining convenient business")
st.info("C.	Eligibility criteria for each credit product")
st.info("D.	Documents required for each credit product")
st.info("E.	Processing time for each credit request")
st.info("F.	Processing fee for each credit request")
st.info("G.	Location of alternative centres/ branches /outlets/ entertaining the request")
st.info("H.	Support for customer request related with credit Service")
st.info("I.	Support for any possible enquiry about new/ordinary credit products")
 
if st.button("Credit Request Processing"):
	image = Image.open("LOGO_OR.png")
	st.image(image, caption='Baankii Uummataa!', use_column_width=True)
	st.info("Development to entertain new or existing credit customer loan requests.")
	st.info("A.	Preliminary screening and assessment")
	st.info("B.	Credit application")
	st.info("C.	Documentation delivery")
	st.info("D.	Credit process status notification")
	st.info("E.	Credit decision communication")
	st.info("F.	Customer enquiry and appeal or comment")

if st.button("Internal credit processing"):
	image = Image.open("LOGO_OR.png")
	st.image(image, caption='Baankii Uummataa!', use_column_width=True)
	st.info("A.Document exchange among credit performers automation ")
	st.info("B.	Loan delivery time(LDT)to each credit performer automation")
	st.info("C.	Credit analysis and project appraisal process automation")
	st.info("D.	Credit approval/approving team deliberation automation")
	st.info("E.	Documentation Retrieval automation ")
	st.info("F.	 Reference system for credit appraisal and approval.")
	st.info("G.	Recommend credit worthiness of loan applicant")
if st.button("Credit Customer Services"):
	image = Image.open("LOGO_OR.png")
	st.image(image, caption='Baankii Uummataa!', use_column_width=True)
	st.info("A.Loan Repayment or collection through alternative channels")
	st.info("B.	Loan statement delivery")
	st.info("C.	Credit related notifications and advises")
image = Image.open("AIC_Logo_New.png")
st.image(image, caption='AI For National Growth!', use_column_width=True)
