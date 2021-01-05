import streamlit as st
from PIL import Image
import time
import pickle
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestClassifier
####
from sklearn.preprocessing import normalize
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix
####
from sklearn.model_selection import cross_val_score


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

pickle_in = open("Abinet_DL.pkl", "rb")
classifier = pickle.load(pickle_in)

gender_lebel = df.groupby(['gender'])['default'].mean().sort_values().index
enumerate(gender_lebel, 0)
gender_lebel2 = {k: i for i, k in enumerate(gender_lebel, 0)}

education_lebel = df.groupby(['education'])['default'].mean().sort_values().index
enumerate(education_lebel, 0)
education_lebel2 = {k: i for i, k in enumerate(education_lebel, 0)}

occupation_lebel = df.groupby(['occupation'])['default'].mean().sort_values().index
enumerate(occupation_lebel, 0)
occupation_lebel2 = {k: i for i, k in enumerate(occupation_lebel, 0)}

organization_type_lebel = df.groupby(['organization_type'])['default'].mean().sort_values().index
enumerate(organization_type_lebel, 0)
organization_type_lebel2 = {k: i for i, k in enumerate(organization_type_lebel, 0)}

seniority_lebel = df.groupby(['seniority'])['default'].mean().sort_values().index
enumerate(seniority_lebel, 0)
seniority_lebel2 = {k: i for i, k in enumerate(seniority_lebel, 0)}

house_type_lebel = df.groupby(['house_type'])['default'].mean().sort_values().index
enumerate(house_type_lebel, 0)
house_type_lebel2 = {k: i for i, k in enumerate(house_type_lebel, 0)}

vehicle_type_lebel = df.groupby(['vehicle_type'])['default'].mean().sort_values().index
enumerate(vehicle_type_lebel, 0)
vehicle_type_lebel2 = {k: i for i, k in enumerate(vehicle_type_lebel, 0)}

marital_status_lebel = df.groupby(['marital_status'])['default'].mean().sort_values().index
enumerate(marital_status_lebel, 0)
marital_status_lebel2 = {k: i for i, k in enumerate(marital_status_lebel, 0)}

df['gender'] = df['gender'].map(gender_lebel2)
df['education'] = df['education'].map(education_lebel2)
df['occupation'] = df['occupation'].map(occupation_lebel2)
df['organization_type'] = df['organization_type'].map(organization_type_lebel2)
df['seniority'] = df['seniority'].map(seniority_lebel2)
df['house_type'] = df['house_type'].map(house_type_lebel2)
df['vehicle_type'] = df['vehicle_type'].map(vehicle_type_lebel2)
df['marital_status'] = df['marital_status'].map(marital_status_lebel2)
#st.write(df)

st.set_option('deprecation.showPyplotGlobalUse', False)

# Best 10 features
x = df.drop('default', axis='columns')
y = df['default']
#apply SelectKBest class to extract top 10 best features
st.subheader("Top 8 best features using normalizing method")
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(normalize(x, axis=0, norm='l2'),y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(x.columns)
#concat two dataframes for better visualization
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
st.write(featureScores.nlargest(8,'Score'))  #print 10 best features

from collections import Counter
import warnings
warnings.filterwarnings('ignore')

x = df.iloc[:, 0:12].values
y = df.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=5)

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
    
    education = st.sidebar.radio('Your education level', ['Graduate','Under Graduate', 'Post Graduate', 'Other'])
    if education == 'Graduate':
        education = 0
    elif education == 'Post Graduate':
        education = 1
    elif education == 'Other':
        education = 2
    else:
        education = 3

    occupation = st.sidebar.radio('Your occupation', ['Business', 'Salaried', 'Professional', 'Student'])
    if occupation == 'Salaried':
        occupation = 0
    elif occupation == 'Business':
        occupation = 3
    elif occupation == 'Professional':
        occupation = 1
    else:
        occupation = 2
    organization_type = st.sidebar.radio('Type of your organization', ['Tier 1','Tier 2', 'Tier 3', 'None'])
    if organization_type == 'Tier 1':
        organization_type = 0
    elif organization_type == 'Tier 2':
        organization_type = 3
    elif organization_type == 'Tier 3':
        organization_type = 1
    else:
        organization_type = 2
    seniority = st.sidebar.radio('Your seniority', ['Entry','Junior','Mid-level 1','Mid-level 2','Senior','None'])
    if seniority == 'Mid-level 2':
        seniority = 1
    elif seniority == 'Senior':
        seniority = 2
    elif seniority == 'Mid-level 1':
        seniority = 3
    elif seniority == 'Entry':
        seniority = 4
    elif seniority == 'None':
        seniority = 5
    else:
        organization_type = 4

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
    
    vehicle_type = st.sidebar.radio('Type of your vehicle', ['None' ,'Two Wheeler', 'Four Wheeler'])
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
    
    no_card = st.sidebar.number_input('Number of accounts in banks', 0.0, 100.0)

    #default = st.sidebar.number_input('Your annual income', 0, 1)

    st.sidebar.info("""About Loan Products:
                    A term loan is simply a loan provided for business purposes that needs to be paid back within a specified time frame.
                    It typically carries a fixed interest rate, monthly or quarterly repayment schedule - and includes a set maturity date.
                    loans can be both secure (i.e. Some collateral is provided) and unsecured.
                    A secured term loan will usually have a lower interest rate than an unsecured one.
                    Depending upon the repayment period this loan type is classified as under:          
                    -	Short term loan: Repayment period less than 1 year.
                    -	Medium term loan: Repayment period between 1 to 3 years.
                    -	Long term loan: Repayment period above 3 years.
                    """)

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

Random_Forest_Classifier = RandomForestClassifier()
Random_Forest_Classifier.fit(x_train, y_train)

st.write("Model test accuracy Score: ")
st.write(str(accuracy_score(y_test, classifier.predict(x_test)) * 100) + '%')

prediction = Random_Forest_Classifier.predict(user_Select)

#####

with st.spinner('Precessing....'):
    time.sleep(5)
st.success('Process completed')

#@st.cache(allow_output_mutation = True)

def main():
    result = ""
    if st.button("predict"):
        result = prediction
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
