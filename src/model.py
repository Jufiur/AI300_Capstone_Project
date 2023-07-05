# model.py
import pymysql
import pandas as pd
import joblib
from catboost import CatBoostClassifier

# Load the dataset

DB_HOST='ai300-capstone-db.c2ced10ceyki.ap-southeast-1.rds.amazonaws.com'
PORT=3306
DB_NAME='capstone'
DB_USER='student'
DB_PWD='ZXLNo#sJI0K*uT3h&4spyDebP'
CURSORCLASS=pymysql.cursors.DictCursor

connection = pymysql.connect(host=DB_HOST, port=PORT, user=DB_USER, passwd=DB_PWD, db=DB_NAME, cursorclass=CURSORCLASS)

sql_query = f'SELECT * FROM capstone.account a JOIN capstone.account_usage au ON a.account_id = au.account_id JOIN capstone.churn_status cs ON a.customer_id = cs.customer_id JOIN capstone.customer c ON a.customer_id = c.customer_id JOIN capstone.city ct ON c.zip_code = ct.zip_code;'

def get_records(sql_query):
    try:
        with connection.cursor() as cursor:
            cursor.execute(sql_query)

        # Connection is not autocommit by default, so we must commit to save changes
        connection.commit()
        
        # Fetch all the records from SQL query output
        results = cursor.fetchall()
        
        # Convert results into pandas dataframe
        df = pd.DataFrame(results)
        
        print(f'Successfully retrieved records')
        
        return df
        
    except Exception as e:
        print(f'Error encountered: {e}')

df = get_records(sql_query)
df

# 2.1 Clean Target Columns

#This code is to drop rows based on the condition df["customer_status"] == "Churned" & df["churn_label"] != "Yes" (NaN values)
# To delete nan rows based on the churn_label value "nan" or "None"
df.drop(df[(df["customer_status"] == "Churned") & (df["churn_label"] != "Yes")].index, inplace=True)

# This function is to change the value of column from "Yes" and "No" to 1 and 0. 
def change_yes_no(column):
    if column == 'Yes':
        return 1
    else:
        return 0

# Next we apply the function
df["has_internet_service"] = df["has_internet_service"].apply(change_yes_no)
df["has_unlimited_data"] = df["has_unlimited_data"].apply(change_yes_no)
df["has_premium_tech_support"] = df["has_premium_tech_support"].apply(change_yes_no)
df["has_online_security"] = df["has_online_security"].apply(change_yes_no)
df["has_online_backup"] = df["has_online_backup"].apply(change_yes_no)
df["has_device_protection"] = df["has_device_protection"].apply(change_yes_no)
df["paperless_billing"] = df["paperless_billing"].apply(change_yes_no)
df["stream_tv"] = df["stream_tv"].apply(change_yes_no)
df["stream_movie"] = df["stream_movie"].apply(change_yes_no)
df["churn_label"] = df["churn_label"].apply(change_yes_no)
df["senior_citizen"] = df["senior_citizen"].apply(change_yes_no)
df["married"] = df["married"].apply(change_yes_no)

# This function is to change tenure_months data.
def change_tenure_months(tenure_months):
    if tenure_months < 12:
        return 0
    elif tenure_months < 24:
        return 1
    elif tenure_months < 36:
        return 2
    elif tenure_months < 48:
        return 3
    elif tenure_months < 60:
        return 4
    else:
        return 5
    
# Next we apply the function
df["tenure_months"] = df["tenure_months"].apply(change_tenure_months)

# This function is to clean internet_type data.
def change_internet_type(internet_type):
    if internet_type == "Cable":
        return 1
    elif internet_type == "DSL":
        return 2
    elif internet_type == "Fiber Optic":
        return 3
    else:
        return 0
    
# Next we apply the function
df["internet_type"] = df["internet_type"].apply(change_internet_type)

# This function is to clean contract_type data.
def change_contract_type(contract_type):
    if contract_type == "Month-to-Month":
        return 0
    elif contract_type == "One Year":
        return 1
    else:
        return 2
    
# Next we apply the function
df["contract_type"] = df["contract_type"].apply(change_contract_type)

# This function is to clean payment_method data.
def change_payment_method(payment_method):
    if payment_method == "Bank Withdrawal":
        return 1
    elif payment_method == "Credit Card":
        return 2
    else:
        return 0
    
# Next we apply the function
df["payment_method"] = df["payment_method"].apply(change_payment_method)

# 2.2 Replace NaNs with "Unknown" in columns churn_category and churn_reason. 
def change_na_in_column(column):
    if pd.isna(column):
        return "Unknown"
    else:
        return column
    
# Next we apply the function
df["churn_category"] = df["churn_category"].apply(change_na_in_column)
df["churn_reason"] = df["churn_reason"].apply(change_na_in_column)

# 2.3 Feature Engineering
features = ['has_internet_service', 'internet_type', 'has_unlimited_data', 'paperless_billing', 'stream_tv',
            'stream_movie', 'senior_citizen', 'tenure_months', 'num_referrals', 'has_premium_tech_support',
            'has_online_security', 'has_online_backup', 'has_device_protection', 'contract_type', 
            'payment_method', 'married', 'num_dependents']

X = df[features]
y = df["churn_label"]


# # 3. Train Model
model = CatBoostClassifier(learning_rate=0.1, max_depth=7, random_seed=17, subsample=0.5, scale_pos_weight=30, l2_leaf_reg=12, random_strength=10)
model.fit(X, y, verbose=False)

# Save the model
joblib.dump(model, 'model/catboost_model.pkl')
