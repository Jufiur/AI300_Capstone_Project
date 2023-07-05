#app.py
from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Load the model
        model = joblib.load('model/catboost_model.pkl')

        has_internet_service = int(request.form['has_internet_service'])
        internet_type = int(request.form['internet_type'])
        has_unlimited_data = int(request.form['has_unlimited_data'])
        paperless_billing = int(request.form['paperless_billing'])
        stream_tv = int(request.form['stream_tv'])
        stream_movie = int(request.form['stream_movie'])
        senior_citizen = int(request.form['senior_citizen'])
        tenure_months = int(request.form['tenure_months'])
        num_referrals = int(request.form['num_referrals'])
        has_premium_tech_support = int(request.form['has_premium_tech_support'])
        has_online_security = int(request.form['has_online_security'])
        has_online_backup = int(request.form['has_online_backup'])
        has_device_protection = int(request.form['has_device_protection'])
        contract_type = int(request.form['contract_type'])
        payment_method = int(request.form['payment_method'])
        married = int(request.form['married'])
        num_dependents = int(request.form['num_dependents'])
     
        # Get the parameters for the prediction
        parameters = [has_internet_service, internet_type, has_unlimited_data, paperless_billing, stream_tv, stream_movie, senior_citizen, tenure_months, num_referrals, has_premium_tech_support, has_online_security, has_online_backup, has_device_protection, contract_type, payment_method, married, num_dependents]
        
        # Make the prediction
        prediction = model.predict(parameters)

        # Return the result along with the form
        return render_template('home.html', prediction=prediction)

    # If it's a GET request, just render the form
    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)
