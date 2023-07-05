# AI300 Capstone Project
 
1. Team Number & Names of Team Members:
   - Team Number: team03
   - Team Member Names: Jufiur Lim (Leader) and Benjamin Ng 
    
2. Website URL of deployed Flask web application:
   - Public IPv4 address: http://52.77.220.238/
   - Public IPv4 DNS: http://ec2-52-77-220-238.ap-southeast-1.compute.amazonaws.com/

4. Details on chosen final model and model parameters:
   Final Model:
   - CatBoost Model - Train Test Split
      features = ['has_internet_service', 'internet_type', 'has_unlimited_data', 'paperless_billing', 'stream_tv', 'stream_movie', 'senior_citizen', 'tenure_months', 'num_referrals',
                 'has_premium_tech_support', 'has_online_security', 'has_online_backup', 'has_device_protection', 'contract_type', 'payment_method', 'married', 'num_dependents']

      X = df[features]
      y = df['churn_label']
      
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
      
      catb_model = CatBoostClassifier(learning_rate=0.1, max_depth=7, random_seed=17, subsample=0.5, scale_pos_weight=30, l2_leaf_reg=12, random_strength=10)
      catb_model.fit(X_train, y_train, verbose=False)
      y_predict = catb_model.predict(X_test)
      y_predict_proba = catb_model.predict_proba(X_test)[:,1]
      
      print('AUC:', metrics.roc_auc_score(y_test, y_predict_proba))
      print('Accuracy:', metrics.accuracy_score(y_test, y_predict))

  Model Parameters: 
  - catb_model = CatBoostClassifier(learning_rate=0.1, max_depth=7, random_seed=17, subsample=0.5, scale_pos_weight=30, l2_leaf_reg=12, random_strength=10)

  4. Offline AUC metric of chosen final model:
     AUC: 0.902867443221879
