# Insurance_Price_Predictor_MLModel
 ![img1](https://github.com/user-attachments/assets/7f0463c8-ae1e-43fa-9722-93f528593035)
## 🤔**Problem Statement**
Insurance companies need to accurately predict the cost of health insurance for individuals to set premiums appropriately. However, traditional methods of cost prediction often rely on broad actuarial tables and historical averages, which may not account for the nuanced differences among individuals. By leveraging machine learning techniques, insurers can predict more accurately the insurance costs tailored to individual profiles, leading to more competitive pricing and better risk management.

## 🎯 **Insurance Cost Prediction need**
The primary need for this project arises from the challenges insurers face in pricing policies accurately while remaining competitive in the market. Inaccurate predictions can lead to losses for insurers and unfairly high premiums for policyholders. By implementing a machine learning model, insurers can:
-	Enhance Precision in Pricing: Use individual data points to determine premiums that reflect actual risk more closely than generic estimates.
-	Increase Competitiveness: Offer rates that are attractive to consumers while ensuring that the pricing is sustainable for the insurer.
-	Improve Customer Satisfaction: Fair and transparent pricing based on personal health data can increase trust and satisfaction among policyholders.
-	Enable Personalized Offerings: Create customized insurance packages based on predicted costs, which can cater more directly to the needs and preferences of individuals.
-	Risk Assessment: Insurers can use the model to refine their risk assessment processes, identifying key factors that influence costs most significantly.
-	Policy Development: The insights gained from the model can inform the development of new insurance products or adjustments to existing ones.
-	Strategic Decision Making: Predictive analytics can aid in broader strategic decisions, such as entering new markets or adjusting policy terms based on risk predictions.
-	Customer Engagement: Insights from the model can be used in customer engagement initiatives, such as personalized marketing and tailored advice for policyholders.
## 📄 **Data Description**
The dataset comprises the following 11 attributes:
|Field| Description|
|-------------|-------------|
|	Age| Numeric, ranging from 18 to 66 years.|
|	Diabetes| Binary (0 or 1), where 1 indicates the presence of diabetes.|
|	BloodPressureProblems| Binary (0 or 1), indicating the presence of blood pressure-related issues.|
|	AnyTransplants| Binary (0 or 1), where 1 indicates the person has had a transplant.|
|	AnyChronicDiseases| Binary (0 or 1), indicating the presence of any chronic diseases.|
|Height| Numeric, measured in centimeters, ranging from 145 cm to 188 cm.|
|	Weight| Numeric, measured in kilograms, ranging from 51 kg to 132 kg.|
|	KnownAllergies| Binary (0 or 1), where 1 indicates known allergies.|
|	HistoryOfCancerInFamily| Binary (0 or 1), indicating a family history of cancer.|
|	NumberOfMajorSurgeries| Numeric, counting the number of major surgeries, ranging from 0 to 3 surgeries.|
|	PremiumPrice| Numeric, representing the premium price in currency, ranging from 15,000 to 40,000.|

## 🕵 Research Methodology

This project employs a data-driven approach to predict health insurance premiums using machine learning regression techniques. We began with data preprocessing, including feature selection and encoding of categorical variables, followed by multicollinearity analysis through Variance Inflation Factor (VIF) to retain essential predictors. Various regression models were tested, including Linear, Ridge, Lasso, Decision Tree, Random Forest, XGBoost, LightGBM, and Gradient Boosting. Each model was evaluated on R², Mean Absolute Error (MAE), and Mean Squared Error (MSE) metrics, with hyperparameter tuning applied to improve performance. Based on cross-validation and model accuracy, an ensemble of Random Forest, LightGBM, and Gradient Boosting was selected for final predictions, ensuring robustness and reliability in premium predictions.

## 🕵📝 Expected Outcomes

Insurance cost prediction is a complex yet rewarding task that requires a robust understanding of the relationships between health factors, demographic data, and premium pricing. Among the models tested, Random Forest achieved the highest accuracy, balancing both training and test R² scores with minimal overfitting. XGBoost and Decision Trees also performed well, though they required careful parameter tuning.

This analysis underscores the importance of features like age, chronic conditions, and organ transplants in driving premium costs. Predictive models that incorporate such factors can enable insurance providers to offer competitive, risk-adjusted pricing while ensuring fair access to coverage across diverse populations.

## App Link

Click the below link to play with ML model
https://insurancepricepredictormodel-nbxnkjq6i5xm4mcqdshiyn.streamlit.app/

## 📝 **Important Notes**

### Notebook Improvements
This repository has been optimized with the following corrections:
- **Fixed Imports**: Corrected import statements for regression problem (removed classification-specific imports)
- **Enhanced Data Loading**: Added universal data loading instructions for both Google Colab and local environments
- **Improved Documentation**: Fixed spelling errors and enhanced code readability
- **Application Enhancements**: Fixed age range inconsistencies and improved user interface

### Requirements
The complete list of dependencies is available in `requirements.txt`:
- streamlit
- scikit-learn==1.5.2
- numpy
- pandas
- lightgbm
- xgboost
- matplotlib
- seaborn
- scipy
- statsmodels

### Running the Application
```bash
pip install -r requirements.txt
streamlit run insurance_app_prediction.py
```





 

