# **Insurance_price_predictor_model**

![image](https://github.com/user-attachments/assets/985a00ab-2247-4aae-a768-acd69b84498b)

## **About the Case study**
Insurance companies need to accurately predict the cost of health insurance for individuals to set premiums appropriately. However, traditional methods of cost prediction often rely on broad actuarial tables and historical averages, which may not account for the nuanced differences among individuals. By leveraging machine learning techniques, insurers can predict more accurately the insurance costs tailored to individual profiles, leading to more competitive pricing and better risk management.
## **Objective**
The primary need for this project arises from the challenges insurers face in pricing policies accurately while remaining competitive in the market. Inaccurate predictions can lead to losses for insurers and unfairly high premiums for policyholders. By implementing a machine learning model, insurers can:
- Enhance Precision in Pricing: Use individual data points to determine premiums that reflect actual risk more closely than generic estimates.
- Increase Competitiveness: Offer rates that are attractive to consumers while ensuring that the pricing is sustainable for the insurer.
- Improve Customer Satisfaction: Fair and transparent pricing based on personal health data can increase trust and satisfaction among policyholders.
- Enable Personalized Offerings: Create customized insurance packages based on predicted costs, which can cater more directly to the needs and preferences of individuals.
- Risk Assessment: Insurers can use the model to refine their risk assessment processes, identifying key factors that influence costs most significantly.
- Policy Development: The insights gained from the model can inform the development of new insurance products or adjustments to existing ones.
- Strategic Decision Making: Predictive analytics can aid in broader strategic decisions, such as entering new markets or adjusting policy terms based on risk predictions.
- Customer Engagement: Insights from the model can be used in customer engagement initiatives, such as personalized marketing and tailored advice for policyholders.
## **Data description**
The dataset comprises the following 11 attributes:
| Field| Description|
|-----------|-----------|
|	Age| Numeric, ranging from 18 to 66 years.|
| Diabetes| Binary (0 or 1), where 1 indicates the presence of diabetes.|
|	BloodPressureProblems| Binary (0 or 1), indicating the presence of blood pressure-related issues.|
|	AnyTransplants| Binary (0 or 1), where 1 indicates the person has had a transplant.|
|AnyChronicDiseases| Binary (0 or 1), indicating the presence of any chronic diseases.|
|	Height| Numeric, measured in centimeters, ranging from 145 cm to 188 cm.|
|	Weight| Numeric, measured in kilograms, ranging from 51 kg to 132 kg|
|	KnownAllergies| Binary (0 or 1), where 1 indicates known allergies.|
|	HistoryOfCancerInFamily| Binary (0 or 1), indicating a family history of cancer|
|	NumberOfMajorSurgeries|Numeric, counting the number of major surgeries, ranging from 0 to 3 surgeries|
|	PremiumPrice| Numeric, representing the premium price in currency, ranging from 15,000 to 40,000|

## **🕵 Research Methodology**

- **Data Collection**: Gather relevant data from a insurance company, on the age, height,weight, premium price details along with list if they have Diabetes, Blood Pressure Problems, AnyTransplants, Any Chronic Diseases, KnownAllergies,	HistoryOfCancerInFamily and NumberOfMajorSurgeries.
- **Data Analysis**: We will use statistical and machine learning techniques to analyse the data and identify the patterns and correlation between the variables and the Premium price. We will also perform T-test for for categoical variables with respect to premium price, Chi square test for categorical columns with respect to other categorical columns to identfy any dependecy between them.
- **Visualization**: Create visual representations of the findings to make the insights more accessible and actionable for stakeholders.
- **Model Building**: We will start with simple linear regression models by transforming the data using minmax scaler, then we proceed with Logistic Regression, Descision Tree Regression, Random Forest Regression, Gradiant boosting Regression, XGBoost Regression and Light GBM. For each model we will evaluate the r2 score on train and test data to identify which models performs with best r2 score and use it to predict Premium price.
- **Model Deployment**: We will package the model using pickle and then use streamlite to build the front end and integrate the model with the front end to feed to model with necessary data inputs and hten the modle can predict the premium price.

## **🕵📝 Expected Outcomes**
Based on the age, height, weight and the type of diseases we give the input to the app, the model will predict the insurance price and will provide the output Premium Price which will help the insurance seller to determine at what price range can they provide the Insurance Premium for their clients.

![image](https://github.com/user-attachments/assets/5f302707-bad4-4eef-b146-1c026e796034)


