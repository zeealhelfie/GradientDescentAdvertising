# GradientDescentAdvertising
Implementation of a gradient descent algorithm designed for optimizing advertising sales using Facebook data.

## Abstract

This research paper focuses on the application of the gradient descent algorithm to optimize advertising sales for a company using Facebook data. The objective is to improve the effectiveness of advertising campaigns by leveraging data-driven insights. The methodology involves performing a method for estimating the coefficients, and optimization algorithm on multiple linear regression, and evaluating the performance of the model each time using mean squared error (MSE) and R-squared R^2 metrics. A dataset from Kaggle datasets containing 1,143 rows and 11 columns was utilized for the analysis. The findings indicate that the gradient descent algorithm, in conjunction with multi-linear regression, demonstrates favorable results in optimizing advertising sales. The model achieved a fair appropriate estimation in MSE and an increase in R^2, indicating accuracy and predictive power. These findings suggest the potential of utilizing Facebook data and gradient descent algorithms for effective advertising strategies. 

## Introduction

Advertising sales play a crucial role in the success of online platforms. Effective advertising sales strategies can help businesses reach their target audience, generate revenue, and maximize their return on investment.

In this research, the objective is to explore the application of the gradient descent algorithm to optimize advertising sales using Facebook data. The gradient descent algorithm provides an optimization framework that can refine the relationship between various factors, such as ad spend, audience demographics, and campaign duration, and their impact on advertising sales. Through this approach, we aim to develop a predictive model that can effectively optimize advertising sales performance.

The evaluation of the model's performance will provide valuable insights into the effectiveness of the applied gradient descent algorithm for advertising sales optimization. By comparing the results obtained from the optimized model with those of the initial multi-linear regression model, we can assess the improvements achieved through the optimization process. Metrics such as mean squared error (MSE) and R-squared $(R^2)$ will be utilized to measure the accuracy and predictive power of the model.

## Time Outline:


1- Initial Data Analysis and Simple Linear Regression
  
- Data search and exploration
- Data cleaning
- Visualization of the data
- Interpretation of the visualizations and insights gained
- Fitting a simple linear regression (SLR) model
- Application of the Gradient Descent Method to SLR
- Step size selection for the Gradient Descent Method
  
2- Expansion to Multiple Linear Regression (MLR)
- Introduction of additional variables and features for more comprehensive analysis
- Construction of a multiple linear regression model incorporating the new variables
- Evaluation of the model performance and identification of potential improvements

3- Iterative Application of Gradient Descent
- Iterative application of the gradient descent algorithm to improve the accuracy of the MLR model
- Tracking the progression of optimization results and performance metrics
- Comparison of findings and insights gained during each iteration
- Feature selection process to identify the most relevant variables
- Evaluation of the impact of feature selection on model performance and optimization results

4- Data Processing and Feature Engineering
- Assessment of the impact of data processing and feature engineering on optimization results
- Examined the distribution of the predictor variables and applied appropriate normalization techniques to ensure better model performance.
- Explored the significance of understanding the data distribution and its impact on training MLR models, emphasizing the need for data distribution analysis as a critical step in preparing data for predictive modeling.

## Methodology:

### Dataset Overview:

The dataset used in this research comes from Kaggle datasets. The file contains 1143 observations (rows) in 11 variables (columns).

### Features: 

- ad\_id: represents the unique identifier for each advertisement.
- xyz\_campaign\_id: indicates the campaign ID associated with the advertisement on Facebook.
- fb\_campaign\_id: indicates the specific Facebook campaign ID for the advertisement.
- age: represents the age group of the target audience for the advertisement.
- gender: indicates the gender of the target audience for the advertisement.
- interest: represents the specific interest category or topic that the target audience is interested in.
- impressions: measures the number of times the advertisement was shown to users.
- clicks: represents the number of times users clicked on the advertisement.
- spent: indicates the amount of money spent on the advertisement.
- total\_conversion: measures the total number of conversions (desired actions) resulting from the advertisement.
- approved\_conversion: represents the number of conversions that were approved or validated.

### Response vs. Predictor Variables:

- Response Variable: spent.
- Predictor Variables: 10: [ad\_id, xyz\_campaign\_id, fb\_campaign\_id, age, gender, interest, impressions, clicks, total\_conversion, total\_conversion, approved\_conversion]

### Data Exploration and Preprocessing:

- Check for missing values:  No missing values
- Applying the Box-Cox Transformation: a technique used for stabilizing variance and addressing skewness in the data. Box-Cox transformation helps achieve a more normal distribution of the variable and satisfies the assumption of homoscedasticity. The Box-Cox transformation was applied to the Response Variable `spent`. Before the transformation, the skewness is 2.7, indicating a highly skewed distribution. After applying the Box-Cox transformation, the skewness is reduced to 0.018, approaching a more normal and symmetric distribution.
- Transforming object type features to integer type makes smoother analysis and modeling: the `age` and `gender` columns were transformed from object type to integer type. For the `age` column, the original data consisted of age ranges expressed as character strings, such as `30-34`, `35-39`, `40-44`, and `45-49`. A mapping was created to assign a specific number to each age range to convert these age ranges to numerical values. That number represents the mean of the interval. for the `gender` column, the original data contained categorical values denoting gender as `M` for males and `F` for females. To convert these categorical values to integers, we used a mapping that assigned the number 0 to `M` and the number 1 to `F`.  

### Feature Selection:

Feature selection involves identifying the most relevant and informative features from the dataset to improve model performance and interpretability. The method that was used in the feature selection process: 

- Stepwise Selection:
  Stepwise selection is a method used to select a subset of variables from a larger set of predictors. It iteratively adds or removes variables from the model based on their individual contribution to the Akaike Information Criterion (AIC). The AIC quantifies the trade-off between model complexity and goodness of fit. It aims to select a model that adequately explains the data while avoiding overfitting. Based on the p-values in the summary(Figure 1.2), we can determine which variables are statistically significant in relation to the response variable. Selecting variables with low p-values $<$ 0.05. All variables will be selected.   
  stepAIC(): function in R is used to perform the stepwise selection of variables in the linear regression model based on the AIC. The model with the lower AIC is considered to be a better fit for the data.

- Scaling by Standardization: StandardScaler()
  Scaling is primarily used to normalize the range and scale of the features in a dataset, making them comparable and preventing certain variables from dominating the analysis solely based on their scale. However, scaling can indirectly impact the treatment of outliers in some cases. When outliers are present in the dataset, scaling can influence their effect on the analysis. 
  Standardization: Scaling by standardization can weaken the impact of outliers to some extent. By subtracting the mean and dividing by the standard deviation, outliers can be `pulled` towards the mean and their influence on the analysis can be reduced. However, extreme outliers may still have a considerable impact even after scaling.
  z-score standardization: a technique that involves transforming the data in such a way that the resulting values have a mean of 0 and a standard deviation of 1. Scaling preserves the shape and distribution of the data but changes the scale. The standardization formula is as follows:
  $$z = \frac{x-\mu}{\sigma}$$
  
- z is the standardized value.
- x represents an individual data point.
- $\mu$ refers to the average value of each feature.
- $\sigma$ The standard deviation of the measure of the dispersion or spread of the values within each feature.
