# GradientDescentAdvertising
Implementation of a gradient descent algorithm designed for optimizing advertising sales using Facebook data.

## Abstract

This research paper focuses on the application of the gradient descent algorithm to optimize advertising sales for a company using Facebook data. The objective is to improve the effectiveness of advertising campaigns by leveraging data-driven insights. The methodology involves performing a method for estimating the coefficients, and optimization algorithm on multiple linear regression, and evaluating the performance of the model each time using mean squared error (MSE) and R-squared R^2 metrics. A dataset from Kaggle datasets containing 1,143 rows and 11 columns was utilized for the analysis. The findings indicate that the gradient descent algorithm, in conjunction with multi-linear regression, demonstrates favorable results in optimizing advertising sales. The model achieved a fair appropriate estimation in MSE and an increase in R^2, indicating accuracy and predictive power. These findings suggest the potential of utilizing Facebook data and gradient descent algorithms for effective advertising strategies. 

## Introduction

Advertising sales play a crucial role in the success of online platforms. Effective advertising sales strategies can help businesses reach their target audience, generate revenue, and maximize their return on investment.

In this research, the objective is to explore the application of the gradient descent algorithm to optimize advertising sales using Facebook data. The gradient descent algorithm provides an optimization framework that can refine the relationship between various factors, such as ad spend, audience demographics, and campaign duration, and their impact on advertising sales. Through this approach, we aim to develop a predictive model that can effectively optimize advertising sales performance.

The evaluation of the model's performance will provide valuable insights into the effectiveness of the applied gradient descent algorithm for advertising sales optimization. By comparing the results obtained from the optimized model with those of the initial multi-linear regression model, we can assess the improvements achieved through the optimization process. Metrics such as mean squared error (MSE) and R-squared $(R^2)$ will be utilized to measure the accuracy and predictive power of the model.

## Time Outline:

\begin{enumerate}
  \item Initial Data Analysis and Simple Linear Regression
  \begin{itemize}
    \item Data search and exploration
    \item Data cleaning
    \item Visualization of the data
    \item Interpretation of the visualizations and insights gained
    \item Fitting a simple linear regression (SLR) model
    \item Application of the Gradient Descent Method to SLR
    \item Step size selection for the Gradient Descent Method
  \end{itemize}

  \item Expansion to Multiple Linear Regression (MLR)
  \begin{itemize}
    \item Introduction of additional variables and features for more comprehensive analysis
    \item Construction of a multiple linear regression model incorporating the new variables
    \item Evaluation of the model performance and identification of potential improvements
  \end{itemize}

  \item Iterative Application of Gradient Descent
  \begin{itemize}
    \item Iterative application of the gradient descent algorithm to improve the accuracy of the MLR model
    \item Tracking the progression of optimization results and performance metrics
    \item Comparison of findings and insights gained during each iteration
    \item Feature selection process to identify the most relevant variables
    \item Evaluation of the impact of feature selection on model performance and optimization results
  \end{itemize}

  \item Data Processing and Feature Engineering
    \begin{itemize}
    \item Assessment of the impact of data processing and feature engineering on optimization results
    \item Examined the distribution of the predictor variables and applied appropriate normalization techniques to ensure better model performance.
    \item Explored the significance of understanding the data distribution and its impact on training MLR models, emphasizing the need for data distribution analysis as a critical step in preparing data for predictive modeling.
  \end{itemize}
\end{enumerate}
