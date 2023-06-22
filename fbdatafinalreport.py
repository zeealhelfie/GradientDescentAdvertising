# %%
import warnings
warnings.filterwarnings('ignore')

# Import package
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import hvplot.pandas
import pandas as pd 
import statsmodels.api as sm
from matplotlib import rcParams
from matplotlib import cm
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from tabulate import tabulate
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D
rcParams.update({'font.size': 18})
from scipy.stats import norm

# %%
# data
advertising = pd.read_csv("fbdata.csv")
print(advertising.info())

# %%
# Replace character string age ranges with number
advertising.loc[advertising['age'] == '30-34', 'age'] = 32
advertising.loc[advertising['age'] == '35-39', 'age'] = 37
advertising.loc[advertising['age'] == '40-44', 'age'] = 42
advertising.loc[advertising['age'] == '45-49', 'age'] = 47

# Convert gender variable to integer
advertising['gender'] = advertising['gender'].replace({'M': 0, 'F': 1})
advertising['gender'] = advertising['gender'].astype(int)



# %%
# clean up column names that contain whitespace
advertising.columns = advertising.columns.str.replace(' ', '')
advertising.head()

# %%
# abbreviate some variable names
advertising.fillna(0, inplace=True)
advertising = advertising.rename(columns={"Total_Conversion": "conv", "Impressions": "impr", "Approved_Conversion": "appConv", "xyz_campaign_id": "xyzID", "fb_campaign_id": "fbID" })
print(advertising.head())

# %%
# scatter plor
pd.plotting.scatter_matrix(advertising, figsize=(10, 10))
plt.show()

# %%
print(advertising['Spent'].skew())

# %%
# Plot a histogram of the 'Spent' column
plt.hist(advertising['Spent'], bins=50)
plt.xlabel('Count')
plt.ylabel('Spent')
plt.show()

# %%
# The Box-Cox transformation requires all data to be positive. 
# Assuming that 'Spent' has no negative values:

# Add a small constant to avoid zero
advertising['Spent'] = advertising['Spent'] + 1

# Apply Box-Cox transformation
advertising['Spent'], _ = stats.boxcox(advertising['Spent'])

# Check the skewness 
print(advertising['Spent'].skew())


# %%
# Histogram
sns.histplot(advertising['Spent'], kde=True, stat='density')


# %%
#Pearson correlation coefficient
correlations = advertising.corr(method='pearson')
correlations

# %% [markdown]
# Simple Linear Regression:

# %%
X = advertising['Clicks']
Y = advertising['Spent']

# %%
# Let's first plot (x,y) and see what it looks like
plt.plot('Clicks','Spent',data = advertising, marker = 'o', linestyle = '', label = 'data')
plt.xlabel('x',fontsize = 18)
plt.ylabel('y', fontsize = 18)
plt.legend(fontsize = 14)
plt.show()

# %%

class Linear_Regression:
	def __init__(self, X, Y):
		self.X = X
		self.Y = Y
		self.b = [0, 0] #np.random.rand(2)
	

	# Parameter Updation
	def update_coeffs(self, learning_rate):
		Y_pred = self.predict()
		Y = self.Y
		m = len(Y)
		
			
			# b = b - (learning_rate * (dJ/db))
			# The trace of A is the sum of its diagonal entries
		self.b[0] = self.b[0] - (learning_rate * ((1/m) *
								np.sum(Y_pred - Y)))

		self.b[1] = self.b[1] - (learning_rate * ((1/m) *
								np.sum((Y_pred - Y) * self.X)))
		
		return self.b
		
	#prediction function: (y) = beta_0 + (beta_1 * x) adding 
	def predict(self, X=[]):
		Y_pred = np.array([])
		if not X: X = self.X
		b = self.b
		for x in X:
			Y_pred = np.append(Y_pred, b[0] + (b[1] * x)) 
		return Y_pred
	
	def get_current_accuracy(self, Y_pred):
		p, e = Y_pred, self.Y
		n = len(Y_pred)
		return 1-sum(
			[
				abs(p[i]-e[i])/e[i]
				for i in range(n)
				if e[i] != 0]
		)/n
	
	# MSE: GS
	def compute_cost(self, Y_pred):
		m = len(self.Y)
		J = 1/(m) * np.sum((Y_pred - Y)**2)
		return J

	


	def plot_best_fit(self, Y_pred, fig):
				f = plt.figure(fig)
				plt.scatter(self.X, self.Y, color='b')
				plt.plot(self.X, Y_pred, color='r')
				f.show()
    
    

X = np.array([x for x in X])
Y = np.array([y for y in Y])


def main():
    regressor = Linear_Regression(X, Y)
    iterations = 0
    steps = 500
    learning_rate = 0.00001
    costs = [] #mse
    updated_b = []  # store updated values of b
    
    

    #original best-fit line
    Y_pred = regressor.predict()
    regressor.plot_best_fit(Y_pred, 'Initial Best Fit Line')
    
    while 1:
        Y_pred = regressor.predict()
        cost = regressor.compute_cost(Y_pred)
        costs.append(cost)
        
        
        updated_b = regressor.update_coeffs(learning_rate)  # get updated values of b
        iterations += 1

        if iterations % steps == 0:
            print(iterations, "iterations")
            
            
            
            stop = input("Do you want to stop (y/*)??")
            if stop == "y":
                break
 
            
            # Check for convergence
        if len(costs) >= 2 and abs(costs[-1] - costs[-2]) < 1e-7:
            print("Converged at iteration:", iterations)
            break
            

            
    
    print("Updated coefficients", updated_b)  # print updated values of b for last iteration
    print("Cost(MSE):", cost)
    print("Current accuracy is :",
                  regressor.get_current_accuracy(Y_pred))
   
    #final best-fit line
    regressor.plot_best_fit(Y_pred, 'Final Best Fit Line')

    #plot to verify cost function decreases
    h = plt.figure('Verification')
    plt.plot(range(iterations), costs, color='b')
    h.show()

    # if user wants to predict using the regressor:
    #regressor.predict([i for i in range(0)])
    
    

if __name__ == '__main__':
    main()


# %%
#Creating additional features: 
#advertising.fillna(0, inplace=True)
advertising['CTR'] = 100 * advertising['Clicks'] / advertising['impr']
advertising['CPC'] = advertising['Spent'] / advertising['Clicks']
advertising['CPI'] = advertising['Spent'] / advertising['impr']
advertising.head()

# %%
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
# Fill in any missing values with 0
advertising.fillna(0, inplace=True)

# Prepare the data for linear regression
X = advertising[['Clicks','age','gender', 'impr', 'interest', 'conv', 'ad_id','xyzID','fbID',	'appConv']]
Y = advertising['Spent']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Normalize the input features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# %%
print(X.shape)
print(X_test.shape)
print(Y_test.size)
print(Y.size)

# %% [markdown]
# Multi Linear regression:

# %%
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Fill in any missing values with 0
advertising.fillna(0, inplace=True)

# Prepare the data for linear regression
X = advertising[['Clicks','age','gender', 'impr', 'interest', 'conv', 'ad_id','xyzID','fbID',	'appConv']]
Y = advertising['Spent']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Normalize the input features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Fit a linear regression model
model = LinearRegression()
model.fit(X_train, Y_train)

# Print the intercept and coefficients
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

# Make predictions on the training set
Y_train_pred = model.predict(X_train)

# Calculate the mean squared error on the training set
ml_train_mse = mean_squared_error(Y_train, Y_train_pred)
ml_train_rmse = np.sqrt(ml_train_mse)

# Calculate the R-squared value for the training set
ml_train_r2 = r2_score(Y_train, Y_train_pred)


# Make predictions on the test set
Y_test_pred = model.predict(X_test)

# Calculate the mean squared error on the test set
ml_test_mse = mean_squared_error(Y_test, Y_test_pred)
ml_test_rmse = np.sqrt(ml_test_mse)

# Calculate the R-squared value for the test set
ml_test_r2 = r2_score(Y_test, Y_test_pred)



print("Training MSE:", ml_train_mse)
print("Test MSE:", ml_test_mse)
print("Training R-squared value:", ml_train_r2)
print("Test R-squared value:", ml_test_r2)
print("Training RMSE:", ml_train_rmse)
print("Test RMSE:", ml_test_rmse)

# %%
print(Y_train.shape)
print(Y_test.shape)
print(Y_train_pred.shape)
print(Y_test_pred.shape)


# %% [markdown]
# Ridge and Lasso:

# %%

# Prepare the data for linear regression
#advertising.fillna(0, inplace=True)
X = advertising[['Clicks','age','gender', 'impr', 'interest', 'conv', 'ad_id','xyzID','fbID', 'appConv']]
Y = advertising['Spent']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# normalizing the data by subtracting the mean and dividing by the standard deviation may not work well with outliers since they can skew the mean and standard deviation values.

# Normalize the input features. 
#X_train = (X_train - X_train.mean()) / X_train.std()
#X_test = (X_test - X_test.mean()) / X_test.std()


# StandardScaler method not only normalizes the data but also standardizes it by subtracting the mean and dividing by the standard deviation, which is useful in cases where the data may have different scales or units. Additionally, the StandardScaler method can handle outliers well since it uses the mean and standard deviation to scale the data.

# Create a StandardScaler object
scaler = StandardScaler()
# Fit the scaler on the training data and transform the data
X_train = scaler.fit_transform(X_train)
# Transform the testing data using the fitted scaler
X_test = scaler.transform(X_test)


# Initialize the hyperparameters
alpha = 0.01
num_iters = 100
ridge_alpha = 0.01

# Create a Ridge model and fit it on the training data
ridge_model = Ridge(alpha=ridge_alpha)
ridge_model.fit(X_train, Y_train)

# Print the model parameters and R-squared value
Y_pred = ridge_model.predict(X_test)
Ridge_mse = np.mean((Y_test - Y_pred) ** 2)
Ridge_r2 = r2_score(Y_test, Y_pred)
print("Ridge regression")
print("Intercept:", ridge_model.intercept_)
print("Coefficients:", ridge_model.coef_)
print("MSE:", Ridge_mse)
print("R-squared value:", Ridge_r2)
Ridge_rmse = np.sqrt(Ridge_mse)
print("Regression RMSE:", Ridge_rmse) 

# Create a Lasso model and fit it on the training data
lasso_model = Lasso(alpha=0.01)
lasso_model.fit(X_train, Y_train)

# Print the model parameters and R-squared value
Y_pred = lasso_model.predict(X_test)
Lasso_mse = np.mean((Y_test - Y_pred) ** 2)
Lasso_r2 = r2_score(Y_test, Y_pred)
print("\nLasso regression")
print("Intercept:", lasso_model.intercept_)
print("Coefficients:", lasso_model.coef_)
print("MSE:", Lasso_mse)
print("R-squared value:", Lasso_r2)
Lasso_rmse = np.sqrt(Lasso_mse)
print("Regression RMSE:", Lasso_rmse)

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class MultiLinearRegression:
    def __init__(self, X, Y):
        self.X = np.hstack((np.ones((X.shape[0], 1)), X))
        self.Y = Y.values.reshape(-1, 1)
        self.b = np.zeros((self.X.shape[1], 1))

    def update_coeffs(self, learning_rate):
        Y_pred = self.predict()
        m = len(self.Y)
        error = Y_pred - self.Y
        gradient = (1/m) * np.dot(self.X.T, error)
        self.b -= learning_rate * gradient
        return self.b

    def predict(self, X=None):
        if X is None:
            X = self.X
        return np.dot(X, self.b)

    def compute_cost(self, Y_pred):
        m = len(self.Y)
        J = (1/(2*m)) * np.sum((Y_pred - self.Y)**2)
        return J

    def r_squared(self, Y_pred):
        sse = np.sum((self.Y - Y_pred)**2)
        sst = np.sum((self.Y - np.mean(self.Y))**2)
        r2 = 1 - (sse/sst)
        return r2

    def plot_best_fit(self, Y_pred, fig):
        f = plt.figure(fig)
        plt.scatter(self.X[:, 1], self.Y, color='b')
        plt.plot(self.X[:, 1], Y_pred, color='r')
        f.show()

advertising.fillna(0, inplace=True)

# Prepare the data for linear regression
#X = advertising[['Clicks', 'impr', 'interest', 'conv', 'ad_id','xyzID','fbID',	'appConv',	'CTR',	'CPC',	'CPI']]
X = advertising[['Clicks', 'impr', 'conv', 'ad_id', 'appConv']]
Y = advertising['Spent']


# Normalize the input features
X = (X - X.mean()) / X.std()

regressor = MultiLinearRegression(X, Y)

iterations = 0
steps = 500
learning_rate = 0.1
convergence_threshold = 1e-6
costs = []
updated_b = []



while 1:
    Y_pred = regressor.predict()
    r2 = regressor.r_squared(Y_pred)
    cost = regressor.compute_cost(Y_pred)
    costs.append(cost)

    updated_b = regressor.update_coeffs(learning_rate)
    iterations += 1

    if iterations % steps == 0:
        print("Iteration:", iterations)
        
        stop = input("Do you want to stop (y/*)??")
        if stop == "y":
            break
        
        if len(costs) >= 2 and np.abs(cost - costs[-2])<1e-9:
            print("Converged at iteration:", iterations)  
            break

mse = mean_squared_error(Y, Y_pred)
print("Updated b:", updated_b.ravel())
print("Cost:", cost)
print("R-squared:", r2)
print("Mean squared error:", mse)
rmse = np.sqrt(mse)
print("RMSE:", rmse)

# %%
# Fill in any missing values with 0
advertising.fillna(0, inplace=True)

from sklearn.preprocessing import StandardScaler

# Prepare the data for linear regression
X = advertising[['Clicks','age','gender', 'impr', 'interest', 'conv', 'ad_id','xyzID','fbID',	'appConv']]
Y = advertising['Spent']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Normalize the input features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Add a column of ones for the bias term
X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_test = np.c_[np.ones(X_test.shape[0]), X_test]

# Initialize the model parameters
theta = np.zeros(X_train.shape[1])

# Set the learning rate and number of iterations
alpha = 0.1
num_iters = 1000

# Define the cost function
def compute_cost(X, Y, theta):
    m = Y.shape[0]
    J = 1 / (2*m) * np.sum((np.dot(X, theta) - Y) ** 2)
    return J

# Define the gradient descent algorithm
def gradient_descent(X, Y, theta, alpha, num_iters):
    m = Y.shape[0]
    J_history = np.zeros(num_iters)
    for i in range(num_iters):
        theta -= alpha * (1 / m) * np.dot(X.T, (np.dot(X, theta) - Y))
        J_history[i] = compute_cost(X, Y, theta)
    return theta, J_history

# Fit the linear regression model using gradient descent
theta, J_history = gradient_descent(X_train, Y_train, theta, alpha, num_iters)

# Make predictions on the test/train set
Y_pred_test = np.dot(X_test, theta)
Y_pred_train = np.dot(X_train, theta)

# Calculate evaluation metrics on both training and test sets
mse_train = np.mean((Y_pred_train - Y_train)**2)
mse_test = np.mean((Y_pred_test - Y_test)**2)
tss_test = np.sum((Y_test - Y_test.mean())**2)
rss_test = np.sum((Y_test - Y_pred_test)**2)
r_squared_test = 1 - (rss_test / tss_test)
tss_train = np.sum((Y_train - Y_train.mean())**2)
rss_train = np.sum((Y_train - Y_pred_train)**2)
r_squared_train = 1 - (rss_train / tss_train)
rmse_train = np.sqrt(mse_train)
rmse_test = np.sqrt(mse_test)

# Print the model parameters, final cost, and evaluation metrics
print("Intercept:", theta[0])
print("Coefficients:", theta[1:])
print("Final cost:", J_history[-1])
print('MSE on training set:', mse_train)
print('MSE on test set:', mse_test)
print('R-squared on training set:', r_squared_train)
print('R-squared on test set:', r_squared_test)
print("RMSE on training set:", rmse_train)
print("RMSE on test set:", rmse_test)


# %%
# Plot the cost function over time
plt.plot(np.arange(num_iters), J_history)
plt.xlabel("Iteration")
plt.ylabel("Cost function")
plt.show()

# %%
# Create a table of the mean squared error and R-squared value for all models
table = [["The ordinary least squares: Training", round
          (ml_train_mse, 2), round(ml_train_r2*100, 2), round(ml_test_rmse, 2)]
         ,["The ordinary least squares: Testing", round(ml_test_mse, 2), round(ml_test_r2*100, 2), round(ml_test_rmse, 2)],
         #["Ridge regression", round(Ridge_mse, 2), round(Ridge_r2*100, 2), round(Ridge_rmse, 2)],
         #["Lasso regression", round(Lasso_mse, 2), round(Lasso_r2*100, 2), round(Lasso_rmse, 2)],
         ["Gradient descent: Training", round(mse_train, 2), round(r_squared_test*100, 2), round(rmse_train, 2)],
         ["Gradient descent: Testing", round(mse_test, 2), round(r_squared_train*100, 2), round(rmse_test, 2)]]

print(tabulate(table, headers=["Model: Linear regression", "MSE", "R-squared(%)"], tablefmt="pretty"))

#Root Mean Squared Erro


