## Data Set 
  
library(ISLR2)                    # Load library
Gitters <- na.omit(Hitters)       # Delete all the data with missing values
n <- nrow(Gitters)                # Create training and testing set. 
set.seed(13)
ntest <- trunc(n / 3)
testid <- sample(1:n, ntest)


#Preparation Phase, Gitters data set is a major league baseball player data set that contains a variety of information about each player regarding their baseball careers, including but not limited to:
  
#1.  AtBat - Number of times at bat in 1986

#2.  Hits - Number of hits in 1986

#3.  HmRun - Number of home runs in 1986

#4.  Run - Number of runs in 1986

#5.  RBI - Number of runs batted in 1986

#6.  Walks - Number of Walks in 1986

#7.  Anything that starts with a "C" represents a career statistics.

## Modeling

#1. Linear Regression
lfit <- lm(Salary ~ ., data = Gitters[-testid, ]) # Simple linear model using the training set.
lpred <- predict(lfit, Gitters[testid, ])         # Create a vector that stores prediction using testing set. 
with(Gitters[testid, ], mean(abs(lpred - Salary)))  # Calculate the mean absolute error of the data

#Notice the use of the with() command: the first argument is a data frame, and the second an expression that can refer to elements of the data frame by name. In this instance the data frame corresponds to the test data and the expression computes the mean absolute prediction error on this data.

#2. Lasso Regression

x <- scale(model.matrix(Salary ~ . - 1, data = Gitters)) # Create an scale a model matrix that's similar to linear model. 
y <- Gitters$Salary

library(glmnet)

cvfit <- cv.glmnet(x[-testid, ], y[-testid], type.measure = "mae") # Create the Lasso Regression
cpred <- predict(cvfit, x[testid, ], s = "lambda.min") # Create a plot using the lambda that produce the minimum MSE
mean(abs(y[testid] - cpred)) # Calculate the mean absolute error 

#plot(cvfit)
#plot(cpred)

#3. Neural NetWork

library(keras)

modnn <- keras_model_sequential() %>%
  layer_dense(units = 50, activation = "relu", 
              input_shape = ncol(x)) %>% # Create a hidden layer with 50 hidden units and a ReLU activation function
  layer_dropout(rate = 0.4) %>% # Create a dropout layer in which a random 40% of the 50 activation from the previous layer are set to zero during each iteration of stochastic gradiant descent algorithm (An optimization algorithm often used to find the model parameter that correspond to the best fit between predicted and actual outputs)
  layer_dense(units = 1) # The output layer has just one unit with no activation function, indicating that the model provides a single quantitative output.

modnn %>% compile(loss = "mse", # This adds details to the fitting algorithm that minimize the squared-error loss as it tracks the mean absolute error on the training data, and on the testing data if it is supplied. 
                  optimizer = optimizer_rmsprop(),
                  metrics = list("mean_absolute_error")
)

# Now we are actually fitting the model 
history <- modnn %>% fit(
  #    x[-testid, ], y[-testid], epochs = 1500, batch_size = 32,
  x[-testid, ], y[-testid], epochs = 600, batch_size = 32,
  validation_data = list(x[testid, ], y[testid])
)

npred <- predict(modnn, x[testid, ])
mean(abs(y[testid] - npred))
