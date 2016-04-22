library(glmnet)
library(caret)

# Read data and split into 70% train and 30% test
data = read.csv("default\ of\ credit\ card\ clients.csv", skip = 1)
n = nrow(data)
split = 0.7
trainIdx = sample(1:n, round(n*split))
xTrain = as.matrix(data[trainIdx, -25])
yTrain = as.matrix(data[trainIdx, 25])
xTest = as.matrix(data[-trainIdx, -25])
yTest = as.matrix(data[-trainIdx, 25])

# Build logistic regression with different regularization schemes
lasso = cv.glmnet(xTrain, yTrain, family = "binomial", type.measure = "class", alpha = 1)
ridge = cv.glmnet(xTrain, yTrain, family = "binomial", type.measure = "class", alpha = 0)
enet0.25 = cv.glmnet(xTrain, yTrain, family = "binomial", type.measure = "class", alpha = 0.25)
enet0.50 = cv.glmnet(xTrain, yTrain, family = "binomial", type.measure = "class", alpha = 0.50)
enet0.75 = cv.glmnet(xTrain, yTrain, family = "binomial", type.measure = "class", alpha = 0.75)

# Predict the binomial label using the regression models
yNoreg = predict(enet0.50, xTest, s = 0, type = "class") #set lambda equal to 0 in order to turn off regularization
yLasso = predict(lasso, xTest, s = "lambda.min", type = "class")
yRidge = predict(ridge, xTest, s = "lambda.min", type = "class")
yEnet0.25 = predict(enet0.25, xTest, s = "lambda.min", type = "class")
yEnet0.50 = predict(enet0.50, xTest, s = "lambda.min", type = "class")
yEnet0.75 = predict(enet0.75, xTest, s = "lambda.min", type = "class")

# Output the accuracy of each measurement
nTest = nrow(yTest)
noregAccur = sum(yNoreg == yTest)/nTest
lassoAccur = sum(yLasso == yTest)/nTest
ridgeAccur = sum(yRidge == yTest)/nTest
enetAccur0.25 = sum(yEnet0.25 == yTest)/nTest
enetAccur0.50 = sum(yEnet0.50 == yTest)/nTest
enetAccur0.75 = sum(yEnet0.75 == yTest)/nTest
