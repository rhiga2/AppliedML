library(glmnet)
library(MASS)
library(car)

#Import Dataset
data = read.csv("Geographical\ Original\ of\ Music/default_plus_chromatic_features_1059_tracks.txt", header = FALSE)
n = nrow(data)
split = 0.7
trainIdx = sample(1:n, round(n*split))
train = data[trainIdx, ]
test = data[-trainIdx, ]
ntest = nrow(test)
latitudeTrain = as.matrix(train[, 117])
longitudeTrain = as.matrix(train[, 118])
featuresTrain = as.matrix(train[, -c(117,118)])
latitudeTest = as.matrix(test[, 117])
longitudeTest = as.matrix(test[, 118])
featuresTest = as.matrix(test[, -c(117, 118)])

#Make simple linear model
latLm = lm(V117 ~ . - V117 - V118, data = train)
latLmEstimate = predict(latLm, test)
plot(latitudeTest, latLmEstimate)
latLmRes = latitudeTest - latLmEstimate #residuals
plot(latLmEstimate, latLmRes, main = "Latitudinal Simple Regression Residuals",
     xlab = "Fitted Values", ylab = "Residuals") #fitted value versus residuals
latLmMSE = sum((latLmRes)^2)/ntest 

longLm = lm(V118 ~ . - V117 - V118, data = train)
longLmEstimate = predict(longLm, test)
plot(longitudeTest, longLmEstimate)
longLmRes = longitudeTest - longLmEstimate #residuals
plot(longLmEstimate, longLmRes, main = "Longitudinal Simple Regression Residuals",
     xlab = "Fitted Values", ylab = "Residuals") #fitted value versus residuals
longLmMSE = sum((longLmRes)^2)/ntest

#Test Box-Cox Transformation
trainBC = train
testBC = test
trainBC$V117 = train$V117 + 90
trainBC$V118 = train$V118 + 90
testBC$V117 = test$V117 + 90
testBC$V118 = test$V118 + 90

# Transform latitude with best lambda (box-cox NOT regularization constant)
latLambdas = boxcox(V117 ~ . - V117 - V118, data = trainBC, lambda = seq(0, 5, 0.1))
latMaxLambda = powerTransform(trainBC$V117) 
trainBC$V117 = bcPower(trainBC$V117, latMaxLambda$lambda)
testBC$V117 = bcPower(testBC$V117, latMaxLambda$lambda)

# Linearly regress on transformed latitudes
latLmBC = lm(V117 ~ . - V117 - V118, data = trainBC)
latLmEstimateBC = predict(latLmBC, testBC)
plot(testBC$V117, latLmEstimateBC)
latLmRes = testBC$V117 - latLmEstimateBC
plot(latLmEstimateBC, latLmRes, main = "Latitudinal Box-Cox Regression Residuals",
     xlab = "Fitted Values", ylab = "Residuals") #fitted value versus residuals
latLmBCMSE = sum((latLmRes)^2)/ntest

# Transform longitude with best lambda 
longLambdas = boxcox(V118 ~ . - V117 - V118, data = trainBC, lambda = seq(0, 5, 0.1))
longMaxLambda = powerTransform(trainBC$V118)
trainBC$V118 = bcPower(trainBC$V118, longMaxLambda$lambda)
testBC$V118 = bcPower(testBC$V118, longMaxLambda$lambda)

# Linearly regress on transformed longitudes
longLmBC = lm(V118 ~ . - V117 - V118, data = trainBC)
longLmEstimateBC = predict(longLmBC, testBC)
plot(testBC$V118, longLmEstimateBC)
longLmRes = testBC$V118 - longLmEstimateBC
plot(longLmEstimateBC, longLmRes, main = "Longitudinal Box-Cox Regression Residuals",
     xlab = "Fitted Values", ylab = "Residuals") #fitted value versus residuals
longLmBCMSE = sum((longLmRes)^2)/ntes

#Create regularized linear regression models for latitude
latLasso = cv.glmnet(featuresTrain, latitudeTrain, alpha = 1)
latRidge = cv.glmnet(featuresTrain, latitudeTrain, alpha = 0)
latElasticNet0.25 = cv.glmnet(featuresTrain, latitudeTrain, alpha = 0.25)
latElasticNet0.50 = cv.glmnet(featuresTrain, latitudeTrain, alpha = 0.50)
latElasticNet0.75 = cv.glmnet(featuresTrain, latitudeTrain, alpha = 0.75)

#Plot cross-validation of lambda parameters based on the mean-squared error
plot(latLasso)
plot(latRidge)
plot(latElasticNet0.25)
plot(latElasticNet0.50)
plot(latElasticNet0.75)

#Predict latitude using the different regularization schemes
latLassoEstimate = predict(latLasso, featuresTest, s = "lambda.min")
latRidgeEstimate = predict(latRidge, featuresTest, s = "lambda.min")
latElasticNetEstimate0.25 = predict(latElasticNet0.25, featuresTest, s = "lambda.min")
latElasticNetEstimate0.50 = predict(latElasticNet0.50, featuresTest, s = "lambda.min")
latElasticNetEstimate0.75 = predict(latElasticNet0.75, featuresTest, s = "lambda.min")

#Evaluate the MSE for regularized regression on latitude
latLassoMSE = sum((latitudeTest - latLassoEstimate)^2) / ntest
latRidgeMSE = sum((latitudeTest - latRidgeEstimate)^2) / ntest
latElasticNetMSE0.25 = sum((latitudeTest - latElasticNetEstimate0.25)^2) / ntest
latElasticNetMSE0.50 = sum((latitudeTest - latElasticNetEstimate0.50)^2) / ntest
latElasticNetMSE0.75 = sum((latitudeTest - latElasticNetEstimate0.75)^2) / ntest

#Create regularized linear regression for longitude
longLasso = cv.glmnet(featuresTrain, longitudeTrain, alpha = 1)
longRidge = cv.glmnet(featuresTrain, longitudeTrain, alpha = 0)
longElasticNet0.25 = cv.glmnet(featuresTrain, longitudeTrain, alpha = 0.25)
longElasticNet0.50 = cv.glmnet(featuresTrain, longitudeTrain, alpha = 0.50)
longElasticNet0.75 = cv.glmnet(featuresTrain, longitudeTrain, alpha = 0.75)

#Plot cross-validation of lambda parameters based on the mean-squared error
plot(longLasso)
plot(longRidge)
plot(longElasticNet0.25)
plot(longElasticNet0.50)
plot(longElasticNet0.75)

#Predict longitude using the different regularization schemes
longLassoEstimate = predict(longLasso, featuresTest, s = "lambda.min")
longRidgeEstimate = predict(longRidge, featuresTest, s = "lambda.min")
longElasticNetEstimate0.25 = predict(longElasticNet0.25, featuresTest, s = "lambda.min")
longElasticNetEstimate0.50 = predict(longElasticNet0.50, featuresTest, s = "lambda.min")
longElasticNetEstimate0.75 = predict(longElasticNet0.75, featuresTest, s = "lambda.min")


#Evaluate the MSE for regularized regression on latitude
longLassoMSE = sum((longitudeTest - longLassoEstimate)^2) / ntest
longRidgeMSE = sum((longitudeTest - longRidgeEstimate)^2) / ntest
longElasticNetMSE0.25 = sum((longitudeTest - longElasticNetEstimate0.25)^2) / ntest
longElasticNetMSE0.50 = sum((longitudeTest - longElasticNetEstimate0.50)^2) / ntest
longElasticNetMSE0.75 = sum((longitudeTest - longElasticNetEstimate0.75)^2) / ntest

#subplots in R
par(mfrow=c(2,3))
plot(latLmEstimate, latitudeTest - latLmEstimate, main = "Latitudinal No Reg", xlab = "Fitted Values", ylab = "Residuals")
plot(latLassoEstimate, latitudeTest - latLassoEstimate, main = "Latitudinal Lasso", xlab = "Fitted Values", ylab = "Residuals")
plot(latRidgeEstimate, latitudeTest - latRidgeEstimate, main = "Latitudinal Ridge", xlab = "Fitted Values", ylab = "Residuals")
plot(latElasticNetEstimate0.25, latitudeTest - latElasticNetEstimate0.25, main = "Latitudinal Elastic Net 0.25", xlab = "Fitted Values", ylab = "Residuals")
plot(latElasticNetEstimate0.50, latitudeTest - latElasticNetEstimate0.50, main = "Latitudinal Elastic Net 0.50", xlab = "Fitted Values", ylab = "Residuals")
plot(latElasticNetEstimate0.75, latitudeTest - latElasticNetEstimate0.75, main = "Latitudinal Elastic Net 0.75", xlab = "Fitted Values", ylab = "Residuals")

#subplots in R
par(mfrow=c(2,3))
plot(longLmEstimate, longitudeTest - longLmEstimate, main = "Longitudinal No Reg", xlab = "Fitted Values", ylab = "Residuals")
plot(longLassoEstimate, longitudeTest - longLassoEstimate, main = "Longitudinal Lasso", xlab = "Fitted Values", ylab = "Residuals")
plot(longRidgeEstimate, longitudeTest - longRidgeEstimate, main = "Longitudinal Ridge", xlab = "Fitted Values", ylab = "Residuals")
plot(longElasticNetEstimate0.25, longitudeTest - longElasticNetEstimate0.25, main = "Longitudinal Elastic Net 0.25", xlab = "Fitted Values", ylab = "Residuals")
plot(longElasticNetEstimate0.50, longitudeTest - longElasticNetEstimate0.50, main = "Longitudinal Elastic Net 0.50", xlab = "Fitted Values", ylab = "Residuals")
plot(longElasticNetEstimate0.75, longitudeTest - longElasticNetEstimate0.75, main = "Longitudinal Elastic Net 0.75", xlab = "Fitted Values", ylab = "Residuals")
