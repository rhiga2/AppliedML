#Homework 2
library(caret)

#read in data
data = read.csv("adult.data.txt", header = FALSE)
X = as.matrix(data[,-c(2, 4, 6, 7, 8, 9, 10, 14, 15)])
means = apply(X, 2, mean)
sds = apply(X, 2, sd)
y = data[,15]
for (i in 1:nrow(X)){
  X[i, ] = (X[i, ] - means)/sds  
}
y = as.vector(y, mode = 'numeric')
y = 2*y - 3

#split into 10% test, 10% cross-validation, and 80% train
trainIdx = createDataPartition(y, times = 1, p = 0.8, list = FALSE)
XTrain = X[trainIdx, ]
yTrain = y[trainIdx]
XValidation = X[-trainIdx, ]
yValidation = y[-trainIdx]
testIdx = createDataPartition(yValidation, times = 1, p = 0.5, list = FALSE)
XTest = XValidation[testIdx, ]
yTest = yValidation[testIdx]
XValidation = XValidation[-testIdx,]
yValidation = yValidation[-testIdx]

#plot and train for different values of lambda
lambdas = c(1e-3, 1e-2, 1e-1,1)
colors = c("red", "green", "blue", "purple")
aArray = list()
bArray = list()

plot.new()
plot( 0, 0, xlab = "Steps", ylab = "Accuracy", main = "Accuracy for Different Regularization Parameters", xlim = c(0, 15000), ylim = c(0,1), type = 'n')
for (i in 1:4){
  result = train(XTrain, yTrain, lambdas[i])
  accuracies = as.numeric(unlist(result[4]))
  steps = as.numeric(unlist(result[3]))
  lines(steps, accuracies, "l", col = colors[i])
  a = as.numeric(unlist(result[1]))
  b = as.numeric(unlist(result[2]))
  aArray[[i]] = a
  bArray[[i]] = b
  accuracyTrain = test(XTrain, yTrain, a, b)
  accuracyValidation = test(XValidation, yValidation, a, b)
  print(paste("Training Accuracy for Lambda =", lambdas[i], ": ", accuracyTrain))
  print(paste("Validation Accuracy for Lambda =", lambdas[i], ": ", accuracyValidation))
}
legend("bottomright", legend = c("1e-3", "1e-2", "1e-1", "1"), col = colors, title = "Regularization Parameters", lwd = 1)

#Test with the best regularization constant
a = as.numeric(unlist(aArray[1]))
b = as.numeric(unlist(bArray[1]))
accuracyTest = test(XTest, yTest, a, b)
print(paste("Test Accuracy : ", accuracyTest))
