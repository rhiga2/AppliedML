#Part c
#input necessary libraries
library(klaR)
library(caret)

#read in data from data set
data = read.csv('pima_indians_diabetes.txt', header = FALSE)

numTrials = 10
y = as.factor(data[,9])
X = data[,-9]

trainScores = array(dim = numTrials)
testScores = array(dim = numTrials)

for(i in 1:numTrials)
{
  #split data in 80-20 test-train split
  trainIdx = createDataPartition(y, p = 0.8, list = FALSE)
  XTrain = X[trainIdx, ]
  yTrain = y[trainIdx]
  XTest = X[-trainIdx, ]
  yTest = y[-trainIdx]
  
  #train naive bayes classifier
  model = train(XTrain, yTrain, 'nb', trControl = trainControl(method = 'cv', number = 10))
  
  #predict on training and test data
  decisionTrain = predict(model, newdata = XTrain)
  decisionTest = predict(model, newdata = XTest)
  
  numCorrectTrain = decisionTrain == yTrain
  numCorrectTest = decisionTest == yTest
  trainScores[i] = sum(numCorrectTrain)/length(numCorrectTrain)
  testScores[i] = sum(numCorrectTest)/length(numCorrectTest)
}

trainAccuracy = mean(trainScores)
testAccuracy = mean(testScores)

print(paste("Train Accuracy: ", trainAccuracy))
print(paste("Test Accuracy: ", testAccuracy))