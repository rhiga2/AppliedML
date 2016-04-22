#Part b
#input necessary libraries
library(caret)

#read csv
data = read.csv("pima_indians_diabetes.txt", header = FALSE, sep = ",")
y = data[,9]
X = data[,-9]

#mark missing values as NA
for (j in c(3, 4, 6, 8))
{
  i<-X[, j]==0
  X[i, j]=NA
}

numTrials = 10
trainScores = array(dim = numTrials)
testScores = array(dim = numTrials)

for (i in 1:numTrials) {
  #Split Data Into Test-Train Split
  trainIdx = createDataPartition(y, p = 0.8, list= FALSE)
  XTrain = X[trainIdx, ]
  yTrain = y[trainIdx]
  XTest = X[-trainIdx, ]
  yTest = y[-trainIdx]
  
  #train Bayes Classifier
  trainPosFlags = yTrain > 0
  XPosTrain = XTrain[trainPosFlags,] # positive training features
  XNegTrain = XTrain[!trainPosFlags,] #negative training features
  posMeans = sapply(XPosTrain, mean, na.rm = TRUE)
  negMeans = sapply(XNegTrain, mean, na.rm = TRUE)
  posSd = sapply(XPosTrain, sd, na.rm = TRUE)
  negSd = sapply(XNegTrain, sd, na.rm = TRUE)
  
  #test normal Naive Bayes
  
  
  #test on training set
  posTrainOffsets = t(t(XTrain)-posMeans)
  posTrainScaled = t(t(posTrainOffsets)/posSd)
  posTrainLogs = -(1/2)*rowSums(apply(posTrainScaled, c(1,2), function(x)x^2), na.rm = TRUE) - sum(log(posSd))
  
  negTrainOffsets = t(t(XTrain)-negMeans)
  negTrainScaled = t(t(negTrainOffsets)/negSd)
  negTrainLogs = -(1/2)*rowSums(apply(negTrainScaled, c(1,2), function(x)x^2), na.rm = TRUE) - sum(log(negSd))
  
  trainDecision = posTrainLogs > negTrainLogs
  numTrainCorrect = trainDecision == yTrain
  trainScores[i] = sum(numTrainCorrect) / length(numTrainCorrect)
  
  #test on test set
  posTestOffsets = t(t(XTest)-posMeans)
  posTestScaled = t(t(posTestOffsets)/posSd)
  posTestLogs = -(1/2)*rowSums(apply(posTestScaled, c(1,2), function(x)x^2), na.rm = TRUE) - sum(log(posSd))
  negTestOffsets = t(t(XTest)-negMeans)
  negTestScaled = t(t(negTestOffsets)/negSd)
  negTestLogs = -(1/2)*rowSums(apply(negTestScaled, c(1,2), function(x)x^2), na.rm = TRUE) - sum(log(negSd))
  
  testDecision = posTestLogs > negTestLogs
  numTestCorrect = testDecision == yTest
  testScores[i] = sum(numTestCorrect)/length(numTestCorrect)
}

trainAccuracy = mean(trainScores)
testAccuracy = mean(testScores)

print(paste("Train Accuracy: ", trainAccuracy))
print(paste("Test Accuracy: ", testAccuracy))