library(klaR)
library(caret)

#read in data
data = read.csv("data.txt", header = FALSE)
X = data[,-9]
y = as.factor(data[,9])

numTrials = 10
trainScores = array(dim = numTrials)
testScores = array(dim = numTrials)

for(i in 1:numTrials)
{
  #create 80-20 train-test split 
  trainIdx = createDataPartition(y, p = 0.8, list = FALSE)
  XTrain = X[trainIdx,]
  yTrain = y[trainIdx]
  XTest = X[-trainIdx,]
  yTest = y[-trainIdx]
  
  #train svm using training data only
  svm = svmlight(XTrain, yTrain, pathsvm = 'svm_light')
  
  #test on training set
  labelsTrain = predict(svm, XTrain)
  decisionsTrain = labelsTrain$class
  numCorrectTrain = decisionsTrain == yTrain
  trainScores[i] = sum(numCorrectTrain)/length(numCorrectTrain)
  
  #test on test set
  labelsTest = predict(svm, XTest)
  decisionsTest = labelsTest$class
  numCorrectTest = decisionsTest == yTest
  testScores[i] = sum(numCorrectTest)/length(numCorrectTest)
}

trainAccuracy = mean(trainScores)
testAccuracy = mean(testScores)

print(paste("Train Accuracy: ", trainAccuracy))
print(paste("Test Accuracy: ", testAccuracy))
