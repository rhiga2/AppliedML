#Part 1
#input necessary libraries
library(caret)

#read csv
data = read.csv("pima_indians_diabetes.txt", header = FALSE, sep = ",")

#divide into 80-20 train-test split.
y = data[,9]
X = data[,-9]
hist(X[,1], main = "Histogram of Number of Times Pregnant", xlab = "Pregnancy Count", xlim = c(0,15))

