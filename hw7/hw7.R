library(glmnet)

# Gaussian kernel function
# Inputs: Matrix X, vector b for base point, scalar h for scale.
# Returns: Vector of data applied with kernel function.
kernel = function(X, b, h) {
  sqdist = colSums((t(X) - b)^2)
  return(exp(-sqdist/(2*h^2)))
}

#read in base point files
bases = read.csv("Locations.txt", sep = " ")
bases$East_UTM = bases$East_UTM/1000.0
bases$North_UTM = bases$North_UTM/1000.0
temperature = read.csv("Oregon_Met_Data.txt", sep = " ")

#clean temperature data
temp = matrix(0, nrow = 112, ncol = 1)
for (i in 1:112) {
  baseArray = c(0,0,0,0,0);
  for (j in 1:5) {
    year = j + 1999 #2000-2004
    tempVector = temperature$Tmin_deg_C[temperature$SID == i & temperature$Year == year]
    if (length(tempVector) == 0) { # no data available for that base point during that year
      baseArray[j] = NA
    }
    else {
      tempVector[tempVector == 9999] = NA
      baseArray[j] = sum(tempVector, na.rm = TRUE)/length(tempVector[!is.na(tempVector)])
    }
  }
  temp[i] = mean(baseArray, na.rm = TRUE)
}

#Split dataset
basesMat =cbind(bases$East_UTM, bases$North_UTM) 
distMat = as.matrix(dist(basesMat, method = "euclidean", diag = FALSE, upper = FALSE))
n = nrow(distMat)
split = 0.7
trainIdx = sample(1:n, round(n*split))
trMat = distMat[trainIdx, ] #training data
tsMat = distMat[-trainIdx, ] #testing data
trTemp = temp[trainIdx] #training labels
tsTemp = temp[-trainIdx] #testing labels

# Construct the gram matrix for the training data and perform cross-validation
# ENABLE CROSS-VALIDATION BY UNCOMMENTING THE FOLLOWING LINE AND COMMENTING THE NEXT LINE. 
# IT TAKES SOME TIME TO TRAIN
#h = c(1:9 %o% 10^(-2:3))
h = c(40)
m = length(h)
errors = matrix(0, ncol = m)

for (i in 1:m) {
  trGMat = exp(-trMat^2/(2*h[i]^2))
  tsGMat = exp(-tsMat^2/(2*h[i]^2))

  #Train models and evaluate
  tmodel = cv.glmnet(trGMat, trTemp, alpha = 0.50) #temperature model
  tEst = predict(tmodel, tsGMat, s = "lambda.min") #unregularized predictions
  errors[i] = sum((tsTemp-tEst)^2)/length(tsTemp)
}
plot(log(h), log(errors))
scale = h[which.min(errors)]
lines(c(log(scale), log(scale)), c(-5, max(log(errors)+10)))

#make new model with lowest errors. Test Lasso and ElasticNet Techniques
newTrGramMat = exp(-trMat^2/(2*scale^2))
newTsGramMat = exp(-tsMat^2/(2*scale^2))
trTemp = temp[trainIdx] #training labels
tsTemp = temp[-trainIdx] #testing labels

#CHANGE GLMNET PARAMETERS HERE
tempModel = cv.glmnet(newTrGramMat, trTemp, alpha = 0.50) #temperature model
tempEst = predict(tempModel, newTsGramMat, s = "lambda.min") #unregularized
mse = sum((tempEst-tsTemp)^2)/length(tsTemp)
  
#X is the 100x100 grid.
X = matrix(0, ncol = 2, nrow = 10000)
east = seq(min(bases$East_UTM), max(bases$East_UTM), length=100) #East UTM
north = seq(min(bases$North_UTM), max(bases$North_UTM), length = 100) #North UTM

for (i in 1:100) {
  for (j in 1:100) {
    X[(i-1)*100+j,1] = east[i]
    X[(i-1)*100+j,2] = north[j]
  } 
}

gridGram = matrix(0, ncol = 112, nrow = 10000)
#construct the gram matrix
for (i in 1:112) {
  gridGram[,i] = kernel(X, c(bases$East_UTM[i], bases$North_UTM[i]), h[1])
}

#CHANGE PREDICTION PARAMETERS HERE
tempPred = predict(tempModel, newx = gridGram, s = "lambda.min")
wscale = max(abs(min(tempGrid)), abs(max(tempGrid)))
gridTemp = matrix(tempPred, nrow = 100, ncol = 100) 
image(east, north, t(gridTemp), xlab = "East", 
      ylab = "North", col = rev(rainbow(40, alpha =1, start = 0, end = 0.8)))
points(basesMat, pch = 1)
contour(east, north, t(gridTemp),col = "black", levels = -seq(-20, 20),  add = TRUE)
title("Temperature Map")