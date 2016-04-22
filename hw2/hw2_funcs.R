#train linear svm via stochastic gradient descent
#lambda = regularization constant
#Ne = number of epochs
#Ns = number of steps per epoch
#Vr = validation rate
train = function(XTrain, yTrain, lambda, p = 0.1, q = 50, Ne = 50, Ns = 300, Vr = 30){
  Nv = Ns/Vr #number of validations per epoch
  steps = (1:(Nv*Ne))*Vr
  accuracies = array(dim = (Nv*Ne))
  numInstances = nrow(XTrain)
  sampleNum = numInstances - 50
  numFeatures = ncol(XTrain)
  a = sample(1:5, numFeatures, replace = TRUE)
  b = sample(1:10, 1)
  
  for(e in 1:Ne){
    #separate 50 training examples at random for validation
    validateIdx = sample(1:numInstances, 50)
    XSample = XTrain[validateIdx, ]
    ySample = yTrain[validateIdx]
    XNewTrain = XTrain[-validateIdx, ]
    yNewTrain = yTrain[-validateIdx]
    
    #find learning rate/step size
    rate = 1/(p*e+q)
    
    for(s in 1:Ns){
      #pick random data instance
      i = sample(1:sampleNum, 1)
      xi = XNewTrain[i, ]
      yi = yNewTrain[i]
      
      #update via Stochastic Gradient Descent
      if(yi*((a %*% xi) + b) >= 1){
        a = a - rate*lambda*a
      } else {
        a = a-rate*(lambda*a-yi*xi)
        b = b+rate*yi
      }
      
      #validate at 30 step mark
      if (s %% 30 == 0){
        decision = sign((XSample %*% a)+b) == ySample
        accuracies[(e-1)*(Nv)+s/Vr] = sum(decision)/50
      }
    }
  }
  return(list(a, b, steps, accuracies))
}

test = function(XTest, yTest, a, b){
  decision = sign((XTest %*% a)+b) == yTest
  accuracy = sum(decision)/length(yTest)
  return(accuracy)
}