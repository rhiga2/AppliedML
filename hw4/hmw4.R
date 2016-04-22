#import necessary libraries
library(lattice)
library(plsdepot)
library(scatterplot3d)

#Run sections individually.
#Problem 3.4
######### import Iris Dataset #########
irisdat = read.csv('iris.data.txt' , header=FALSE );
numiris=irisdat[,c(1, 2, 3, 4) ]
postscript ("irisscatterplot.eps")
speciesnames = c ('setosa' , 'versicolor', 'virginica')
pchr = c( 1 , 2 , 3)
colr = c( 'red', 'green' , 'blue' , 'yellow' , 'orange' )
ss = expand.grid(species = 1:3)
parset = with ( ss , simpleTheme (pch=pchr [species], col=colr[species]))
splom (irisdat[,c(1:4)] , groups=irisdat$V5,
        par.settings = parset,
        varnames=c( 'Sepal\nLength' , 'Sepal\nWidth' ,
                    'Petal\nLength' , 'Petal\nWidth' ) ,
        key=list(text=list(speciesnames),
                      points=list(pch=pchr), columns=3))
dev.off()

########## PCA for iris data ######### 
plot(0)
iris_pca = prcomp(numiris, center = TRUE, scale = FALSE)
pca_irisdat = iris_pca$x
splom (pca_irisdat[,c(1:2)] , groups=irisdat$V5,
       par.settings = parset,
       varnames=c( 'PC 1' , 'PC 2') ,
       key=list(text=list(speciesnames),
                points=list(pch=pchr) , columns=3))
title("Principal Components of Iris Data")

########## PLS1 for iris data #########
plot(0)
iris_pls1 = plsreg1(numiris, as.numeric(irisdat$V5), comps = 2)
title("PSL1 Components for Iris Data")
splom (iris_pls1$x.scores[,c(1:2)] , groups=irisdat$V5,
       par.settings = parset,
       varnames=c( 'DC 1' , 'DC 2') ,
       key=list(text=list(speciesnames),
                points=list(pch=pchr) , columns=3))

########## Problem 3.5 #########
#import Wine Dataset
winedat = read.csv('wine.data.txt' , header=FALSE )
wine_feats = winedat[, 2:14]

#PCA for wine data
wine_pca = prcomp(wine_feats, center = TRUE, scale = FALSE)
wine_variances = wine_pca$sdev^2
plot(wine_variances, xlab = "Principal Components", ylab = "Eigenvalues")
lines(wine_variances, xlab = "Principal Components", ylab = "Eigenvalues")
title("Principal Components Eigenvalues for Wine Dataset")


########## Plot stem plot #########
pca_winedat = wine_pca$x
plot(0, xlab = "Examples", ylab = "Principal Component", xlim=c(0,13), ylim=c(-1, 1))
stem(1:nrow(wine_pca$rotation), wine_pca$rotation[,1], pointcol = "red", linecol = "red", clinecol = "red")
stem(1:nrow(wine_pca$rotation), wine_pca$rotation[,2], pointcol = "green", linecol = "green", clinecol = "green")
stem(1:nrow(wine_pca$rotation), wine_pca$rotation[,3], pointcol = "blue", linecol = "blue", clinecol = "blue")
legend(x = "topright", c("PCA1  ", "PCA2  ", "PCA3  "), col = c("red", "green", "blue"), pch = 16)
title("Stem Plot For Principal Component Analysis on Wine Data")

########## Plot projection of data onto first two principle components. #########
winedat$color = "black"
winedat$color[winedat$V1 == 1] = "red"
winedat$color[winedat$V1 == 2] = "green"
winedat$color[winedat$V1 == 3] = "blue"
plot(pca_winedat[,1], pca_winedat[,2], col = winedat$color, 
     pch = winedat$V1, xlab = "PC1", ylab = "PC2")
legend(x = "bottomright", c("1  ", "2  ", "3  "), col = c("red", "green", "blue"), pch = c(1,2,3))
title("Wine Data Projected Onto First Two Principal Components")

########## Problem 3.7 #########
#PCA for WDBC data
wdbcdat = read.csv("wdbc.data.txt", header = FALSE)
wdbc_feats = wdbcdat[,3:32]
wdbc_labels = as.numeric(wdbcdat[,2])
wdbc_pca = prcomp(wdbc_feats, center = TRUE, scale = FALSE)
pca_wdbc = wdbc_pca$x
scatterplot3d(as.numeric(pca_wdbc[,1]), as.numeric(pca_wdbc[,2]), as.numeric(pca_wdbc[,3]), 
              color = wdbc_labels, pch = wdbc_labels,
              xlab = "PC1", ylab = "PC2", zlab = "PC3")
legend(x = "topright", c("Benign  ", "Malignant  "), col = c("black", "red"), pch = c(1,2))
title("3d Scatterplot Matrix of WDBC Data on Three Principal Components")

########## PLS1 for WDBC Data #########s
wdbc_pls1 = plsreg1(wdbc_feats, wdbc_labels, comps = 3)
scatterplot3d(wdbc_pls1$x.scores[,1], wdbc_pls1$x.scores[,2], wdbc_pls1$x.scores[,3], 
              color = wdbc_labels, pch = wdbc_labels,
              xlab = "DC1", ylab = "DC2", zlab = "DC3")
legend(x = "topright", c("Benign  ", "Malignant  "), col = c("black", "red"), pch = c(1,2))
title("Three PSL1 Components for WDBC Data")