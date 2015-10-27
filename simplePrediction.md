#' ---
#' title: "Simple model to predict hand-writing using R neuralnet library"
#' author: "Kean Loon Lee"
#' date: "Oct 10th, 2015"
#' ---
#'   
#' Predict NIST hand-writing data using R neural network package: neuralnet
#'   
library(neuralnet) # for neural network
library(ggplot2) # for plotting
library(Rmisc)
library(reshape2) # for data frame manipulation

#'  
#'  Read in test and train data provided by Kaggle
test <- read.csv("test.csv")
train <- read.csv("train.csv")

#' Define a function that randomly split a dataframe
#' into ratio:(1-ratio) for training set and cross validation set
splitdf <- function(dataframe, ratio, seed=NULL) {
  if (!is.null(seed)) set.seed(seed)
  n <- nrow(dataframe)
  trainindex <- sample(1:n, n*ratio)
  trainset <- dataframe[trainindex, ] # training set
  cvset <- dataframe[-trainindex, ] # cross-validation set
  list(trainset=trainset,cvset=cvset)
}

#' Split our training data into 80:20 training and cross validation set
df <- splitdf(train,0.8,81)
trainset <- df$trainset
cvset <- df$cvset
rm(df) # no use with df now

#' Visualise the first 9 rows of cross-validation set.
plots <- list() # new empty list
for (i in 1:9){
  z1 <- melt(cvset[i,1:785],id.vars = "label")
  y <- rep(28:1,each=28)
  x <- rep(1:28,times=28)
  z <- cbind(x,y,z1)
  
  # add each plot to the plot list
  plots[[i]] <- ggplot(z) + geom_raster(aes(x=x, y=y, fill=value)) + 
                guides(fill=FALSE) + # legend is redundant
                scale_x_continuous(limits=c(1, 28)) +
                scale_y_continuous(limits=c(1, 28)) +    
                scale_fill_continuous(low="#000000", high="#ffffff") +
    # remove the background,ticks,labels, etc to make the plots look nicer
    theme(axis.line=element_blank(),axis.text.x=element_blank(),
          axis.text.y=element_blank(),axis.ticks=element_blank(),
          axis.title.x=element_blank(),
          axis.title.y=element_blank(),legend.position="none",
          panel.background=element_blank(),panel.border=element_blank(),panel.grid.major=element_blank(),
          panel.grid.minor=element_blank(),plot.background=element_blank())    
}
rm(x,y,z,z1) # remove redundant variables
multiplot(plotlist = plots, cols = 3)

#' Check if the labels agree with the plotted images.
cvset[1:9,1]

#' Convert label 0 into 10 for easier manupulation later.
trainset[trainset$label==0,"label"] = 10

#' Normalise the data set (scale the input to manageable range for neuralnet library).
#' This step is actually very important.
#' I have spent several days to figure out why the neuralnet library does not produce
#' results until I normalised the input grey scale values.
trainset[,2:785] <- trainset[,2:785]/255.
cvset[,2:785] <- cvset[,2:785]/255.
test <- test/255.

#' Define a function to convert digit label into logical matrix
convertLabel <- function(w){
                  a <-rep(0,10)
                  a[w] <- 1
                  a
}

#' Add the converted logical array to the train data
m <- as.data.frame(t(sapply(trainset$label,convertLabel)))
names(m) <- paste('d',1:10,sep='')
trainset <- cbind(m,trainset)
rm(m)

#' Train the neural network. We are going to have 100 nodes with a single hidden layer.
#' Threshold is a numeric value specifying the threshold for the partial
#' derivatives of the error function as stopping criteria.

#' formula for the neural network
netformula <- paste(paste(paste('d',1:10,sep=''),collapse='+'),' ~ ',
                    paste(paste('pixel',0:783,sep=''), collapse='+'), sep='')

#' Choose the number of training examples to plot learning curve
ntrain = 1000

#' Train the network
net <- neuralnet(netformula,trainset[1:ntrain,], hidden=c(100),err.fct="ce",
                 linear.output = FALSE,threshold=0.01,lifesign='minimal')

#' Output tells us the train error (cross-entropy) multiplies the number of test examples, 
#' and the time taken to train the neural network.

#' Compute predictions on our cross-validatation set
results <- compute(net,cvset[,2:785])
myPredict <- apply(results$net.result,1,which.max)

#' Accuracy of our prediction
myPredict[myPredict==10] <- 0 # convert 10 back into 0
acc_cv <- sum(myPredict == cvset$label)/nrow(cvset) 
acc_cv

#' We can repeat the neural network for different number of training examples
#' and number of nodes in the hidden layer.
#' However, we always get 0 error in using the trained parameters
#' on the training set. I believe that overfitting is present here.
#' A remedy would be to implement regularisation, which is not available
#' in this R package.
#'
#' Summary of the number of nodes and layers that I have tried.
combine.res <- read.csv("neuralnetPrediction.csv")

#' Cross-validation accuracy 
p1 <-  ggplot(combine.res,aes(x=num_train_ex,y=1.-prediction_accuracy,colour=nodes)) + 
  geom_line(stat="identity") + geom_point(fill="white") +
  labs(x="Num. of trainig sets") + labs(y="Cross-validation error")
print(p1) 

#' Computing time on a single processor (2.9GHz Intel Core i7)
p2 <-  ggplot(combine.res,aes(x=num_train_ex,y=time,colour=nodes)) + 
  geom_line(stat="identity") + geom_point(fill="white") +
  labs(x="Num. of trainig sets") + labs(y="Computing time [min]")
print(p2) 

#' Finally, compute predictions on Kaggle test set.
#' Kaggle returns 0.958 accuracy for 400 nodes, single-hidden-layer, 
#' 33600 training examples.
#' This is close to the cross-validation result (0.955), as expected.
results <- compute(net,test)
myPredict <- apply(results$net.result,1,which.max)
myPredict[myPredict==10] <- 0 # convert 10 back into 0
kaggle_data <-data.frame(ImageId=1:nrow(test),Label=myPredict)
write.csv(kaggle_data,file="kaggle_result.csv",row.names=FALSE)

