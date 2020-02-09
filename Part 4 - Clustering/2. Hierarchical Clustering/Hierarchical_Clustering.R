#Hierarchical clustering

# Importing the dataset
dataset = read.csv('Mall_Customers.csv')
x<-dataset[,4:5]

#using the dendogram to find the optmal number of clusters
dendogram=hclust(dist(x,method = 'euclidean'),method = 'ward.D')
plot(dendogram,
     main=paste('Dendogram'),
     xlab = 'Customers',
     ylab='Euclidean Distance')

#Fitting the herarchical clustering to the dataset
hc=hclust(dist(x,method = 'euclidean'),method = 'ward.D')
y_hc=cutree(hc,5)

#Visualizing the clusters
library(cluster)
clusplot(x,
         y_hc,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         labels=2,
         plotchar = FALSE,
         span = TRUE,
         main=paste('Cluster of Clients'),
         xlab='Anual Income',
         ylab = 'Spending Score')
