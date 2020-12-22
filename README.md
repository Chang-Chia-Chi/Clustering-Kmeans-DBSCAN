# Clustering-Kmeans-DBSCAN
To discover behavior of differenct customer and utilize these information, **clustering** is always a good start point. Most intuitive way is clustering customers based on what they have bought in past, that is, using purchase record of each customer as vectors with product as features. This project implentmented three different methods:  

1. **Kmeans** with **euclidean distance** measure, using sklearn API.
2. **Kmeans** with **jaccard distance** measure, hand-craft code.
3. **Dbscan** with **jaccard distance** measure, hand-craft code.   

The purpose of this project is to help understand fundimental clustering algorithms and think how to choose appropriate distance measure for different task.
## Dataset
![image](http://snap.stanford.edu/images/snap_logo.png) [Link](http://snap.stanford.edu/data/web-Amazon-links.html)   
The data format looks like:
```
product/productId: B00006HAXW
product/title: Rock Rhythm & Doo Wop: Greatest Early Rock
product/price: unknown
review/userId: A1RSDE90N6RSZF
review/profileName: Joseph M. Kotow
review/helpfulness: 9/9
review/score: 5.0
review/time: 1042502400
review/summary: Pittsburgh - Home of the OLDIES
review/text: I have all of the doo wop DVD's and this one is as good or better than the
1st ones. Remember once these performers are gone, we'll never get to see them again.
Rhino did an excellent job and if you like or love doo wop and Rock n Roll you'll LOVE
this DVD !!
```
We choose `Music.txt` because it contains around **~6M** reviews which is suitible for practice purpose.  
## Data preprocess
Only `productID` and `userID` is required to build feature matrix, which means data preprocessing is essential to parse information we need. To better understand this dataset, we compute some statistc property and find the matrix will be very sparse. Besides, number of users and products are large, we have to use **sparse matrix** to store information. A intermediate text output file was created for future use of sparse matrix building.   
Output format looks like, where `ProductId` means item user have bought:   
```
UserId1, ProductId1, ProductId2,...
UserId2, ProductId1, ProductId2,...
UserId3, ProductId1, ProductId2,...   
...
```   
Processed output file `music_data.txt` has been uploaded. If you want to run it from scratch, follow steps below:   
1. Download `Music.txt.gz` from snap website and put file in this folder
2. run `preprocess.py`
## Clustering
As mention above, we implement three different methods to cluster. These methods may not all suitable for this specific task, so we need some extra tools to help us identify which one is better. Two ways are used for this project:
1. Overall distance: The most simple and useful way is to compute sum of euclidean distance of each data point to its closest centroid.   

|Algorithm       |Kmeans-Euc|Dbscan-Jac|Kmeans-Jac|
|----------------|----------|----------|----------|
|Overall Distance|2231      |857       |2242      |   

2. Visualization: Visualization is another common technique to compare performance of clustering. Because there are around 550k distince products, TSNE method is used to reduce dimension of matrix to 2D.     
All visualization pictures get rid of points in largest cluster, which should be useless, to show meaningful clusters more clearly.   
**Kmeans with euclidean distance, C=10.**     
![image](https://github.com/Chang-Chia-Chi/Clustering-Kmeans-DBSCAN/blob/main/pics/Music-Kmeans-Eud-C-10.jpg)   
**Dbsacne with jaccard distance.**     
![image](https://github.com/Chang-Chia-Chi/Clustering-Kmeans-DBSCAN/blob/main/pics/Music-DBSCAN-Jaccard.jpg)    
**Kmeans with jaccard distance, C=10.**     
![image](https://github.com/Chang-Chia-Chi/Clustering-Kmeans-DBSCAN/blob/main/pics/Music-Kmeans-Jaccard-C-10.jpg)

## Summary
1. Number of distinct products most users buy is 2 if we run `preprocess.py`. This means the matrix will be very sparse that Euclidean distance is not proper choice for this problem.   
2. **Dbscan-jaccard** method has smallest overall distance and cluster visualization shows more apparent tendency comparing to other two methods, so
Dbscan with jaccard distance is better for this task.

## Reference
1. http://snap.stanford.edu/data/web-Amazon-links.html
2. https://github.com/aahoo/dbscan/blob/master/dbscan.py
3. https://www.youtube.com/watch?v=DBpTY6J3ttM&list=PLgQKp4YLJUdy5fm8ifluSl8J96RwsbHAd&index=12
