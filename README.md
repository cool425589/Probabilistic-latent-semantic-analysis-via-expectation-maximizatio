1. About
2. basic function
3. Quick start
4. Contacts
---------------

1. About
----------

In this project, you will have

16 Short Queries

2265 Documents

A Background Language Model(BGLM.txt)

A Set of Documents for Topic Model Training (Collection.txt)

Our goal is to implement the PLSA model, and incorporate the PLSA and query likelihood measure for retrieval

Thus, the ultimate goal is to enhance the estimation of each document language model

2. basic function
----------------------

Here are the basic function for this web:

Initial : 

Initial matrix with information for expectation-maximization needed

OpenFile:

Open and Read Query、Document、trainning data(collection)、BGLM

Rank:

Get score with each document for each Query

Check:

Check normalization after update the matrix

Check Max likelihood if its be smaller after expectation-maximization

Denormalization:

Update Denormalization probability after expectation-maximization

Normalization:

Normalization probability after update Denormalization probability

EMStep:

Call Denormalization、Normalization,and so on to implement expectation-maximization


3. Quick start
--------------------------

Choose PLSA_Space exchange time.py(need 7GB memery but trainning faster upper 20 times) or PLSA_time exchange Space.py(300MB memory) you wanted
    
Download data you need
  
Edit Topic size、a、b、number for em times you wanted
     
Run the p.y
     
Get the result for our query

4. Contacts
----------------
You can send your suggestions and send them directly to cool425589@gmail.com.
