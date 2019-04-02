import os
import numpy as np
import math
#Get the logsumexp in scipy.special
from scipy.special import logsumexp
import time
from tempfile import TemporaryFile

#Probability of word and topic
w_given_T = []
#Probability of Topic and document
T_given_d = []
#Save the preview probability of word and topic
temp_w_given_T = []
#Save the preview probability of Topic and document
temp_T_given_d = []

#The count of words in document
count_word_document = []
#term word document
term_word_document = []

#log sum
log_sum = []

#Word length
word_length = 51253
#The row in collection
document_size = 18461
#Topic size
topic_size = 16
#Document number
document_number = 2265

BGLM_idf = []

#likelihood
L = -1000000000000
LZERO = -1.0E10

a = 0.3
b = 0.6

BGLM_idf = []
"""
path ='C:/'
basepath = os.path.join(os.path.sep, path ,'Users','M10615079','Desktop','Homework4')
os.chdir(basepath)
collection_path = os.path.join(os.path.sep,os.getcwd(),'Collection.txt')
BGLM_path = os.path.join(os.path.sep,os.getcwd(),'BGLM.txt')
query_path = os.path.join(os.path.sep,os.getcwd(),'Query')
document_path = os.path.join(os.path.sep,os.getcwd(),'Document')
"""
collection_path = 'Homework3/Collection.txt'
BGLM_path = 'Homework3/BGLM.txt'
query_path = 'Homework3/Query'
document_path = 'Homework3/Document'

queryname = []
filename = []

"""
==============================================Initial======================================
"""
def random_w_given_T():
    global w_given_T
    global temp_w_given_T

    w_given_T = np.random.rand(word_length, topic_size)
    temp_w_given_T = np.zeros((word_length, topic_size))
def log_w_given_T():
    global w_given_T
    w_given_T = np.log(w_given_T)  
def random_T_given_d():
    global T_given_d
    global temp_T_given_d

    T_given_d = np.random.rand(topic_size, document_size)
    temp_T_given_d = np.zeros((topic_size, document_size))
def log_T_given_d():
    global T_given_d
    T_given_d = np.log(T_given_d)
def init_w_given_T_normalization():
    global w_given_T 
    w_given_T = w_given_T / w_given_T.sum(axis = 0)
def init_T_given_d_normalization():
    global T_given_d
    T_given_d = T_given_d / T_given_d.sum(axis = 0)  
"""
==============================================OpenFile======================================
"""
def open_collection_file():

    global count_word_document
    global term_word_document
    global log_sum

    with open(collection_path) as f:
        temp_file = f.readlines()

        for line_file in temp_file:
            line_count = []
            line_word = []
            line_file_split = line_file.split()
            for word in line_file_split:
                if word not in line_word:
                    line_count.append(np.log(line_file_split.count(word)))
                    line_word.append(word)
            count_word_document.append(line_count)
            term_word_document.append(line_word)
            log_sum.append(np.log(len(line_file_split)))
def open_Document_file():

    global filename
    global count_word_document
    global term_word_document
    global log_sum
    log_sum = []
    count_word_document = []
    term_word_document = []

    for doc_open in os.listdir(document_path):
        filename.append(doc_open)
    for now_file_name in filename:
        line_count = []
        line_word = []
        with open(os.path.join(document_path, now_file_name)) as f:
            f.readline()
            f.readline()
            f.readline()
            file_read = f.read()
            file_split = file_read.split()
            for word in file_split:
                if word != '-1':
                    if word not in line_word:
                        line_count.append(np.log(file_split.count(word)))
                        line_word.append(word)
            count_word_document.append(line_count)
            term_word_document.append(line_word)
            log_sum.append( logsumexp(line_count) )
def read_query():
    global queryname

    for query_open in os.listdir(query_path):
        queryname.append(query_open)

    for now_query_name in queryname:
        #print (now_query_name)
        writefile.write("\n"+now_query_name+",")
        query_array = []
        with open(os.path.join(query_path, now_query_name)) as q:
            q = q.read()
            q = q.split()
            for query_term in q:
                if query_term != '-1':
                    query_array.append(int(query_term))
            rank(query_array)
def read_BGLM():
    global BGLM_idf
    BGLM_idf = np.zeros(word_length)
    lines = 0
    with open(BGLM_path) as f:
        BGLM_file = f.readlines()

        for line in BGLM_file:
            BGLM_idf[lines] = line.split()[1]
            lines = lines + 1
"""
==============================================Rank======================================
"""
def grade(w_i, d_j):
    if str(w_i) not in term_word_document[d_j]:
        return a * (  np.exp( LZERO - log_sum[d_j] ) ) + b * ( np.exp( logsumexp(w_given_T[w_i, :] + T_given_d[:, d_j]) ) ) + (1-a-b) * np.exp(BGLM_idf[w_i]) 
    else:
        return a * (  np.exp( count_word_document[d_j][term_word_document[d_j].index(str(w_i))] - log_sum[d_j] ) ) + b * ( np.exp( logsumexp(w_given_T[w_i, :] + T_given_d[:, d_j]) ) ) + (1-a-b) * np.exp(BGLM_idf[w_i]) 
def get_d_j_grade(w_iarray, d_j):
    ans = 1
    for loopw_i in w_iarray:
        ans = ans * grade(loopw_i, d_j)
    return ans
def get_grade(query_words):
    d_j_grade = np.zeros(T_given_d.shape[1])
    for d_j in range(T_given_d.shape[1]):
        d_j_grade[d_j] = get_d_j_grade(query_words, d_j)
    return d_j_grade
def rank(query_words):
    temp = get_grade(query_words)
    temp = -temp   
    rankresult = np.argsort(temp)
    for file_index in rankresult:
        #print (filename[file_index])
        writefile.write(filename[file_index]+" ")

"""
==============================================Check======================================
"""
def check_likelihood():
    global L
    new_L = 0
    
    for d_j in range(T_given_d.shape[1]):
        new_L = new_L + temp_get_likelihood(d_j)
    print (" L = " + str(L)+" , new_L = " + str(new_L))
    assert new_L >= L, "New_likelihood be smaller!  L = " + str(L)+" , new_L = " + str(new_L)
    L = new_L
def check_normalization(normalization_matrix):
    num = 0
    for check in np.exp(normalization_matrix).sum(axis = 0):
        assert math.isclose(1, check, rel_tol=1e-1), " normalization_matrix shape : "+ str(normalization_matrix.shape[0] ) +","+str(normalization_matrix.shape[1] ) + "num : " + str(num) + " check : "+str(check)
        num+=1  
def temp_get_likelihood(d_j):
    d_j_array = np.zeros(w_given_T.shape[0])
    for index in term_word_document[d_j]:
        d_j_array[int(index)] = np.exp(count_word_document[d_j][term_word_document[d_j].index(index)])
    return np.sum(d_j_array*logsumexp(temp_w_given_T + temp_T_given_d[:,d_j], axis = 1))
"""
==============================================Denormalization======================================
"""
def w_given_ALL_T(w_i):
    global temp_w_given_T_dic
    up = (T_given_d + w_given_T[w_i, :,np.newaxis])
    down = logsumexp( T_given_d + w_given_T[w_i, :, np.newaxis] , axis = 0)
    
    w_i_array = np.zeros(T_given_d.shape[1])
    for d_j in range(T_given_d.shape[1]):
        if str(w_i) not in term_word_document[d_j]:
            w_i_array[d_j] = LZERO
        else:
            w_i_array[d_j] = count_word_document[d_j][term_word_document[d_j].index(str(w_i))]
    temp_w_given_T[w_i, :] = logsumexp( w_i_array + (up - down), axis = 1 )
def ALL_T_given_d(d_j):
    global temp_T_given_d
    up = (w_given_T + T_given_d[:, d_j]) 
    down = logsumexp( w_given_T + T_given_d[:, d_j] , axis = 1)
    ex = up - down[:,np.newaxis]
    
    d_j_array = np.zeros(w_given_T.shape[0])
    d_j_array[d_j_array==0] = LZERO
    for index in term_word_document[d_j]:
        d_j_array[int(index)] = count_word_document[d_j][term_word_document[d_j].index(index)]
        
    temp_T_given_d[:,d_j] = logsumexp( (d_j_array[:, np.newaxis] + ex), axis = 0 )
def denormalization_Max_w_T_func():
    def update_Max_w_T(w_i) :
        w_given_ALL_T(w_i)
    return np.frompyfunc(update_Max_w_T, 1, 0)
def denormalization_Max_T_d_func():
    def update_Max_T_d(d_j) :
        ALL_T_given_d(d_j)
    return np.frompyfunc(update_Max_T_d, 1, 0)
"""
==============================================Normalization======================================
"""
def normalization_w_T_func():
    def update_normalization_w_T(T_k):
        global temp_w_given_T
        w_given_T_sum = np.logaddexp.reduce(temp_w_given_T[:, T_k], dtype=np.float64)
        temp_w_given_T[:,T_k] = temp_w_given_T[:,T_k] - w_given_T_sum
    return np.frompyfunc(update_normalization_w_T, 1, 0)
def normalization_T_d_func():
    def update_normalization_T_d(d_j):
        global temp_T_given_d        
        temp_T_given_d[:,d_j] = temp_T_given_d[:,d_j] - log_sum[d_j]
    return np.frompyfunc(update_normalization_T_d, 1, 0)
"""
==============================================EMStep======================================
"""
def EM_times(times):
    for t in range(times):
        print ("EM_step"+str(t))
        global w_given_T
        global T_given_d
        start_time = time.time()
        denormalization_Max_w_T_func()(np.arange(w_given_T.shape[0]))
        print ("denormalization_Max_w_T_func--- %s seconds ---" % (time.time() - start_time))
        start_time = time.time()
        denormalization_Max_T_d_func()(np.arange(T_given_d.shape[1]))
        print ("denormalization_Max_T_d_func--- %s seconds ---" % (time.time() - start_time))
        start_time = time.time()
        normalization_w_T_func()(np.arange(topic_size))
        print ("normalization_w_T_func--- %s seconds ---" % (time.time() - start_time))
        start_time = time.time()
        normalization_T_d_func()(np.arange(T_given_d.shape[1]))
        print ("normalization_T_d_func--- %s seconds ---" % (time.time() - start_time))
        start_time = time.time()
        
        check_normalization(temp_w_given_T)
        print ("check_normalization--- %s seconds ---" % (time.time() - start_time))
        start_time = time.time()
        check_normalization(temp_T_given_d)
        print ("check_normalization--- %s seconds ---" % (time.time() - start_time))
        start_time = time.time()
        check_likelihood()
        print ("check_likelihood--- %s seconds ---" % (time.time() - start_time))
        
        w_given_T = temp_w_given_T.copy()
        T_given_d = temp_T_given_d.copy()
"""
==============================================Main======================================
"""
def main():
    global document_size
    global L
    print ("open_collection_file:")
    open_collection_file()
    
    print ("random_w_given_T:")
    random_w_given_T()
    print ("init_w_given_T_normalization:")
    init_w_given_T_normalization()
    print ("log_w_given_T:")
    log_w_given_T()
    print ("check_normalization:")
    check_normalization(w_given_T)
    
    print ("random_T_given_d:")
    random_T_given_d()
    print ("init_T_given_d_normalization:")
    init_T_given_d_normalization()
    print ("log_T_given_d:")
    log_T_given_d()
    print ("check_normalization:")
    check_normalization(T_given_d)
    
    EM_times(50)
    read_BGLM()
    document_size = 2265
    print ("open document:")
    open_Document_file()
    print ("renew_T_given_d:")
    random_T_given_d()
    print ("renew_T_given_d_normalization:")
    init_T_given_d_normalization()
    print ("renew_log_T_given_d:")
    log_T_given_d()
    print ("check_renew_normalization:")
    check_normalization(T_given_d)

    L = -1000000000000000
    EM_times(50)

    np.save("WGT.npy", w_given_T)
    np.save("TGD.npy", T_given_d)
    np.save("CWD.npy", count_word_document)
    np.save("TWD.npy", term_word_document)
    np.save("BGLMidf.npy", BGLM_idf)
    np.save("logsum.npy", log_sum)
    np.save("f_name.npy", queryname)
    np.save("q_name.npy", filename)

    read_query()

with open("submission.txt", "w") as writefile:
    writefile.write("Query,RetrievedDocuments")
    main()
