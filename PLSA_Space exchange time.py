import fnmatch
import os
import math
import numpy as np
import time
import logging
from scipy.special import logsumexp
logging.basicConfig(filename='best_answer.log',level=logging.DEBUG)
#ignore log 0 warning
np.seterr(divide='ignore')
start_time = time.time()
LZERO = -1.0E10
LSMALL = -0.5E10
minLogExp = -np.log(-LZERO)
#under log
w_given_T_dic = []
#under log
T_given_d_dic  = []
#under log
EM_w_given_T_dic = []
#under log
EM_T_given_d_dic  = []
document_name_index = []
#word dict
word_lengh = 51253
#collection size for trainning
document_size = 18461
#Topic size
Topic_size = 16
#Likelihood function 
L = -1000000000000
querydocument_size = 2265
BGLM_array = np.ndarray( (word_lengh) )
query_count_matirx = np.ndarray( (word_lengh) )
query_ranking_result = np.ndarray( (querydocument_size) )
a = 0.5
b = 0.15
#Collection path
path ='C:/'
basepath = os.path.join(os.path.sep, path ,'Users','M10615079','Desktop','topic8')
#path ='E:/'
#basepath = os.path.join(os.path.sep, path ,'M10615079','Homework3')
os.chdir(basepath)
collection_path = os.path.join(os.path.sep,os.getcwd(),'Collection.txt')
#open Collection.txt
collection = open(collection_path, 'rb')
write_result_path = os.path.join(os.path.sep,os.getcwd(),'result2.txt')
write_file = open(write_result_path, 'w', encoding = 'UTF-8')

print("Topic_size : " + str(Topic_size) )
print(basepath)
#word count in document
"""
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                                                                      initialize
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
"""
d_w_count_matirx = np.ndarray( (document_size,word_lengh) )
d_w_logsum_matirx = np.ndarray( (document_size,1) )
def init_d_w_count_matirx():
    global d_w_count_matirx
    matrix_path = os.path.join(os.path.sep,os.getcwd(),'d_w_count_matirx.npz.npy')
    if os.path.exists(matrix_path):
        print ("d_w_count_matirx.npz.npy exists now load...")
        d_w_count_matirx = np.load(matrix_path)
    else :
        d_j = 0
        for line in collection.readlines() :
            line = [int(i) for i in line.split()]  
            d_w_count_matirx[d_j,:] = np.bincount(line,  minlength = word_lengh) 
            d_j +=1
        np.save(matrix_path, d_w_count_matirx)
def init_d_w_logsum_matirx_func():
    def init_d_w_logsum_matirx(d_j):
        global d_w_logsum_matirx
        d_w_logsum_matirx[d_j] = np.log ( np.sum( d_w_count_matirx[d_j,:] ) )
    return np.frompyfunc(init_d_w_logsum_matirx, 1, 0)
def init_log_d_w_count_matirx():
    matrix_path = os.path.join(os.path.sep,os.getcwd(),'log_d_w_count_matirx.npz.npy')
    global d_w_count_matirx 
    if os.path.exists(matrix_path):
        print ("d_w_count_matirx.npz.npy exists now load...")
        d_w_count_matirx = np.load(matrix_path)
    else :
        d_w_count_matirx = np.log(d_w_count_matirx)
        d_w_count_matirx[ d_w_count_matirx == -np.inf] = LZERO
        np.save(matrix_path, d_w_count_matirx)      
def init():
    w_given_T_dic_path = os.path.join(os.path.sep,os.getcwd(),'w_given_T_dic.npz.npy')
    T_given_d_dic_dic_path = os.path.join(os.path.sep,os.getcwd(),'T_given_d_dic.npz.npy')
    global w_given_T_dic
    global T_given_d_dic
    global EM_w_given_T_dic
    global EM_T_given_d_dic
   
    if os.path.exists(w_given_T_dic_path):
        print ("w_given_T_dic.npz.npy exists now load...")
        w_given_T_dic = np.load(w_given_T_dic_path)
    else :
        w_given_T_dic  = np.random.uniform(1, 1000, (word_lengh, Topic_size) )
        #np.save(w_given_T_dic_path, w_given_T_dic) 
    
    if os.path.exists(T_given_d_dic_dic_path):
        print ("T_given_d_dic.npz.npy exists now load...")
        T_given_d_dic = np.load(T_given_d_dic_dic_path)
    else :
        T_given_d_dic = np.random.uniform(1, 1000, (Topic_size, document_size) ) 
        #np.save(T_given_d_dic_dic_path, T_given_d_dic)      

    EM_w_given_T_dic  = np.zeros((word_lengh, Topic_size))
    EM_T_given_d_dic = np.zeros((Topic_size, document_size))
def init_normalization():
    global w_given_T_dic
    global T_given_d_dic
    w_given_T_dic = w_given_T_dic / w_given_T_dic.sum(axis = 0)
    T_given_d_dic = T_given_d_dic / T_given_d_dic.sum(axis = 0)
def init_log():
    global w_given_T_dic
    global T_given_d_dic
    w_given_T_dic = np.log( w_given_T_dic )
    T_given_d_dic = np.log( T_given_d_dic )
"""
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                                                                      EM step
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
"""
def w_given_ALL_T(w_i):
    global EM_w_given_T_dic
    up = (T_given_d_dic + w_given_T_dic[w_i, :,np.newaxis])
    down = logsumexp( T_given_d_dic + w_given_T_dic[w_i, :, np.newaxis] , axis = 0)
    EM_w_given_T_dic[w_i, :] = logsumexp( d_w_count_matirx[:, w_i] + (up - down  ), axis = 1 )  
def ALL_T_given_d(d_j):
    global EM_T_given_d_dic
    up = (w_given_T_dic + T_given_d_dic[:, d_j]) 
    down = logsumexp( w_given_T_dic + T_given_d_dic[:, d_j] , axis = 1)
    ex = up - down[:,np.newaxis]
    EM_T_given_d_dic[:,d_j] = logsumexp( (d_w_count_matirx[d_j, :, np.newaxis] + ex), axis = 0 )
def denormalization_Max_w_T_func():
    def update_Max_w_T(w_i) :
        w_given_ALL_T(w_i)
    return np.frompyfunc(update_Max_w_T, 1, 0)
def denormalization_Max_T_d_func():
    def update_Max_T_d(d_j) :
        ALL_T_given_d(d_j)
    return np.frompyfunc(update_Max_T_d, 1, 0)
"""
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                                                                normalization 
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
"""
def normalization_w_T_func():
    def update_normalization_w_T(T_k):
        global EM_w_given_T_dic
        w_given_T_sum = np.logaddexp.reduce(EM_w_given_T_dic[:, T_k], dtype=np.float64)
        EM_w_given_T_dic[:,T_k] = EM_w_given_T_dic[:,T_k] - w_given_T_sum
    return np.frompyfunc(update_normalization_w_T, 1, 0)
def normalization_T_d_func():
    def update_normalization_T_d(d_j):
        global EM_T_given_d_dic
        global d_w_logsum_matirx
        EM_T_given_d_dic[:,d_j] = EM_T_given_d_dic[:,d_j] - d_w_logsum_matirx[d_j]

    return np.frompyfunc(update_normalization_T_d, 1, 0)
"""
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                                                                      check step
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
"""
def check_init_normalization():
    for T_k in range(Topic_size): 
        sum = 0
        for w_i in range(word_lengh):
            sum += w_given_T_dic[w_i,T_k]
        assert math.isclose(sum, 1.0, rel_tol=1e-5) , "P(w "+"|T_"+str(T_k)+") error sum =" + str(sum) 
    for d_j in range(document_size): 
        sum = 0
        for T_k in range(Topic_size): 
            sum += T_given_d_dic[T_k, d_j]
        assert math.isclose(sum, 1.0, rel_tol=1e-5) , "P(T "+"|d_"+str(d_j)+") error sum = " + str(sum) 
def check_w_T_normalization_func():
    def check_w_T_normalization(T_k):
        global EM_w_given_T_dic
        sum = np.sum ( np.exp( EM_w_given_T_dic[:,T_k] ) )
        #print ("check_w_T_normalization for w_given_T_dic => "+ "P(w "+"|T_"+str(T_k)+")  sum =" + str(sum) )
        assert math.isclose(sum, 1.0, rel_tol=1e-1) , "P(w "+"|T_"+str(T_k)+") error sum =" + str(sum)
    return np.frompyfunc(check_w_T_normalization, 1, 0)
def check_T_d_normalization_func():
    def check_T_d_normalization(d_j):
        global EM_T_given_d_dic
        sum = np.sum ( np.exp( EM_T_given_d_dic[:, d_j] ) )
        #print ("check_T_d_normalization_func for T_given_d_dic => "+ "P(T "+"|d_"+str(d_j)+") sum =" + str(sum) )
        assert math.isclose(sum, 1.0, rel_tol=1e-1) , "P(T "+"|d_"+str(d_j)+") error sum =" + str(sum)       
    return np.frompyfunc(check_T_d_normalization, 1, 0)    
def Likelihood_func():
    def get_Likelihood(d_j) :
        return np.sum( np.exp(d_w_count_matirx[d_j, :]) * logsumexp( EM_w_given_T_dic + EM_T_given_d_dic[:, d_j] , axis = 1)  )
    return np.frompyfunc(get_Likelihood, 1, 1)
def check_Likelihood():
    global L
    new_L = np.sum( Likelihood_func()( np.arange(T_given_d_dic.shape[1]) ).astype(np.float64) )
    print (" check_Likelihood  => L : " + str(L) +"  new L : " + str(new_L) )
    assert new_L >= L , "error Likelihood be smaller!  L = " + str(L)+" , new_L = " + str(new_L)    
    L = new_L 
def EM_step(step_num):
    for step in range(step_num):
        global w_given_T_dic
        global T_given_d_dic
        TOTOAL_start_time = time.time()
        print ("Start denormalization_Max_w_T  EM : " + str(step) )
        start_time = time.time()
        denormalization_Max_w_T_func()(np.arange(word_lengh))     
        print("End denormalization_Max_w_T --- %s seconds ---" % round((time.time() - start_time),1))
        print ("Start denormalization_Max_T_d  EM : " + str(step))
        start_time = time.time()     
        denormalization_Max_T_d_func()(np.arange(T_given_d_dic.shape[1]))
        print("End denormalization_Max_T_d --- %s seconds ---" % round((time.time() - start_time),1))             
        print ("Start normalization_w_T  EM : " + str(step))
        start_time = time.time()      
        normalization_w_T_func()(np.arange(Topic_size))
        print("End normalization_w_T --- %s seconds ---" % round((time.time() - start_time),1))  
        
        print("Start normalization_T_d  EM : " + str(step))
        start_time = time.time()
        normalization_T_d_func()(np.arange(T_given_d_dic.shape[1]))
        print("End normalization_T_d --- %s seconds ---" % round((time.time() - start_time),1))  
        
        print("Start check P(w_i|T_k) normalization  EM : " + str(step))
        start_time = time.time()
        check_w_T_normalization_func()(np.arange(Topic_size))
        print("End check P(w_i|T_k) normalization --- %s seconds ---" % round((time.time() - start_time),1))
        
        print("Start check P(T_k|d_j) normalization   EM : " + str(step))
        start_time = time.time()
        check_T_d_normalization_func()(np.arange(T_given_d_dic.shape[1]))
        print("End check P(w_i|T_k) normalization --- %s seconds ---" % round((time.time() - start_time),1))
 
        print("Start check likelihood  EM : " + str(step))
        start_time = time.time()
        #check_Likelihood()
        print("End check Likelihood --- %s seconds ---" % round((time.time() - start_time),1))
        
        print("TOTAL one EM time --- %s seconds ---" % round((time.time() - TOTOAL_start_time),1))
        logging.info("Done Step : " + str(step))
        logging.info("L : " + str(L))
        EM_w_given_T_dic_path = os.path.join(os.path.sep,os.getcwd(),'EM_w_given_T_dic'+ str(step)+'.npz.npy')
        EM_T_given_d_dic_path = os.path.join(os.path.sep,os.getcwd(),'EM_T_given_d_dic'+str(step) +'.npz.npy')
        #np.save(EM_w_given_T_dic_path, EM_w_given_T_dic)
        #np.save(EM_T_given_d_dic_path, EM_T_given_d_dic)        
        w_given_T_dic = EM_w_given_T_dic.copy()
        T_given_d_dic = EM_T_given_d_dic.copy()
"""
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                                                                     Query Fold-in
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
"""
def init_BGLM():
    global BGLM_array
    BGLM_path = os.path.join(os.path.sep,os.getcwd(),'BGLM.txt')
    BGLM_matrix_path = os.path.join(os.path.sep,os.getcwd(),'BGLM.npz.npy')
    if os.path.exists(BGLM_matrix_path):
        print ("BGLM.npz.npy exists now load...")
        BGLM_array = np.load(BGLM_matrix_path)
    else :
        BGLM = open(BGLM_path, 'rb')
        num = 0
        for line in BGLM.readlines() :
            line = [float(i) for i in line.split()]
            for term in line:
                if term < 0 :
                    BGLM_array[ num ] = term
            num+=1
        np.save(BGLM_matrix_path, BGLM_array)
def renew_T_given_d_dic():
    global T_given_d_dic
    global EM_T_given_d_dic
    T_given_d_dic = np.random.uniform(1, 1000, (Topic_size, querydocument_size) ) 
    EM_T_given_d_dic = np.zeros((Topic_size, querydocument_size))    
def renormalization_T_given_d_dic():
    global T_given_d_dic
    T_given_d_dic = T_given_d_dic / T_given_d_dic.sum(axis = 0)
def check_renew_T_given_d_dic_normalization():
    for d_j in range(querydocument_size): 
        sum = 0
        for T_k in range(Topic_size): 
            sum += T_given_d_dic[T_k, d_j]
        assert math.isclose(sum, 1.0, rel_tol=1e-5) , "P(T "+"|d_"+str(d_j)+") error sum = " + str(sum) 
def init_log_T_given_d_dic():
    global T_given_d_dic
    T_given_d_dic = np.log( T_given_d_dic ) 
def readfile(filename):
    global d_w_count_matirx
    global d_w_logsum_matirx
    d_w_logsum_matirx = np.zeros( (querydocument_size,1) )
    d_w_count_matirx = np.zeros((querydocument_size, word_lengh ))
    start_time = time.time()
    Readingfile = filename
    readnum = 0
    os.chdir(filename)
    for filename in os.listdir(os.getcwd()):     
        relative_path = os.path.join(os.path.sep,os.getcwd(),filename)
        if os.path.isfile(relative_path):
            with open(relative_path, 'rb') as f:
                f = open(relative_path) 
                data = f.read().split()   
                if fnmatch.fnmatch(Readingfile, 'Document'):
                    renew_d_w_count_matirx(readnum, data, filename);    
        else:
            print (relative_path + "not exists")
        readnum += 1
    print ("END renew_d_w_count_matirx------------------------- ")
    print("--- %s seconds ---" %  round((time.time() - start_time),1))
    renew_log_d_w_count_matirx()
    print ("END renew_log_d_w_count_matirx------------------------- ")
    print("--- %s seconds ---" %  round((time.time() - start_time),1))
    os.chdir("..")
def renew_d_w_count_matirx(readnum,read_data, filename):
    global document_name_index
    global d_w_count_matirx
    global d_w_logsum_matirx
    document_name_index.append(filename)
    r=range(0,word_lengh)
    num = 0
    for term in read_data:
        if(num>=5):
            term = int(term)
            if ( term in r ):
                d_w_count_matirx[readnum,term] += 1 
        num += 1
    d_w_logsum_matirx[readnum] = np.log ( np.sum( d_w_count_matirx[readnum,:] ) ) 
def renew_d_w_logsum_matirx_func():
    def renew_d_w_logsum_matirx(d_j):
        global d_w_logsum_matirx
        global d_w_count_matirx
        d_w_logsum_matirx = np.zeros( (querydocument_size,1) )
        d_w_logsum_matirx[d_j] = np.log ( np.sum( d_w_count_matirx[d_j,:] ) )
    return np.frompyfunc(renew_d_w_logsum_matirx, 1, 0)
def renew_log_d_w_count_matirx():
    global d_w_count_matirx 
    d_w_count_matirx = np.log(d_w_count_matirx)
    d_w_count_matirx[ d_w_count_matirx == -np.inf] = LZERO   
"""
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                                                                     ranking
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
"""
def Get_grade(d_j):
    def getpro(w_i):
        return a * (  np.exp( d_w_count_matirx[d_j, w_i] - d_w_logsum_matirx[d_j] ) ) + b * ( np.exp( logsumexp(w_given_T_dic[w_i, :] + T_given_d_dic[:, d_j]) ) ) + (1-a-b) * np.exp(BGLM_array[w_i])
    return np.frompyfunc(getpro, 1, 1)
def rank_onedocument(d_j, query_data):
    """
    need_word_pro = []
    for word in np.where(query_count_matirx != 0)[0] :
        for repeat in range( query_count_matirx[word] ):
            need_word_pro.append(word) 
    """   
    return np.prod(Get_grade(d_j)(query_data).astype(np.float64))
def rank_onequery_func(query_data):
    def getpro(d_j):
        return rank_onedocument( d_j, query_data)
    return np.frompyfunc(getpro, 1, 1)
def get_rank(query_data,filename):
    temp = rank_onequery_func(query_data)(np.arange(querydocument_size)).astype(np.float64)
    temp = -temp   
    rankresult = np.argsort(temp)
    num = 0
    print (filename )    
    write_file.write(filename +',')
    for doc in rankresult:
        if(num < len(rankresult)-1):
            x = 0
            #print ("num < len(rankresult)-1")
            write_file.write(document_name_index[doc]+' ')
        else:
            x= 0
            #print ("num > len(rankresult)-1")
            write_file.write(document_name_index[doc]+'\n') 
        num+=1
    print (num)
    print ("END-result-write")
def init_query_count(quey_path,filename):  
    global query_count_matirx
    query_data = open(quey_path, 'rb').read()
    query_data = [int(i) for i in query_data.split()]
    query_data.remove(-1)
    query_count_matirx = np.bincount(query_data,  minlength = word_lengh)    
    get_rank(query_data,filename)
def readquery(filename):

    write_file.write('Query,RetrievedDocuments\n')
    Readingfile = filename
    os.chdir(filename)
    for filename in os.listdir(os.getcwd()):     
        relative_path = os.path.join(os.path.sep,os.getcwd(),filename)
        if os.path.isfile(relative_path):
            if fnmatch.fnmatch(Readingfile, 'Query'):
                init_query_count(relative_path,filename)
        else:
            print (relative_path + "not exists")
    os.chdir("..")

"""
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                                                                      main
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
"""
if __name__ == '__main__':    
    start_time = time.time()
    init()
    print ("END init------------------------- ")
    print("--- %s seconds ---" %  round((time.time() - start_time),1))
    start_time = time.time()
    init_normalization()
    print ("END init_normalization------------------------- ")
    print("--- %s seconds ---" % round((time.time() - start_time),1))
    start_time = time.time()
    check_init_normalization()
    print ("END check_init_normalization------------------------- ")
    print("--- %s seconds ---" % round((time.time() - start_time),1))
    start_time = time.time()
    init_log()
    print ("END init_log------------------------- ")
    print("--- %s seconds ---" % round((time.time() - start_time),1))
    print ("#######NOW counting word#############")
    start_time = time.time()
    init_d_w_count_matirx()
    print ("END init_d_w_count_matirx------------------------- ")
    print("--- %s seconds ---" % round((time.time() - start_time),1))
    start_time = time.time()
    init_d_w_logsum_matirx_func()(np.arange(T_given_d_dic.shape[1]))
    print ("END init_d_w_logsum_matirx_func------------------------- ")
    print("--- %s seconds ---" % round((time.time() - start_time),1))
    start_time = time.time()
    init_log_d_w_count_matirx()
    print ("END init_log_d_w_count_matirx------------------------- ")
    print("--- %s seconds ---" % round((time.time() - start_time),1))

    EM_step(50)
    print ("###########################renew########################## ")
    print ("renew T_d------------------------- ")
    start_time = time.time()
    init_BGLM() 
    print ("END init_BGLM------------------------- ")
    print("--- %s seconds ---" %  round((time.time() - start_time),1))  
    start_time = time.time()       
    renew_T_given_d_dic()
    print ("END renew_T_given_d_dic------------------------- ")
    print("--- %s seconds ---" %  round((time.time() - start_time),1))
    start_time = time.time()       
    renormalization_T_given_d_dic()
    print ("END renormalization_T_given_d_dic------------------------- ")
    print("--- %s seconds ---" %  round((time.time() - start_time),1))
    start_time = time.time()            
    check_renew_T_given_d_dic_normalization()
    print ("END check_renew_T_given_d_dic_normalization------------------------- ")
    print("--- %s seconds ---" %  round((time.time() - start_time),1))
    start_time = time.time() 
    init_log_T_given_d_dic()
    print ("END init_log_T_given_d_dic------------------------- ")
    print("--- %s seconds ---" %  round((time.time() - start_time),1))  
    readfile('Document')
    L = -100000000000000
    EM_step(50)
    start_time = time.time() 
    print ("Start ranking------------------------- ")
    readquery('Query')
    END_w_given_T_dic_path = os.path.join(os.path.sep,os.getcwd(),'END_w_given_T_dic'+'.npz.npy')
    END_T_given_d_dic_path = os.path.join(os.path.sep,os.getcwd(),'END_T_given_d_dic'+'.npz.npy')
    np.save(END_w_given_T_dic_path, w_given_T_dic)
    np.save(END_T_given_d_dic_path, T_given_d_dic)    
    write_file.close()
