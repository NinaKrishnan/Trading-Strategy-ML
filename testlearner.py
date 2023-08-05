""""""  		  	   		  	  			  		 			     			  	 
"""  		  	   		  	  			  		 			     			  	 
Test a learner.  (c) 2015 Tucker Balch  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		  	  			  		 			     			  	 
Atlanta, Georgia 30332  		  	   		  	  			  		 			     			  	 
All Rights Reserved  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
Template code for CS 4646/7646  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		  	  			  		 			     			  	 
works, including solutions to the projects assigned in this course. Students  		  	   		  	  			  		 			     			  	 
and other users of this template code are advised not to share it with others  		  	   		  	  			  		 			     			  	 
or to make it available on publicly viewable websites including repositories  		  	   		  	  			  		 			     			  	 
such as github and gitlab.  This copyright statement should not be removed  		  	   		  	  			  		 			     			  	 
or edited.  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
We do grant permission to share solutions privately with non-students such  		  	   		  	  			  		 			     			  	 
as potential employers. However, sharing with other current or future  		  	   		  	  			  		 			     			  	 
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		  	  			  		 			     			  	 
GT honor code violation.  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
-----do not edit anything above this line---  		  	   		  	  			  		 			     			  	 
"""  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 

import math
import DTLearner as dt
import RTLearner as rt
import BagLearner as bl
import InsaneLearner as il		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
import numpy as np
import matplotlib.pyplot as plt  	
import time	  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
import LinRegLearner as lrl  	

def author(self):
    return "nkrishnan42"

def split_data():
    file= "Data/Istanbul.csv"

    raw_data= np.genfromtxt(file, delimiter=',')
    data= raw_data[1:, 1:]
    data_x= data[:, :-1]
    data_y= data[:, -1]	  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
    # compute how much of the data is training and testing  		  	   		  	  			  		 			     			  	 
    train_rows = int(0.6 * data.shape[0])  		  	   		  	  			  		 			     			  	 
    test_rows = data.shape[0] - train_rows  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
    # separate out training and testing data  		  	   		  	  			  		 			     			  	 
    train_x = data[:train_rows, 0:-1]  		  	   	  	  			  		 			     			  	 
    train_y = data[:train_rows, -1]  		  	   		  	  			  		 			     			  	 
    test_x = data[train_rows:, 0:-1]  		  	   		  	  			  		 			     			  	 
    test_y = data[train_rows:, -1]  

    return [train_x, train_y, test_x, test_y]



def get_rmse(data_y, pred_y):
    rmse= math.sqrt(((data_y-pred_y)**2).sum()/data_y.shape[0])
    return rmse

def get_corr(data_y, pred_y):
    return np.corrcoef(data_y, pred_y)[0,1]


#Experiment 1: Overfitting, DTLearner


def run_experiment_1(train_x, train_y, test_x, test_y):

    in_sample_rmse= [0]
    out_sample_rmse= [0]

    in_sample_corr= [0]
    out_sample_corr= [0]


    for i in range(1, 82):
        learner= dt.DTLearner(leaf_size=i)
        learner.add_evidence(train_x, train_y)

        in_pred_y= learner.query(train_x)

        in_rmse= get_rmse(train_y, in_pred_y)
        in_corr= get_corr(train_y, in_pred_y)

        in_sample_rmse.append(in_rmse)
        in_sample_corr.append(in_corr)

        out_pred_y= learner.query(test_x)

        out_rmse= get_rmse(test_y, out_pred_y)
        out_corr= get_corr(test_y, out_pred_y)

        out_sample_rmse.append(out_rmse)
        out_sample_corr.append(out_corr)

    plt.xlim(1, 82)
    plt.xlabel("Leaf size")
    plt.ylabel("Root-mean-squared error")
    

    plt.plot(in_sample_rmse, label="in-sample RMSE")
    plt.plot(out_sample_rmse, label="out-of-sample RMSE")

    plt.title("DTLearner: Overfitting with Respect to Leaf Size")
    plt.legend()

    #plt.savefig("Exper_1_rmse.png")
    plt.savefig("DTLearner_rmse")


    plt.clf()





    
def run_experiment_2(train_x, train_y, test_x, test_y):
    bags=[25]
   
    for bag in bags:
        rmse_in_sample= []
        rmse_out_sample= []
        
        for i in range(1, 82):
            learner= bl.BagLearner(dt.DTLearner, {"leaf_size":i}, bag, False, False)
            learner.add_evidence(train_x, train_y)



            in_pred_y= learner.query(train_x)
            out_pred_y= learner.query(test_x)


            in_rmse= get_rmse(train_y, in_pred_y)
            out_rmse= get_rmse(test_y, out_pred_y)

            rmse_in_sample.append(in_rmse)
            rmse_out_sample.append(out_rmse)

        title= "BagLearner: Overfitting with Respect to Leaf Size Using "+ str(bag)+ " Bags"

        plt.title(title)
        plt.xlabel("Leaf size")
        plt.ylabel("Root-mean-squared error")


    
        plt.plot(rmse_in_sample, label="in-sample RMSE")
        plt.plot(rmse_out_sample, label="out-of-sample RMSE")
        plt.legend()
        
        chart= "Exper_2_bags"+str(bag)+".png"
        plt.savefig(chart)
        plt.clf()


def get_mae(predicted_y, data_y):
    n= len(data_y)
    summ= 0

    for i in range(n):
        predicted= predicted_y[i]
        actual= data_y[i]

        error= abs(actual-predicted)

        summ+=error

    return summ/n


def get_random_sample_data():
    train_x, train_y, test_x, test_y= split_data()
    n= train_x.shape[0]

    random_sample= np.random.choice(n, n, True)


    xbag= train_x[random_sample]
    ybag= train_y[random_sample]

    return (xbag, ybag, test_x, test_y)

def compare_mae(train_x, train_y, test_x, test_y,leaf_size):
    dt_in= []
    rt_in= []
    dt_out= []
    rt_out= []

    for i in range(leaf_size):
        dtl= dt.DTLearner(leaf_size=i)
        rtl= rt.RTLearner(leaf_size=i)

        dtl.add_evidence(train_x, train_y)
        rtl.add_evidence(train_x, train_y)

        dt_in_pred= dtl.query(train_x)
        dt_out_pred= dtl.query(test_x)

        rt_in_pred= rtl.query(train_x)
        rt_out_pred= rtl.query(test_x)


        dt_in.append(get_mae(dt_in_pred, train_y))
        dt_out.append(get_mae(dt_out_pred, test_y))

        rt_in.append(get_mae(rt_in_pred, train_y))
        rt_out.append(get_mae(rt_out_pred, test_y))


    return (dt_in, rt_in, dt_out, rt_out)


def get_time_to_train(train_x, train_y):
    dt_time= [0]
    rt_time= [0]

    for i in range(1, 51):
        dtl= dt.DTLearner(leaf_size=i)
        rtl= rt.RTLearner(leaf_size=i)

        start= time.time()
        rtl.add_evidence(train_x, train_y)
        end= time.time()

        rt_time.append(end-start)


        start= time.time()
        dtl.add_evidence(train_x, train_y)
        end= time.time()

        dt_time.append(end-start)


    return (dt_time, rt_time)





def run_experiment_3(train_x, train_y, test_x, test_y):
    plt.clf()

    din, rin, dout, rout= compare_mae(train_x, train_y, test_x, test_y, 81)

    plt.xlim(1, 82)

    plt. xlabel("Leaf size")
    plt.ylabel("Mean absolute error")
    plt.title("Mean Absolute Error with respect to Leaf Size")

    plt.plot(din, label= "DTLearner in sample error")
    plt.plot(rin, label="RTLearner in sample error")

    plt.plot(dout, label= "DTLearner out of sample error")
    plt.plot(rout, label="RTLearner out of sample error")
    plt.legend()

    plt.savefig("Exper_3_MAE_vs_leafsize")

    plt.clf()

    dt_time, rt_time= get_time_to_train(train_x, train_y)

    plt.xlim(1, 55)
    plt.xlabel("Leaf size")
    plt.ylabel("Time spent training")
    plt.title("Time Spent Training with respect to Leaf Size")

    plt.plot(dt_time, label="DTLearner")
    plt.plot(rt_time, label="RTLearner")
    plt.legend()

    plt.savefig("Exper_3_time_to_train.png")



    
if __name__=="__main__": 
    train_x, train_y, test_x, test_y= split_data()

    run_experiment_1(train_x, train_y, test_x, test_y)
    run_experiment_2(train_x, train_y, test_x, test_y)
    run_experiment_3(train_x, train_y, test_x, test_y)



    





