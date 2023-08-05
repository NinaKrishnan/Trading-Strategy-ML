""""""  		  	   		  	  			  		 			     			  	 
"""  		  	   		  	  			  		 			     			  	 
  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
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


import numpy as np  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
class DTLearner(object):  		  	   		  	  			  		 			     			  	 
    """  		  	   		  	  			  		 			     			  	 
    This is a Linear Regression Learner. It is implemented correctly.  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		  	  			  		 			     			  	 
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.  		  	   		  	  			  		 			     			  	 
    :type verbose: bool  		  	   		  	  			  		 			     			  	 
    """  		  	   		  	  			  		 			     			  	 
    def __init__(self, leaf_size=1,verbose=False):  		  	   		  	  			  		 			     			  	 
        self.tree= None
        self.verbose= verbose
        self.leaf_size=leaf_size

       
          # move along, these aren't the drones you're looking for  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
    def author(self):  		  	   		  	  			  		 			     			  	 
        	  	   		  	  			  		 			     			  	 
        return "nkrishnan42"	



  		  	   		  	  			  		 			     			  	 
    def add_evidence(self, data_x, data_y):  		  	   		  	  			  		 			     			  	 
        self.tree= self.build_tree(data_x, data_y)

        if self.verbose==True:
            print(self.tree)
            

    def get_split_val_index(self, data_x, data_y):
        splitVal_index= 0
        max_corr= 0

        for i in range(data_x.shape[1]):
            col= data_x[:, i]
        
            coeff_matrix= np.nan_to_num(np.corrcoef(col, data_y))
            coeff_val= abs(coeff_matrix[0, 1])

            if coeff_val>max_corr:
                max_corr=coeff_val
                splitVal_index= i

        return int(splitVal_index)




    def build_tree(self, data_x, data_y):
        #If data size < leaf size, return
        leaf_size=self.leaf_size
        data_size= data_x.shape[0]
        res= np.array([[-1, np.mean(data_y), -1, -1]])
      
        if data_size <= leaf_size or self.is_same_data(data_y):

          
            return res

        splitVal_index= self.get_split_val_index(data_x, data_y)
        split_feature= data_x[:, splitVal_index]
        split_val= np.median(split_feature)
        if np.all(split_feature<=split_val):
            
            return res

        left_data_x= data_x[split_feature<=split_val]
        right_data_x= data_x[split_feature>split_val]

        left_data_y= data_y[split_feature<=split_val]
        right_data_y= data_y[split_feature>split_val]
        
        

        left_subtree= self.build_tree(left_data_x, left_data_y)
        right_subtree= self.build_tree(right_data_x, right_data_y)

        root= np.array([[int(splitVal_index), split_val, 1, left_subtree.shape[0]+1]])
      
        

        tree= np.vstack((root, left_subtree, right_subtree))

        return tree

        

        

    def is_same_data(self, data_y):
        return np.std(data_y)==0 or data_y[-1]==data_y[0]




  		  	   		  	  			  		 			     			  	 
    def query(self, points):  		  	   		  	  			  		 			     			  	 		  	   		  	  			  		 			     			  	 
        predict_y= np.zeros((points.shape[0]))

        for i in range(len(points)):
            point= points[i] #current query
            curr_index= 0 #start search at index 0

            while True:
                #Check to see if we are at a leaf
                factor= int(self.tree[curr_index, 0])
                if factor == -1:
                    prediction= self.tree[curr_index, 1]
                    predict_y[i]=prediction
                    break
                else:
                    split_val= self.tree[curr_index, 1]
                    query_val= point[factor]

                    if query_val<= split_val:
                        #traverse left subtree
                        curr_index=int(curr_index+self.tree[curr_index, 2])

                    else:
                        curr_index=int(curr_index+self.tree[curr_index, 3])
        return predict_y




if __name__ == "__main__":  		  	   		  	  			  		 			     			  	 
    print("the secret clue is 'zzyzx'")  		  	   		  	  			  		 			     			  	 
