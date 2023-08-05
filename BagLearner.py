import DTLearner as DT
import RTLearner as RT
import numpy as np



class BagLearner(object):
    def __init__(self, learner, kwargs, bags, boost, verbose):
        self.learner= learner
        self.kwargs= kwargs
        self.bags= bags
        self.boost= boost
        self.verbose= verbose
        self.learners= []


    def author(self):
        return "nkrishnan42"

    def add_evidence(self, data_x, data_y):
        n= data_x.shape[0]
    

        for i in range(0,self.bags):
            self.learners.append(self.learner(**self.kwargs))
            random_sample= np.random.choice(n, n, True)

            x_bag= data_x[random_sample]
            y_bag= data_y[random_sample]

            self.learners[i].add_evidence(x_bag, y_bag)

        
    def query(self, points):
        predict_y= []

        for model in self.learners:
            output= model.query(points)
            predict_y.append(output)

        return np.mean(predict_y, axis=0)






