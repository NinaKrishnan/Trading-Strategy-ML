import BagLearner as BL, LinRegLearner as LRL, numpy as np
class InsaneLearner(object):
    def __init__(self, verbose=False, bags=20):
        self.verbose=verbose
        self.learners=[]
    def author(self):
        return "nkrishnan42"
    def create_bags(self, bags=20):
        self.learners=[(BL.BagLearner(LRL.LinRegLearner, {}, 20, False, self.verbose) for i in range(bags))]               
    def add_evidence(self, data_x, data_y):
        for model in self.learners:
            model.add_evidence(data_x, data_y)  
    def query(self, points):
        predict_y= []
        for model in self.learners:
            predict_y.append(model.query(points))
        return np.mean(predict_y, axis=0)



    