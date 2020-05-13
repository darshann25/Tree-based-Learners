import numpy as np
import LinRegLearner as lrl
import BagLearner as bl

class InsaneLearner(object):

    def __init__(self, bag_learner = bl.BagLearner, num_bag_learners = 20, learner=lrl.LinRegLearner, kwargs = {}, verbose = False):

        self.bag_learners = []
        for i in range(num_bag_learners):
            self.bag_learners.append(bag_learner(learner=learner, **kwargs))
        self.verbose = verbose

    def addEvidence(self,dataX,dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """

        for learner in self.bag_learners:
            learner.addEvidence(dataX,dataY)

    def query(self,points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """

        predictions = np.array([learner.query(points) for learner in self.bag_learners])
        return np.mean(predictions,axis=0)

if __name__=="__main__":
    print("Insane Learner")
