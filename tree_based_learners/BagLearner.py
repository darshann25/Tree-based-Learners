import numpy as np

class BagLearner(object):

    def __init__(self, learner, kwargs={}, bags = 20, boost = False, verbose = False):

        self.learners = []
        for i in range(bags):
            self.learners.append(learner(**kwargs))
        self.bags = bags
        self.boost = boost
        self.verbose = verbose

    def addEvidence(self,dataX,dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """

        datapoints = dataX.shape[0]

        for learner in self.learners:

            # get random indices of the same size as the data
            indices = np.random.choice(datapoints, datapoints)

            bagX = dataX[indices]
            bagY = dataY[indices]

            learner.addEvidence(bagX,bagY)

    def query(self,points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """

        predictions = np.array([learner.query(points) for learner in self.learners])
        return np.mean(predictions,axis=0)

if __name__=="__main__":
    print("Bag Learner")
