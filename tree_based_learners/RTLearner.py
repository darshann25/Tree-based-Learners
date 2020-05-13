import numpy as np

class RTLearner(object):

    def __init__(self, leaf_size, verbose = False):
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.tree = None

    def addEvidence(self,dataX,dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """

        dataY = dataY.reshape(len(dataY), 1)
        data = np.append(dataX, dataY, axis = 1)

        # build and save the model
        if self.verbose: print('Building the Random Tree')
        self.tree = self.__build_tree__(data)

    def __build_tree__(self, data):
        """
        @summary: Build a ranomd tree using Adele Cutler's algorithm
        @param dataX: X values of data to add
        @param dataY: the Y training values
        @return: constructed decision tree in the form of np arrays
        """
        dataX = data[:,0:-1]
        dataY = data[:,-1]

        # Check if less than equal to leaf size elements left
        if dataX.shape[0] <= self.leaf_size:
            node = np.array([[-1, np.mean(dataY[:]), -1, -1]])
            if self.verbose:
                print('Number of elements are less than leaf size')
                print(node)
            return node

        # Check if all predictions are the same
        elif np.all(dataY[:] == dataY[0], axis = 0):
            node = np.array([[-1, np.mean(dataY[:]), -1, -1]])
            if self.verbose:
                print('All predictions are the same')
                print(node)
            return node

        else:
            random_feature = self.__random_feature__(dataX, dataY)
            split_value = self.__choose_random_split__(dataX, random_feature)
            left_tree = self.__build_tree__(data[data[:,random_feature] <= split_value])
            right_tree = self.__build_tree__(data[data[:,random_feature] > split_value])
            root = np.array([[random_feature, split_value, 1, left_tree.shape[0] + 1]])
            if self.verbose:
                print('Root : {}'.format(root))
                print('Root shape : {}'.format(root.shape))
                print('Left shape : {}'.format(left_tree.shape))
                print('Right shape : {}'.format(right_tree.shape))

            return np.concatenate((root, left_tree, right_tree), axis=0)

    def __random_feature__(self, dataX, dataY):
        """
        @summary: calculates the random feature
        @param dataX: X values of data to add
        @param dataY: the Y training values
        @return: returns index of random feature
        """

        random_feature = np.random.randint(low = 0, high = dataX.shape[1], size = 1)[0]
        if self.verbose: print('Random feature is {}'.format(random_feature))

        while np.all(dataX[:,random_feature] == dataX[0,random_feature], axis=0):
            random_feature = np.random.randint(low = 0, high = dataX.shape[1], size = 1)[0]
            if self.verbose: print('Random feature is {}'.format(random_feature))

        return int(random_feature)

    def __choose_random_split__(self, dataX, random_feature):
        """
        @summary: choose best split value if split value leads to imbalanced tree
        @param dataX: X values of data to add
        @param random_feature: index of best feature
        @return: split value for balanced tree
        """

        row1 = np.random.randint(low = 0, high = dataX.shape[0], size = 1)[0]
        row2 = np.random.randint(low = 0, high = dataX.shape[0], size = 1)[0]
        split_val = (dataX[row1,random_feature] + dataX[row2,random_feature]) / 2.0
        if self.verbose: print('Split Value : {}'.format(split_val))

        left_data = dataX[dataX[:,random_feature] <= split_val]
        right_data = dataX[dataX[:,random_feature] > split_val]

        while left_data.shape[0] == 0 or right_data.shape[0] == 0:

            row1 = np.random.randint(low = 0, high = dataX.shape[0], size = 1)[0]
            row2 = np.random.randint(low = 0, high = dataX.shape[0], size = 1)[0]
            split_val = (dataX[row1,random_feature] + dataX[row2,random_feature]) / 2.0
            if self.verbose: print('Split Value : {}'.format(split_val))

            left_data = dataX[dataX[:,random_feature] <= split_val]
            right_data = dataX[dataX[:,random_feature] > split_val]

        return split_val

    def query(self,points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """

        predictions = []

        for point in points:

            prediction = self.__query_tree__(point, 0)
            predictions.append(prediction)
            if self.verbose: print('Point : {} | Prediction : {}'.format(point, prediction))

        return predictions

    def __query_tree__(self, point, level = 0):
        """
        @summary: Query data point in decision tree
        @param point: data point to be queried
        @return: prediction for data point
        """

        feature, split_val = self.tree[level, 0:2]
        prediction = -1

        if int(feature) == -1:
            self.verbose: print('Leaf has been reached. Prediction is {}'.format(split_val))
            prediction =  split_val

        elif point[int(feature)] <= split_val:
            if self.verbose: print('Query left subtree')
            prediction = self.__query_tree__(point, level + 1)
            # No need for int(self.tree[level, 2]) since left tree is always indexed 1 relative to current row

        elif point[int(feature)] > split_val:
            if self.verbose: print('Query right subtree')
            prediction = self.__query_tree__(point, level + int(self.tree[level, 3]))

        else:
            print('Tree traversal ERROR!')

        return prediction

if __name__=="__main__":
    print("Random Tree")
