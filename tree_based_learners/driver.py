import numpy as np
import pandas as pd
import math
import LinRegLearner as lrl
import DTLearner as dt
import RTLearner as rt
import BagLearner as bl
import InsaneLearner as it
import sys
import matplotlib.pyplot as plt
from datetime import datetime

def plot_selected(df, columns, start_index, end_index, title='Assess Learners', x_lim=(0, 100), y_lim=(0, 1),
                  x_label='Leaf Size', y_label='RMSE'):
    """
    @summary: Plot the desired columns over index values in the given range.
    @param df: dataframe to plot the data
    @param columns: columns to use for the plot
    @param start_index: start index for the data
    @param end_index: end index for the data
    @param title: title for the chart
    @param x_lim: limit for x axis. Also used to invert the axis.
    @param y_lim: limit for y axis
    @param x_label: label for x axis
    @param y_label: label for y axis
    @return: prints chart + saves chart with name of the title
    """
    df = df.loc[start_index: end_index, columns]

    ax = df.plot(title=title, color='bg')
    ax.set_xlabel(x_label)
    ax.set_xlim(x_lim)
    ax.set_ylabel(y_label)
    ax.grid(True)

    ax.legend(loc="best")
    plt.savefig(title + '.png', dpi=500)
    plt.show(block=False)
    return


def fnExperiment1(trainX, trainY, testX, testY, verbose=False):
    """
    @summary: run experiment 1 to compare in-sample and out-sample rmse for different leaf sizes of decision trees to gauge overfitting
    @param trainX: data with X in-sample features
    @param trainY: data with Y in-sample target variable
    @param testX: data with X out-of-sample features
    @param testY: data with Y out-of-sample target variable
    @verbose: print log statements for debugging
    @return: saves results in .csv file + saves plot in .png file + shows plot
    """

    if verbose: print('Starting experiment 1 ...')

    # create a learner and train it
    results = pd.DataFrame(
        columns={'leaf_size': [], 'insample_rmse': [], 'insample_corr': [], 'outsample_rmse': [], 'outsample_corr': []})
    iterations = 100

    for leaf_size in range(1, iterations + 1):
        learner = dt.DTLearner(leaf_size=leaf_size, verbose=False)
        learner.addEvidence(trainX, trainY)  # train it

        # evaluate in sample
        predY = learner.query(trainX)  # get the predictions
        in_rmse = math.sqrt(((trainY - predY) ** 2).sum() / trainY.shape[0])
        c = np.corrcoef(predY, y=trainY)
        in_corr = c[0, 1]
        if verbose:
            print()
            print('Leaf size : {}'.format(leaf_size))
            print("In sample results")
            print(f"RMSE: {in_rmse}")
            print(f"corr: {c[0, 1]}")

        # evaluate out of sample
        predY = learner.query(testX)  # get the predictions
        out_rmse = math.sqrt(((testY - predY) ** 2).sum() / testY.shape[0])
        c = np.corrcoef(predY, y=testY)
        out_corr = c[0, 1]
        if verbose:
            print("Out of sample results")
            print(f"RMSE: {out_rmse}")
            print(f"corr: {c[0, 1]}")

        results.loc[leaf_size - 1] = [leaf_size, in_rmse, in_corr, out_rmse, out_corr]

    if verbose: print(results)
    results.to_csv('exp1_leafsize_tuning_rmse.csv')
    plot_selected(results, ['insample_rmse', 'outsample_rmse'], 0, iterations - 1,
                  'Experiment 1 - Overfitting by Tuning Leaf Size', x_lim=(0, iterations), y_lim=(0, 1),
                  x_label='Leaf Size', y_label='RMSE')
    plot_selected(results, ['insample_rmse', 'outsample_rmse'], 0, iterations - 1,
                  'Experiment 1 - Overfitting by Tuning Leaf Size (Inverted)', x_lim=(iterations, 0), y_lim=(0, 1),
                  x_label='Decreasing Leaf Size', y_label='RMSE')
    if verbose: print('Experiment 1 completed!')
    return


def fnExperiment2(trainX, trainY, testX, testY, verbose=False):
    """
    @summary: run experiment 2 to compare in-sample and out-sample rmse for different leaf sizes of decision trees with bagging to gauge overfitting
    @param trainX: data with X in-sample features
    @param trainY: data with Y in-sample target variable
    @param testX: data with X out-of-sample features
    @param testY: data with Y out-of-sample target variable
    @verbose: print log statements for debugging
    @return: saves results in .csv file + saves plot in .png file + shows plot
    """

    if verbose: print('Starting experiment 2 ...')

    # create a learner and train it
    results = pd.DataFrame(
        columns={'leaf_size': [], 'insample_rmse': [], 'insample_corr': [], 'outsample_rmse': [], 'outsample_corr': []})
    iterations = 100

    for leaf_size in range(1, iterations + 1):
        learner = bl.BagLearner(learner=dt.DTLearner, kwargs={"leaf_size": leaf_size}, bags=20, boost=False,
                                verbose=False)
        learner.addEvidence(trainX, trainY)  # train it

        # evaluate in sample
        predY = learner.query(trainX)  # get the predictions
        in_rmse = math.sqrt(((trainY - predY) ** 2).sum() / trainY.shape[0])
        c = np.corrcoef(predY, y=trainY)
        in_corr = c[0, 1]
        if verbose:
            print()
            print('Leaf size : {}'.format(leaf_size))
            print("In sample results")
            print(f"RMSE: {in_rmse}")
            print(f"corr: {c[0, 1]}")

        # evaluate out of sample
        predY = learner.query(testX)  # get the predictions
        out_rmse = math.sqrt(((testY - predY) ** 2).sum() / testY.shape[0])
        c = np.corrcoef(predY, y=testY)
        out_corr = c[0, 1]
        if verbose:
            print("Out of sample results")
            print(f"RMSE: {out_rmse}")
            print(f"corr: {c[0, 1]}")

        results.loc[leaf_size - 1] = [leaf_size, in_rmse, in_corr, out_rmse, out_corr]

    if verbose: print(results)
    results.to_csv('exp2_bagging_leafsize_tuning_rmse.csv')
    plot_selected(results, ['insample_rmse', 'outsample_rmse'], 0, iterations - 1,
                  'Experiment 2 - Effects of Bagging on Overfitting 10', x_lim=(0, iterations), y_lim=(0, 1),
                  x_label='Leaf Size', y_label='RMSE')
    plot_selected(results, ['insample_rmse', 'outsample_rmse'], 0, iterations - 1,
                  'Experiment 2 - Effects of Bagging on Overfitting 10 (Inverted)', x_lim=(iterations, 0), y_lim=(0, 1),
                  x_label='Decreasing Leaf Size', y_label='RMSE')
    if verbose: print('Completed experiment 2!')
    return


def fnExperiment3A(trainX, trainY, testX, testY, verbose=False):
    """
    @summary: run experiment 3A to compare in-sample and out-sample variance for different leaf sizes of decision trees and random tree w/ and w/o bagging
    @param trainX: data with X in-sample features
    @param trainY: data with Y in-sample target variable
    @param testX: data with X out-of-sample features
    @param testY: data with Y out-of-sample target variable
    @verbose: print log statements for debugging
    @return: saves results in .csv file + saves plot in .png file + shows plot
    """

    if verbose: print('Starting experiment 3A ...')

    # create a learner and train it
    results = pd.DataFrame(
        columns={'leaf_size': [], 'dt_insample_var': [], 'dt_outsample_var': [], 'rt_insample_var': [],
                 'rt_outsample_var': [], 'bagged_dt_insample_var': [], 'bagged_dt_outsample_var': [],
                 'bagged_rt_insample_var': [], 'bagged_rt_outsample_var': []})
    iterations = 100

    for leaf_size in range(1, iterations + 1):
        # train and query a decision tree
        learner = dt.DTLearner(leaf_size=leaf_size, verbose=False)
        learner.addEvidence(trainX, trainY)  # train it
        # evaluate in sample
        predY = learner.query(trainX)  # get the predictions
        dt_in_var = ((predY - np.mean(predY)) ** 2).sum() / trainY.shape[0]
        # evaluate out of sample
        predY = learner.query(testX)  # get the predictions
        dt_out_var = ((predY - np.mean(predY)) ** 2).sum() / trainY.shape[0]

        # train and query a random tree
        learner = rt.RTLearner(leaf_size=leaf_size, verbose=False)
        learner.addEvidence(trainX, trainY)  # train it
        # evaluate in sample
        predY = learner.query(trainX)  # get the predictions
        rt_in_var = ((predY - np.mean(predY)) ** 2).sum() / trainY.shape[0]
        # evaluate out of sample
        predY = learner.query(testX)  # get the predictions
        rt_out_var = ((predY - np.mean(predY)) ** 2).sum() / trainY.shape[0]

        # train and query a bagged decision tree
        learner = bl.BagLearner(learner=dt.DTLearner, kwargs={"leaf_size": leaf_size}, bags=20, boost=False,
                                verbose=False)
        learner.addEvidence(trainX, trainY)  # train it
        # evaluate in sample
        predY = learner.query(trainX)  # get the predictions
        bldt_in_var = ((predY - np.mean(predY)) ** 2).sum() / trainY.shape[0]
        # evaluate out of sample
        predY = learner.query(testX)  # get the predictions
        bldt_out_var = ((predY - np.mean(predY)) ** 2).sum() / trainY.shape[0]

        # train and query a bagged random tree
        learner = bl.BagLearner(learner=rt.RTLearner, kwargs={"leaf_size": leaf_size}, bags=20, boost=False,
                                verbose=False)
        learner.addEvidence(trainX, trainY)  # train it
        # evaluate in sample
        predY = learner.query(trainX)  # get the predictions
        blrt_in_var = ((predY - np.mean(predY)) ** 2).sum() / trainY.shape[0]
        # evaluate out of sample
        predY = learner.query(testX)  # get the predictions
        blrt_out_var = ((predY - np.mean(predY)) ** 2).sum() / trainY.shape[0]

        results.loc[leaf_size - 1] = [leaf_size, dt_in_var, dt_out_var, rt_in_var, rt_out_var, bldt_in_var,
                                      bldt_out_var, blrt_in_var, blrt_out_var]

    if verbose: print(results)
    results.to_csv('exp3_variance.csv')
    plot_selected(results, ['dt_insample_var', 'rt_insample_var'], 0, iterations - 1,
                  'Experiment 3A - In-Sample Variance', x_lim=(0, iterations), y_lim=(0, 1), x_label='Leaf Size',
                  y_label='Variance')
    plot_selected(results, ['dt_outsample_var', 'rt_outsample_var'], 0, iterations - 1,
                  'Experiment 3A - Out-of-Sample Variance', x_lim=(0, iterations), y_lim=(0, 1), x_label='Leaf Size',
                  y_label='Variance')
    plot_selected(results, ['bagged_dt_insample_var', 'bagged_rt_insample_var'], 0, iterations - 1,
                  'Experiment 3A - Bagged In-Sample Variance', x_lim=(0, iterations), y_lim=(0, 1), x_label='Leaf Size',
                  y_label='Variance')
    plot_selected(results, ['bagged_dt_outsample_var', 'bagged_rt_outsample_var'], 0, iterations - 1,
                  'Experiment 3A - Bagged Out-of-Sample Variance', x_lim=(0, iterations), y_lim=(0, 1),
                  x_label='Leaf Size', y_label='Variance')
    if verbose: print('Completed experiment 3A!')
    return


def fnExperiment3B(trainX, trainY, testX, testY, verbose=False):
    """
    @summary: run experiment 3B to compare learning time for different leaf sizes of decision trees and random tree w/ and w/o bagging
    @param trainX: data with X in-sample features
    @param trainY: data with Y in-sample target variable
    @param testX: data with X out-of-sample features
    @param testY: data with Y out-of-sample target variable
    @verbose: print log statements for debugging
    @return: saves results in .csv file + saves plot in .png file + shows plot
    """

    if verbose: print('Starting experiment 3B ...')

    # create a learner and train it
    results = pd.DataFrame(
        columns={'leaf_size': [], 'dt_time': [], 'rt_time': [], 'bagged_dt_time': [], 'bagged_rt_time': []})
    iterations = 100

    for leaf_size in range(1, iterations + 1):
        # train and query a decision tree
        t1 = datetime.now()
        learner = dt.DTLearner(leaf_size=leaf_size, verbose=False)
        learner.addEvidence(trainX, trainY)  # train it
        t2 = datetime.now()
        delta = t2 - t1
        dt_time = delta.total_seconds()

        # train and query a random tree
        t1 = datetime.now()
        learner = rt.RTLearner(leaf_size=leaf_size, verbose=False)
        learner.addEvidence(trainX, trainY)  # train it
        t2 = datetime.now()
        delta = t2 - t1
        rt_time = delta.total_seconds()

        # train and query a bagged decision tree
        t1 = datetime.now()
        learner = bl.BagLearner(learner=dt.DTLearner, kwargs={"leaf_size": leaf_size}, bags=20, boost=False,
                                verbose=False)
        learner.addEvidence(trainX, trainY)  # train it
        t2 = datetime.now()
        delta = t2 - t1
        bldt_time = delta.total_seconds()

        # train and query a bagged random tree
        t1 = datetime.now()
        learner = bl.BagLearner(learner=rt.RTLearner, kwargs={"leaf_size": leaf_size}, bags=20, boost=False,
                                verbose=False)
        learner.addEvidence(trainX, trainY)  # train it
        t2 = datetime.now()
        delta = t2 - t1
        blrt_time = delta.total_seconds()

        results.loc[leaf_size - 1] = [leaf_size, dt_time, rt_time, bldt_time, blrt_time]

    if verbose: print(results)
    results.to_csv('exp3_time.csv')
    plot_selected(results, ['dt_time', 'rt_time'], 0, iterations - 1, 'Experiment 3B - Time', x_lim=(0, iterations),
                  y_lim=(0, 1), x_label='Leaf Size', y_label='Time')
    plot_selected(results, ['bagged_dt_time', 'bagged_rt_time'], 0, iterations - 1, 'Experiment 3B - Time with Bagging',
                  x_lim=(0, iterations), y_lim=(0, 1), x_label='Leaf Size', y_label='Time')
    if verbose: print('Completed experiment 3B!')
    return


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python driver.py <filename>")
        sys.exit(1)
    inf = open(sys.argv[1])
    # Update: Added list slicer to ignore headers and date column
    data = np.array([list(map(float, s.strip().split(',')[1:])) for s in inf.readlines()[1:]])

    verbose = False

    # compute how much of the data is training and testing  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
    train_rows = int(0.6 * data.shape[0])
    # print(train_rows.head())
    test_rows = data.shape[0] - train_rows

    # separate out training and testing data  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
    trainX = data[:train_rows, 0:-1]
    trainY = data[:train_rows, -1]
    testX = data[train_rows:, 0:-1]
    testY = data[train_rows:, -1]

    fnExperiment1(trainX, trainY, testX, testY, verbose=verbose)
    fnExperiment2(trainX, trainY, testX, testY, verbose=verbose)
    fnExperiment3A(trainX, trainY, testX, testY, verbose=verbose)
    fnExperiment3B(trainX, trainY, testX, testY, verbose=verbose)

    plt.show()
