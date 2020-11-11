import numpy as np
from modAL.models import ActiveLearner
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF
import matplotlib.pyplot as plt
import datetime
import os
from modAL.acquisition import optimizer_PI, optimizer_EI, optimizer_UCB, max_PI, max_EI, max_UCB
import pipette
import logging

X = np.linspace(0, 1, 1000)
X_initial = np.array([112, 372, 448, 982, 554, 603, 884])
# y = [300, 900, 700, 200, 1000, 300, 100, 400]
y = 100 * X + np.random.normal(scale=0.1, size=X.shape)


nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
os.makedirs('al_figures/{}'.format(nowTime))
logging.basicConfig(filename='log/{}.log'.format(nowTime), level=logging.INFO)
# create logger
# logger_name = 'log/{}.log'.format(nowTime)
# logger = logging.getLogger(logger_name)
# logger.setLevel(logging.info)

def GP_regression_std(regressor, X):
    _, std = regressor.predict(X, return_std=True)
    query_idx = np.argmax(std)
    return query_idx, X[query_idx]


# get randomly chosen X array
def initial_X(n_initial, X, real_random=True):
    if real_random:
        initial_idx = np.random.choice(range(len(X)), size=n_initial, replace=False).reshape(-1, 1)
    else:
        initial_idx = np.random.choice(X_initial, size=n_initial, replace=False).reshape(-1, 1)
    initial_instance = X[initial_idx]
    return initial_idx, initial_instance


def run(n_initial, n_queries, X=X, well_counter=0, tip_counter=0):

    def draw_plot(X, y_pred, y_std, X_training, y_training, title='', X_next=False, X_next_idx=0):
        nonlocal well_counter, pic_idx, y_ave
        pic_idx += 1
        with plt.style.context('seaborn-white'):
            fig = plt.figure(figsize=(10, 5))
            ax1 = fig.add_subplot(111)
            plt.plot(X, y_pred)
            plt.fill_between(X, y_pred - y_std, y_pred + y_std, alpha=0.2)
            '''Highlight observed points'''
            plt.scatter(X_training, y_training, c='green', s=100)
            if X_next:
                plt.vlines(X[X_next_idx], y_pred[X_next_idx] - y_std[X_next_idx],
                           y_pred[X_next_idx] + y_std[X_next_idx],
                           colors="r", linestyles="dashed")
            plt.title(title)
            ax1.set_xlabel('Proportion of solute')
            ax1.set_ylabel('Normalized value')
            ax2 = ax1.twinx()
            ax2.set_ylim(y_ave * ax1.get_ylim()[0], y_ave * ax1.get_ylim()[1])
            ax2.set_ylabel('Conductivity(μS/cm)')
            plt.savefig('al_figures/{}/{}'.format(nowTime, pic_idx))
            plt.show()

    # get observed y value for X input (array or a single number)
    def get_y(X_training, X_index):
        nonlocal well_counter, tip_counter
        well_counter = well_counter
        tip_counter = tip_counter
        for i in range(len(X_training)):
            y_input = pipette.run(X_training[i], well_counter, tip_counter)
            # y_input = y[X_index[i]]
            if i == 0:
                y_training = np.array([y_input])
            else:
                y_training = np.vstack((y_training, np.array([y_input])))
            well_counter += 1
            tip_counter += 3
            for j in range(len(y_training)):
                print(X_training[j], y_training[j], '\n')
                logging.info("{}, {}".format(X_training[j], y_training[j]))
        return y_training

    # parameter initialization
    well_counter = well_counter
    tip_counter = tip_counter
    pic_idx = 0
    kernel = RBF(length_scale=0.05, length_scale_bounds=(1e-2, 1e3)) \
        # + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))

    # get initial training data
    initial_idx, X_training = initial_X(n_initial, X, real_random=False)
    y_training = get_y(X_training, initial_idx)

    # print initial data
    for j in range(len(X_training)):
        print(X_training[j], y_training[j], '\n')
        logging.info("{}, {}".format(X_training[j], y_training[j]))

    # normalize y_training
    y_ave = np.average(y_training)
    # y_ave = 1
    y_training_norm = y_training / y_ave

    # gaussian process
    regressor = ActiveLearner(
        estimator=GaussianProcessRegressor(kernel=kernel),
        query_strategy=GP_regression_std,
        X_training=X_training.reshape(-1, 1), y_training=y_training_norm.reshape(-1, 1)
    )

    # active learning
    for i in range(n_queries):

        # predict
        y_pred, y_std = regressor.predict(X.reshape(-1, 1), return_std=True)
        y_pred, y_std = y_pred.ravel(), y_std.ravel()
        # print(y_std[10:20])
        # querying next observation point
        query_idx, query_instance = regressor.query(X.reshape(-1, 1))

        # plot
        draw_plot(X, y_pred, y_std, X_training, y_training_norm, 'Gaussian Process')
        draw_plot(X, y_pred, y_std, X_training, y_training_norm, 'Bayesian Optimization', True, query_idx)

        # observing y at queried X
        print("$$$", "Observing point of X={}".format(X[query_idx]), "$$$", sep='\n')
        y_input = get_y(query_instance, np.array([query_idx]))
        y_input_norm = y_input / y_ave

        # update regressor
        regressor.teach(X[query_idx].reshape(1, -1), y_input_norm.reshape(1, -1))

        # update X_training & y_training
        (X_training, y_training) = (np.vstack((X_training, X[query_idx])), np.vstack((y_training, y_input)))
        logging.info("")
        for j in range(len(X_training)):
            print(X_training[j], y_training[j], '\n')
            logging.info("{}, {}".format(X_training[j], y_training[j]))
        y_training_norm = np.vstack((y_training_norm, y_input_norm))

    # output the result
    y_pred, y_std = regressor.predict(X.reshape(-1, 1), return_std=True)
    y_pred, y_std = y_pred.ravel(), y_std.ravel()
    draw_plot(X, y_pred, y_std, X_training, y_training_norm, 'Final Result')
    y_pred, y_std = regressor.predict(X.reshape(-1, 1), return_std=True)
    print(np.argmax(y_pred) / 1000, max(y_pred) * y_ave)

pipette.sleep(30)
run(n_initial=3, n_queries=3, well_counter=0)

