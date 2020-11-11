import numpy as np
from modAL.models import ActiveLearner
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF
import matplotlib.pyplot as plt
from modAL.acquisition import optimizer_PI, optimizer_EI, optimizer_UCB, max_PI, max_EI, max_UCB
import pipette

'''data domain initialization'''
X = np.random.choice(np.linspace(0, 1, 10000), size=200, replace=False).reshape(-1, 1)
# y = np.sin(X/40) + np.random.normal(scale=0.02, size=X.shape)

# with plt.style.context('seaborn-white'):
#     plt.figure(figsize=(10, 5))
#     plt.scatter(X, y, c='k', s=20)
#     plt.title('sin(x) + noise')
#     plt.show()


def GP_regression_std(regressor, X):
    _, std = regressor.predict(X, return_std=True)
    query_idx = np.argmax(std)
    return query_idx, X[query_idx]


def plot(X, y, y_std, X_next_idx, X_training, y_training):
    with plt.style.context('seaborn-white'):
        plt.figure(figsize=(10, 5))
        plt.plot(X, y)
        plt.fill_between(X, y - y_std, y + y_std, alpha=0.2)
        # plt.scatter(X, y, c='k', s=20)
        '''Highlight observed points'''
        plt.scatter(X_training, y_training, c='green', s=100)
        plt.vlines(X[X_next_idx], 0, y[X_next_idx], colors="r", linestyles="dashed")
        plt.title('Initial prediction')
        plt.show()
        # plt.savefig('al_figures/initial')

# counter settings
well_counter = 0
tip_counter = 0


'''randomly choose initial value'''
n_initial = 3
for i in range(n_initial):
    initial_idx = np.random.choice(range(len(X)), size=1, replace=False)
    # X_initial, y_initial = X[initial_idx], y[initial_idx]
    if i == 0:
        # X_training, y_training = X[initial_idx], np.array([float(input(X[initial_idx]))])
        X_training, y_training = X[initial_idx], np.array([pipette.run(X[initial_idx], well_counter, tip_counter)])
        print(X_training, y_training)
        well_counter += 1
        tip_counter += 2
    else:
        # X_training, y_training = np.vstack((X_training, X[initial_idx])), np.vstack((y_training, np.array([float(input(X[initial_idx]))])))
        X_training, y_training = np.vstack((X_training, X[initial_idx])), np.vstack((y_training, np.array([pipette.run(X[initial_idx], well_counter, tip_counter)])))
        for j in range(len(X_training)):
            print(X_training[j], y_training[j], '\n')
        well_counter += 1
        tip_counter += 2
    # print(initial_idx, X_training, y_training, sep='\n')
# normalize y_training
y_ave = np.average(y_training)
y_training = y_training / y_ave



kernel = RBF(length_scale=0.05, length_scale_bounds=(1e-2, 1e3)) \
         # + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))

regressor = ActiveLearner(
    estimator=GaussianProcessRegressor(kernel=kernel),
    query_strategy=GP_regression_std,
    X_training=X_training.reshape(-1, 1), y_training=y_training.reshape(-1, 1)
)

# create plot points
X_grid = np.linspace(0, 1, 1000)
y_pred, y_std = regressor.predict(X_grid.reshape(-1, 1), return_std=True)
y_pred, y_std = y_pred.ravel(), y_std.ravel()
# draw the initial plot
with plt.style.context('seaborn-white'):
    plt.figure(figsize=(10, 5))
    plt.plot(X_grid, y_pred)
    plt.fill_between(X_grid, y_pred - y_std, y_pred + y_std, alpha=0.2)
    # plt.scatter(X, y, c='k', s=20)
    '''Highlight observed points'''
    plt.scatter(X_training, y_training, c='green', s=100)
    plt.title('Initial prediction')
    # plt.show()
    plt.savefig('al_figures/initial')

# active learning
n_queries = 3
for idx in range(n_queries):
    query_idx, query_instance = regressor.query(X)
    '''take in observation value y_input'''
    # y_input = np.array([float(input(query_instance))]) / y_ave
    print("$$$", "Observing point of X={}".format(X[query_idx]), "$$$", sep='\n')
    y_input = np.array([pipette.run(X[query_idx], well_counter, tip_counter)]) / y_ave
    print(X[query_idx], y_input * y_ave)
    well_counter += 1
    tip_counter += 2
    # regressor.teach(X[query_idx].reshape(1, -1), y[query_idx].reshape(1, -1))
    regressor.teach(X[query_idx].reshape(1, -1), y_input.reshape(1, -1))
    '''update training set'''
    (X_training, y_training) = (np.vstack((X_training, X[query_idx])), np.vstack((y_training, y_input)))

    y_pred_final, y_std_final = regressor.predict(X_grid.reshape(-1, 1), return_std=True)
    y_pred_final, y_std_final = y_pred_final.ravel(), y_std_final.ravel()

    with plt.style.context('seaborn-white'):
        plt.figure(figsize=(10, 8))
        plt.plot(X_grid, y_pred_final)
        plt.fill_between(X_grid, y_pred_final - y_std_final, y_pred_final + y_std_final, alpha=0.2)
        # plt.scatter(X, y, c='k', s=20)
        '''Highlight observed points '''
        plt.scatter(X_training, y_training, c='green', s=100)
        plt.title('Prediction after active learning')
        # plt.show()
        plt.savefig('al_figures/{}'.format(idx))

#output the result
y_pred, y_std = regressor.predict(X_grid.reshape(-1, 1), return_std=True)
print(np.argmax(y_pred) / 1000, np.max(y_pred) * y_ave)
