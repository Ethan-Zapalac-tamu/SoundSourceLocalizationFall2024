from sklearn import linear_model
import pickle


# use OLS model to better predict the direction of sound


# get x and y data points from the hardware data


def generate_model(data_points_x, expected_outputs_y):
    ols_model = linear_model.LinearRegression()
    ols_model.fit(data_points_x, expected_outputs_y)
    print(f"regression coefficients = {ols_model.coef_}")
    # save model using pickle
    with open('model.pkl', 'wb') as f:
        pickle.dump(ols_model, f)
    # load model using this
    # with open('model.pkl', 'rb') as f:
    #     clf2 = pickle.load(f)
