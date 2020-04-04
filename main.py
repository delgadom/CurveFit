import copy
import curvefit
import pandas as pd
import pickle
import datetime
import numpy as np
from curvefit.model_generators import ModelPipeline, BasicModel
from curvefit.functions import expit, log_expit, identity, exponential

def return_se_as_f_of_t(t):
    return np.random.normal(scale=0.01)

dataset = pd.read_csv('dataset_for_CurveFit.csv')
dataset["Date"] = pd.to_datetime(dataset["Date"], infer_datetime_format=True, utc=True).astype(np.int64)
dataset.sort_values(by=["Date"])

dataset["SE"] = np.random.normal(scale=0.01, size=dataset.shape[0])
print(dataset.columns)
print(dataset)
model = BasicModel(all_data=dataset, col_t="Date", col_obs="Deaths", col_group="State/UnionTerritory",
                 col_obs_compare="Confirmed", all_cov_names=["DaysCovariate", "DaysCovariate", "DaysCovariate"], fun=expit, predict_space=expit, fit_dict={'fe_init': np.random.normal(scale=0.5, size=3)},  basic_model_dict={'col_obs_se': "SE", 'col_covs': [["DaysCovariate"], ["DaysCovariate"], ["DaysCovariate"]], 'param_names': ['alphalink', 'betalink', 'gammalink'], 'link_fun': [exponential, identity, exponential], 'var_link_fun': [exponential, identity, exponential]}, obs_se_func=return_se_as_f_of_t)
model.setup_pipeline()

print("Model setup. Running fit...")
model.fit(dataset)

print("Model fitted. Saving model...")

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved.")

model.predict()