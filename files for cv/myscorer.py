import numpy as np
import joblib

def my_custom_scorer(estimator, X, y):
    probas = estimator.predict_proba(X)
    error = y - probas[:, 1]
    mesqerror = np.mean(np.square(error))
    rmse = mesqerror ** 0.5
    if isinstance(rmse, np.memmap):
    	rmse = float(rmse)
    # This part is very hacky
    filename = 'Pickles/'
    if X.shape[1] == 217691:
    	filename += 'Sparse_Basic+Temporal+Recency/ProbasOnValidation/'
    elif X.shape[1] == 28:
    	filename += 'Condensed_Basic+Temporal+Recency/ProbasOnValidation/'
    else:
    	print('Unknown feature set')
    filename += str(rmse) + '.pkl'
    joblib.dump((estimator, probas[:, 1]), filename, compress=3)
    # /hacky part
    return -rmse

def my_custom_scorer_neural(estimator, X, y):
    probas = estimator.predict_proba(X)
    error = y - probas[:, 1]
    mesqerror = np.mean(np.square(error))
    rmse = mesqerror ** 0.5
    if isinstance(rmse, np.memmap):
    	rmse = float(rmse)
    # This part is very hacky
    filename = 'Pickles/'
    if X.shape[1] == 217691:
    	filename += 'Sparse_Basic+Temporal+Recency/ProbasOnValidation/'
    elif X.shape[1] == 28:
    	filename += 'Condensed_Basic+Temporal+Recency/ProbasOnValidation/'
    else:
    	print('Unknown feature set')
    filename += str(rmse) + 'nurl.pkl'
    joblib.dump((estimator.model.to_json(), probas[:, 1]), filename, compress=3)
    # /hacky part
    return -rmse

def basic_rms_scorer(estimator, X, y):
    probas = estimator.predict_proba(X)
    error = y - probas[:, 1]
    mesqerror = np.mean(np.square(error))
    rmse = mesqerror ** 0.5
    if isinstance(rmse, np.memmap):
    	rmse = float(rmse)
    return -rmse
