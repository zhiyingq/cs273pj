import numpy as np
import pandas as pd
from sklearn import preprocessing, metrics
import warnings
warnings.filterwarnings("ignore")

POS_STR = ' >50K'
COLUMNS = [
		"age", "work", "fnlwgt", "edu", "edunum",
		"mstatus", "occ", "relation", "race", "sex",
		"cgain", "closs", "hpw", "nation", "income"
]
COLS_TO_NORM = ['age', 'fnlwgt', 'edunum', 'cgain', 'closs', 'hpw']
CATEGORICAL_COLS = [
		"work", "edu", "mstatus", "occ", "relation", "race", "sex", "nation"
	]

# load the data from train file
# remove the rows with missing values
def load_train_data_from_file(train_file_path):
	df = pd.read_csv(train_file_path, header=None).sample(frac=1, random_state=11)
	df.replace(' ?', np.nan, inplace=True)
	df = df.dropna()
	return df

# load the data from test file
# remove the rows with missing values
def load_test_data_from_file(test_file_path):
	df = pd.read_csv(test_file_path, header=None, skiprows=1)
	df.replace(' ?', np.nan, inplace=True)
	df = df.dropna()
	return df

'''
	@params:
		train_file_path: the path for the training file
		test_file_path: the path for the test file
		valid_rate: the proportion of validation data v.s all the training data
		is_df: return dataframe or not (nparray)
		norm: normalize the data or not
		one_hot: apply one-hot or not

	@return:
		if is_df is set to true, return: train_df, valid_df, test_df
		else return: train_X, train_Y, valid_X, valid_Y, test_X, test_Y
'''
def load_all_data(train_file_path, test_file_path, valid_rate=0.1, is_df=True, norm=False, one_hot=False):
	train_val = load_train_data_from_file(train_file_path)
	train_val_len = train_val.shape[0]
	test = load_test_data_from_file(test_file_path)

	dataset = pd.concat(objs=[train_val, test], axis=0)
	dataset.columns = COLUMNS

	if one_hot:
		dataset = pd.get_dummies(dataset, columns=CATEGORICAL_COLS)

	train_val = dataset[:train_val_len]
	test = dataset[train_val_len:]

	train_df, valid_df = split_train_data(train_val, valid_rate)
	test_df = test

	if norm:
		scaler = preprocessing.StandardScaler()

		train_df.loc[:, COLS_TO_NORM] = scaler.fit_transform(train_df[COLS_TO_NORM].values);
		valid_df.loc[:, COLS_TO_NORM] = scaler.transform(valid_df[COLS_TO_NORM].values);
		test_df.loc[:, COLS_TO_NORM] = scaler.transform(test_df[COLS_TO_NORM].values);

	if is_df:
		return train_df, valid_df, test_df

	train_Y = [1 if x == POS_STR else 0 for x in train_df['income']]
	valid_Y = [1 if x == POS_STR else 0 for x in valid_df['income']]
	test_Y = [1 if x == ' >50K.' else 0 for x in test_df['income']]
	train_df = train_df.drop('income', axis=1)
	valid_df = valid_df.drop('income', axis=1)
	test_df = test_df.drop('income', axis=1)

	return train_df.values, np.array(train_Y), valid_df.values, np.array(valid_Y), test_df.values, np.array(test_Y)

# split the data into training and validation set
# return train_df, valid_df
def split_train_data(df, valid_rate=0.1):
	np.random.seed(11)
	# split the data_frame into training & validation set
	mask = np.random.rand(len(df)) < 1 - valid_rate
	train_df, valid_df = df[mask], df[~mask]

	return train_df, valid_df

# visualize the TN/TP/FN/FP rate, return a dataframe
# Y_true: the true label
# Y_pred: the prediction
def visualize_confusion_matrix(Y_true, Y_pred):
    cm = metrics.confusion_matrix(Y_true, Y_pred).astype(float)
    num_positive = np.sum(Y_true)
    num_negative = len(Y_true) - num_positive

    cm[0][0] /= num_negative # true negative rete
    cm[0][1] /= num_negative # false positive rate
    cm[1][0] /= num_positive # false negative rate
    cm[1][1] /= num_positive # true positive rate

    df = pd.DataFrame(
        cm,
        index=["<= 50K", "> 50K"],
        columns=["<= 50K", "> 50K"]
    )

    df.index.names = ["Actual"]
    df.columns.names = ["Predicted"]
    return df
