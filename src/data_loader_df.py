import numpy as np
import pandas as pd
from sklearn import preprocessing

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

# load all the data from train & test file to get the same categorical columns
# then split the data to train_val and test
# apply one-hot and normalization
def load_all_data(train_file_path, test_file_path, valid_rate=0.1, is_df=True, norm=True):
	train_val = load_train_data_from_file(train_file_path)
	train_val_len = train_val.shape[0]
	test = load_test_data_from_file(test_file_path)

	dataset = pd.concat(objs=[train_val, test], axis=0)
	dataset.columns = COLUMNS
	if norm:
		dataset = normalize(dataset)
	
	dataset = pd.get_dummies(dataset, columns=CATEGORICAL_COLS)
	train_val = dataset[:train_val_len]
	test = dataset[train_val_len:]

	if is_df:
		train_df, valid_df = load_train_data(train_val, valid_rate, is_df, norm)
		test_df = load_test_data(test, is_df, norm)
		return train_df, valid_df, test_df

	train_X, train_Y, val_X, val_Y = load_train_data(train_val, valid_rate, is_df, norm)
	test_X, test_Y = load_test_data(test, is_df, norm)
	return train_X, train_Y, val_X, val_Y, test_X, test_Y


# split the data into training and validation set
def load_train_data(df, valid_rate=0.1, is_df=True):
	np.random.seed(11)
	# split the data_frame into training & validation set
	mask = np.random.rand(len(df)) < 1 - valid_rate
	train_df, valid_df = df[mask], df[~mask]
	if is_df:
		return train_df, valid_df

	train_labels = [1 if x == POS_STR else 0 for x in train_df['income']]
	valid_labels = [1 if x == POS_STR else 0 for x in valid_df['income']]
	train_df = train_df.drop('income', axis=1)
	valid_df = valid_df.drop('income', axis=1)

	return train_df.values, np.array(train_labels), valid_df.values, np.array(valid_labels)

# get test_X and test_Y
def load_test_data(df, is_df=True):
	if is_df:
		return df
	test_labels = [1 if x == POS_STR else 0 for x in df['income']]
	test_df = df.drop('income', axis=1)

	return test_df.values, np.array(test_labels)

# normalize the data using preprocessing.StandardScaler
def normalize(df):
	df.loc[:, COLS_TO_NORM] = preprocessing.StandardScaler().fit_transform(df[COLS_TO_NORM].values)
	return df












