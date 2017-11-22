import pandas as pd
import numpy as np

def LoadFile(filedir, variables):
	print("Loading data...")
	return pd.read_csv(filedir, dtype='float64', names=variables)

def ProcessFrame(df, feature=None, result=None):
	if result is not None and feature is not None:
		df = df[feature == result]	
	return df.drop(df.columns[[0, -2, -1]], axis=1), df.Weight

def GetVariables(df):
	return list(df.columns.values)

def SplitData(df, percentsplit = 0.8, N_Max = None, N_Train = None):
	if N_Max is None:
		if percentsplit > 1 or percentsplit < 0:
			percentsplit = 0.8
	
		print("Splitting data, with percent split", percentsplit, "...")
		splitposition = int(len(df)*percentsplit)
		# return train, test
		return df[:splitposition], df[splitposition:]
		#Using the following return to limit use of data 
	return df[:N_Train], df[N_Train:N_Max]

def TrainandTarget(df, names):
	# return x, y
	print("Splitting data into Training and Target data...")
	return np.array(df[names[1:]]), np.array(df["signal"])
