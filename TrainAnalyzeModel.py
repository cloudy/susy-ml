import theano
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, LSTM, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.backend import backend
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.wrappers import TimeDistributed

def BuildTrainSeqModel(ModelIndex, Epochs, TrainX, TrainY, TestX, TestY):
	print("Building model...")
	model = BuildController(ModelIndex, TrainX.shape)
	print("Training model...")
	history=model.fit(TrainX, TrainY, validation_data=(TestX,TestY), nb_epoch=Epochs, batch_size=2048)
	return model, history

def BuildController(MInd, TrDimension):
	if MInd == 0:
		return ModelZero(TrDimension)
	if MInd == 1:
		return ModelOne(TrDimension)
	if MInd == 2:
		return ModelTwo(TrDimension)
	if MInd == 3:
		return ModelThree(TrDimension)

def ModelZero(TrDimension):
	print =("Model Zero selected...")
	model = Sequential()
	model.add(Dense(12, input_dim=TrDimension[1], init='uniform', activation='relu'))
	model.add(Dense(8, init='uniform', activation='relu'))
	model.add(Dense(1, init='uniform', activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.summary()
	return model

def ModelOne(TrDimension):
	print("Model One selected... MLP")
	model = Sequential()
	model.add(Dense(64, input_dim=TrDimension[1], init='uniform'))
	model.add(Dropout(0.5))
	model.add(Dense(64, init='uniform', activation='tanh'))
	model.add(Dropout(0.5))
	model.add(Dense(1, init='uniform', activation='softmax'))
	sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='binary_crossentropy', optimizer=sgd)
	model.summary()
	return model

def ModelTwo(TrDimension):
	print("Model Two selected... CNN")
	print("check this out", TrDimension[1])
	model = Sequential()
	model.add(Embedding(TrDimension[0], 50, input_length=None))
	model.add(Dropout(0.3))
	model.add(Conv1D(250, 3, activation='relu'))
	model.add(GlobalMaxPooling1D())
	model.add(Dense(250))
	model.add(Dropout(0.3))
	model.add(Activation('relu'))
	model.add(Dense(1))
	model.add(Activation('sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.summary()
	return model

def ModelThree(TrDimension):
	print("Model Three selected...")
	model = Sequential()
	model.add(Dense(1024, input_dim=TrDimension[1], init='uniform', activation='relu'))
	model.add(Dense(1024, init='uniform', activation='tanh'))
	model.add(Dense(1024, init='uniform', activation='tanh'))
	model.add(Dense(1024, init='uniform', activation='tanh'))
	model.add(Dense(1, init='uniform', activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.summary()
	return model
