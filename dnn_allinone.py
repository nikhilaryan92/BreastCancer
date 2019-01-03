from keras.layers import Dense,Dropout,Input
from sklearn.model_selection import train_test_split,StratifiedKFold  
import numpy,math
from sklearn.metrics import roc_curve,auc
from keras.models import Model
from keras.utils import plot_model
from keras import regularizers,initializers,optimizers
import matplotlib.pyplot as plt
from keras.callbacks import LearningRateScheduler
from sklearn.metrics import classification_report, confusion_matrix 

# fix random seed for reproducibility
numpy.random.seed(1)
# load Clinical dataset
dataset_clinical = numpy.loadtxt("/home/nikhil/Desktop/Project/nik/Data/METABRIC_clinical_1980.txt", delimiter=" ")

# split into input (X) and output (Y) variables
X_clinical = dataset_clinical[:,0:25]
Y_clinical = dataset_clinical[:,25]
kfold = StratifiedKFold(n_splits=10, shuffle=False, random_state=1)
cvscores_clinical = []
i=1
epochs = 20

for train_index, test_index in kfold.split(X_clinical, Y_clinical):
	print(i,"th Fold *****************************************")
	i=i+1
	x_train_clinical, x_test_clinical=X_clinical[train_index],X_clinical[test_index]	
	y_train_clinical, y_test_clinical = Y_clinical[train_index],Y_clinical[test_index] 

	# create model
	#init =initializers.glorot_normal(seed=1)
	bias_init =initializers.Constant(value=0.1)
	main_input1 = Input(shape=(25,),name='input')
	init = initializers.TruncatedNormal(mean=0.0, stddev=1.0 / math.sqrt(float(25)/2), seed=1)
	dense1 = Dense(1000,activation='tanh',name='dense1',kernel_initializer=init,bias_initializer=bias_init)(main_input1)
	init = initializers.TruncatedNormal(mean=0.0, stddev=1.0 / math.sqrt(float(1000)/2), seed=1)
	dense2 = Dense(1000,activation='tanh',name='dense2',kernel_initializer=init,bias_initializer=bias_init)(dense1)
	init = initializers.TruncatedNormal(mean=0.0, stddev=1.0 / math.sqrt(float(1000)/2), seed=1)
	dense3 = Dense(1000,activation='tanh',name='dense3',kernel_initializer=init,bias_initializer=bias_init)(dense2)
	init = initializers.TruncatedNormal(mean=0.0, stddev=1.0 / math.sqrt(float(1000)/2), seed=1)
	dropout = Dropout(0.5,name='dropout')(dense3)
	dense4 = Dense(100,activation='tanh',name='dense4',kernel_initializer=init,bias_initializer=bias_init)(dropout)
	init = initializers.TruncatedNormal(mean=0.0, stddev=1.0 / math.sqrt(float(100)/2), seed=1)
	output = Dense(1,name='output',activation='sigmoid',activity_regularizer= regularizers.l2(0.01),kernel_initializer=init,bias_initializer=bias_init)(dense4)
	model_clinical = Model(inputs = main_input1, outputs = output )
	# summarize layers
	#print(model_clinical.summary())
	# plot Model
	#plot_model(model_clinical, to_file='/home/nikhil/Desktop/Project/nik/Code/Submodels/Model Design/DNN Clinical.png')
	# Compile model

	def exp_decay(epoch):
		initial_lrate = 0.001
		k = 0.1
		lrate = initial_lrate * math.exp(-k*epoch)
		return lrate
	lrate = LearningRateScheduler(exp_decay)

	adams=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	model_clinical.compile(loss='binary_crossentropy', optimizer=adams, metrics=['accuracy'])
	x_train, x_val, y_train, y_val = train_test_split(x_train_clinical, y_train_clinical, test_size=0.2,stratify = y_train_clinical)
	# Fit the model
	model_clinical.fit(x_train, y_train, epochs=40, batch_size=64,verbose=2,validation_data=(x_val,y_val),callbacks=[lrate])
	# evaluate the model
	clinical_scores = model_clinical.evaluate(x_test_clinical, y_test_clinical, verbose=0)
	print("%s: %.2f%%" % (model_clinical.metrics_names[1], clinical_scores[1]*100))
	cvscores_clinical.append(clinical_scores[1] * 100)	
	intermediate_layer_model = Model(inputs=main_input1,outputs=dense4)
print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores_clinical), numpy.std(cvscores_clinical)))



# for extracting final layer features 
y_pred_clinical =  model_clinical.predict(X_clinical)
# for extracting one layer before final layer features
#y_pred_clinical = intermediate_layer_model.predict(X_clinical)




# load cnv dataset
dataset_cnv = numpy.loadtxt("/home/nikhil/Desktop/Project/nik/Data/METABRIC_cnv_1980.txt", delimiter=" ")

# split into input (X) and output (Y) variables
X_cnv = dataset_cnv[:,0:200]
Y_cnv = dataset_cnv[:,200]


kfold = StratifiedKFold(n_splits=10, shuffle=False, random_state=1)
cvscores_cnv = []
i=1
epochs = 20
for train_index, test_index in kfold.split(X_cnv, Y_cnv):
	print(i,"th Fold *****************************************")
	i=i+1

	x_train_cnv, x_test_cnv=X_cnv[train_index],X_cnv[test_index]	
	y_train_cnv, y_test_cnv = Y_cnv[train_index],Y_cnv[test_index] 

	# create model
	#init =initializers.glorot_normal(seed=1)
	bias_init =initializers.Constant(value=0.1)
	main_input1 = Input(shape=(200,),name='input')
	init = initializers.TruncatedNormal(mean=0.0, stddev=1.0 / math.sqrt(float(200)/2), seed=1)
	dense1 = Dense(1000,activation='tanh',name='dense1',kernel_initializer=init,bias_initializer=bias_init)(main_input1)
	init = initializers.TruncatedNormal(mean=0.0, stddev=1.0 / math.sqrt(float(1000)/2), seed=1)
	dense2 = Dense(500,activation='tanh',name='dense2',kernel_initializer=init,bias_initializer=bias_init)(dense1)
	init = initializers.TruncatedNormal(mean=0.0, stddev=1.0 / math.sqrt(float(500)/2), seed=1)
	dense3 = Dense(500,activation='tanh',name='dense3',kernel_initializer=init,bias_initializer=bias_init)(dense2)
	init = initializers.TruncatedNormal(mean=0.0, stddev=1.0 / math.sqrt(float(500)/2), seed=1)
	dropout = Dropout(0.5,name='dropout')(dense3)
	dense4 = Dense(100,activation='tanh',name='dense4',kernel_initializer=init,bias_initializer=bias_init)(dropout)
	init = initializers.TruncatedNormal(mean=0.0, stddev=1.0 / math.sqrt(float(100)/2), seed=1)
	output = Dense(1,name='output',activation='sigmoid',activity_regularizer= regularizers.l2(0.01),kernel_initializer=init,bias_initializer=bias_init)(dense4)
	model_cnv = Model(inputs = main_input1, outputs = output )
	# summarize layers
	#print(model_cnv.summary())
	# plot Model
	#plot_model(model_cnv, to_file='/home/nikhil/Desktop/Project/nik/Code/Submodels/Model Design/DNN CNV.png')
	# Compile model

	def exp_decay(epoch):
		initial_lrate = 0.00001
		k = 0.1
		lrate = initial_lrate * math.exp(-k*epoch)
		return lrate
	lrate = LearningRateScheduler(exp_decay)
	adams=optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

	model_cnv.compile(loss='binary_crossentropy', optimizer=adams, metrics=['accuracy'])
	x_train, x_val, y_train, y_val = train_test_split(x_train_cnv, y_train_cnv, test_size=0.2,stratify = y_train_cnv)
	# Fit the model
	model_cnv.fit(x_train, y_train, epochs=epochs, batch_size=64,verbose=2,validation_data=(x_val,y_val),callbacks=[lrate]	)
	#model_cnv.fit(x_train_cnv, y_train_cnv, epochs=epochs, batch_size=8,verbose=2,validation_split=0.20)
	# evaluate the model
	cnv_scores = model_cnv.evaluate(x_test_cnv, y_test_cnv,verbose=0)
	print("%s: %.2f%%" % (model_cnv.metrics_names[1], cnv_scores[1]*100))
	cvscores_cnv.append(cnv_scores[1] * 100)
	intermediate_layer_model = Model(inputs=main_input1,outputs=dense4)
print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores_cnv), numpy.std(cvscores_cnv)))




# for extracting final layer features 
y_pred_cnv =  model_cnv.predict(X_cnv)
# for extracting one layer before final layer features
#y_pred_cnv = intermediate_layer_model.predict(X_cnv)


# load Gene Exp dataset
dataset_exp = numpy.loadtxt("/home/nikhil/Desktop/Project/nik/Data/METABRIC_gene_exp_1980.txt", delimiter=" ")

# split into input (X) and output (Y) variables
X_exp = dataset_exp[:,0:400]
Y_exp = dataset_exp[:,400]
kfold = StratifiedKFold(n_splits=10, shuffle=False, random_state=1)
cvscores_exp = []
i=1
epochs = 20
for train_index, test_index in kfold.split(X_exp, Y_exp):
	print(i,"th Fold *****************************************")
	i=i+1

	x_train_exp, x_test_exp=X_exp[train_index],X_exp[test_index]	
	y_train_exp, y_test_exp = Y_exp[train_index],Y_exp[test_index] 

	# create model
	#init =initializers.glorot_normal(seed=1)
	bias_init =initializers.Constant(value=0.1)
	main_input1 = Input(shape=(400,),name='input')
	init = initializers.TruncatedNormal(mean=0.0, stddev=1.0 / math.sqrt(float(400)/2), seed=1)
	dense1 = Dense(1000,activation='tanh',name='dense1',kernel_initializer=init,bias_initializer=bias_init)(main_input1)
	init = initializers.TruncatedNormal(mean=0.0, stddev=1.0 / math.sqrt(float(1000)/2), seed=1)
	dense2 = Dense(500,activation='tanh',name='dense2',kernel_initializer=init,bias_initializer=bias_init)(dense1)
	init = initializers.TruncatedNormal(mean=0.0, stddev=1.0 / math.sqrt(float(500)/2), seed=1)
	dense3 = Dense(500,activation='tanh',name='dense3',kernel_initializer=init,bias_initializer=bias_init)(dense2)
	init = initializers.TruncatedNormal(mean=0.0, stddev=1.0 / math.sqrt(float(500)/2), seed=1)
	dropout = Dropout(0.5,name='dropout')(dense3)
	dense4 = Dense(100,activation='tanh',name='dense4',kernel_initializer=init,bias_initializer=bias_init)(dropout)
	init = initializers.TruncatedNormal(mean=0.0, stddev=1.0 / math.sqrt(float(100)/2), seed=1)
	output = Dense(1,name='output',activation='sigmoid',activity_regularizer= regularizers.l2(0.01),kernel_initializer=init,bias_initializer=bias_init)(dense4)
	model_exp = Model(inputs = main_input1, outputs = output )
	# summarize layers
	#model_exp.summary()
	# plot Model
	#plot_model(model_exp, to_file='/home/nikhil/Desktop/Project/nik/Code/Submodels/Model Design/DNN Gene.png')
	# Compile model

	def exp_decay(epoch):
		initial_lrate = 0.00001
		k = 0.1
		lrate = initial_lrate * math.exp(-k*epoch)
		return lrate
	lrate = LearningRateScheduler(exp_decay)

	adams=optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	model_exp.compile(loss='binary_crossentropy', optimizer=adams, metrics=['accuracy'])
	
	x_train, x_val, y_train, y_val = train_test_split(x_train_exp, y_train_exp, test_size=0.2,stratify = y_train_exp)
	# Fit the model
	model_exp.fit(x_train, y_train, epochs=epochs, batch_size=64,verbose=2,validation_data=(x_val,y_val),callbacks=[lrate])
	# evaluate the model
	exp_scores = model_exp.evaluate(x_test_exp, y_test_exp,verbose=0)
	print("%s: %.2f%%" % (model_exp.metrics_names[1], exp_scores[1]*100))
	cvscores_exp.append(exp_scores[1] * 100)
	intermediate_layer_model = Model(inputs=main_input1,outputs=dense4)
print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores_exp), numpy.std(cvscores_exp)))




# for extracting final layer features 
y_pred_exp =  model_exp.predict(X_exp)
# for extracting one layer before final layer features
#y_pred_exp = intermediate_layer_model.predict(X_exp)

#y_pred = numpy.concatenate((y_pred_clinical,y_pred_cnv,y_pred_exp),axis=1)

#predicted value multiplied by their respective AUC value
y_pred = numpy.concatenate((0.84*y_pred_clinical,0.70*y_pred_cnv,0.77*y_pred_exp),axis=1)


feature_target=numpy.concatenate((y_pred,Y_exp[:,None]),axis=1)
with open('/home/nikhil/Desktop/Project/nik/Code/Submodels/DNN/final_metadata.csv', 'w') as f:
	for item_exp in feature_target:
		for elem in item_exp:
			f.write(str(elem)+'\t')
		f.write('\n')


