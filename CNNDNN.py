from keras.layers import Input,Dropout, Flatten,Dense
from sklearn.model_selection import StratifiedKFold,train_test_split  
import numpy,math
from sklearn.metrics import roc_curve,auc
from keras.models import Model
from keras.utils import plot_model
from keras import initializers,regularizers,optimizers
from keras.layers.convolutional import Conv1D
import matplotlib.pyplot as plt
from keras.callbacks import LearningRateScheduler
from sklearn.metrics import classification_report, confusion_matrix 
epochs = 20


# fix random seed for reproducibility
numpy.random.seed(1)
# load Clinical dataset
dataset_clinical = numpy.loadtxt("/home/nikhil/Desktop/Project/nik/Data/METABRIC_clinical_1980.txt", delimiter=" ")
#dataset_clinical = numpy.loadtxt("/home/nikhil/Desktop/Project/MDNNMD/data/Stratified Data/Clinical.txt", delimiter=" ")

# split into input (X) and output (Y) variables
X_clinical = dataset_clinical[:,0:25]
Y_clinical = dataset_clinical[:,25]
kfold = StratifiedKFold(n_splits=10, shuffle=False, random_state=1)
cvscores_clinical = []
i=1
for train_index, test_index in kfold.split(X_clinical, Y_clinical):
	print(i,"th Fold *****************************************")
	i=i+1
	x_train_clinical, x_test_clinical=X_clinical[train_index],X_clinical[test_index]	
	y_train_clinical, y_test_clinical = Y_clinical[train_index],Y_clinical[test_index] 	
	x_train_clinical = numpy.expand_dims(x_train_clinical, axis=2)
	x_test_clinical = numpy.expand_dims(x_test_clinical, axis=2)
	# first Clinical CNN Model
	init =initializers.glorot_normal(seed=1)
	bias_init =initializers.Constant(value=0.1)
	main_input1 = Input(shape=(25,1),name='Input')
	conv1 = Conv1D(filters=25,kernel_size=15,strides=2,activation='tanh',padding='same',name='Conv1D',kernel_initializer=init,bias_initializer=bias_init)(main_input1)
	flat1 = Flatten(name='Flatten')(conv1)
	dense1 = Dense(150,activation='tanh',name='dense1',kernel_initializer=init,bias_initializer=bias_init)(flat1)
	output = Dense(1, activation='sigmoid',name='output',activity_regularizer= regularizers.l2(0.01),kernel_initializer=init,bias_initializer=bias_init)(dense1)
	clinical_model = Model(inputs=main_input1, outputs=output)
	# summarize layers
	#print(model.summary())
	# plot graph
	#plot_model(model, to_file='/home/nikhil/Desktop/Project/nik/Code/Submodels/Model Design/CNN Clinical.png')
	clinical_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	x_train, x_val, y_train, y_val = train_test_split(x_train_clinical, y_train_clinical, test_size=0.2,stratify=y_train_clinical)
	clinical_model.fit(x_train, y_train, epochs=epochs, batch_size=8,verbose=2,validation_data=(x_val,y_val))	

	clinical_scores = clinical_model.evaluate(x_test_clinical, y_test_clinical,verbose=2)
	print("%s: %.2f%%" % (clinical_model.metrics_names[1], clinical_scores[1]*100))
	cvscores_clinical.append(clinical_scores[1] * 100)
print("Accuracy = %.2f%% (+/- %.2f%%)\n" % (numpy.mean(cvscores_clinical), numpy.std(cvscores_clinical)))

# load CNV dataset
dataset_cnv = numpy.loadtxt("/home/nikhil/Desktop/Project/nik/Data/METABRIC_cnv_1980.txt", delimiter=" ")
#dataset_cnv = numpy.loadtxt("/home/nikhil/Desktop/Project/MDNNMD/data/cnv.txt", delimiter=",")
# split into input (X) and output (Y) variables
X_cnv = dataset_cnv[:,0:200]
Y_cnv = dataset_cnv[:,200]
kfold = StratifiedKFold(n_splits=10, shuffle=False, random_state=1)
cvscores_cnv = []
i=1
for train_index, test_index in kfold.split(X_cnv, Y_cnv):
	print(i,'th Fold *******************************')
	i=i+1
	#Spliting the data set into training and testing
	x_train_cnv, x_test_cnv=X_cnv[train_index],X_cnv[test_index]	
	y_train_cnv, y_test_cnv = Y_cnv[train_index],Y_cnv[test_index]
	x_train_cnv = numpy.expand_dims(x_train_cnv, axis=2)
	x_test_cnv = numpy.expand_dims(x_test_cnv, axis=2)
	
	# first CNV Model
	#init=initializers.TruncatedNormal(mean=0.0, stddev=0.01, seed=None)
	init=initializers.glorot_normal(seed=1)
	bias_init =initializers.Constant(value=0.1)
	main_input1 = Input(shape=(200,1))
	conv1 = Conv1D(filters=4,kernel_size=15,strides=2,activation='tanh',padding='same',kernel_initializer=init,bias_initializer=bias_init)(main_input1)
	flat1 = Flatten()(conv1)
	dropout1 = Dropout(0.50)(flat1)
	dense1 = Dense(150,activation='tanh',name='dense1',kernel_initializer=init,bias_initializer=bias_init)(dropout1)
	dropout2 = Dropout(0.25,name='dropout2')(dense1)
	output = Dense(1, activation='sigmoid',activity_regularizer= regularizers.l2(0.01),kernel_initializer=init,bias_initializer=bias_init)(dropout2)
	cnv_model =	Model(inputs=main_input1, outputs=output)
	# summarize layers
	#print(model.summary())
	# plot graph
	#plot_model(model, to_file='/home/nikhil/Desktop/Project/nik/Code/Submodels/Model Design/CNN CNV.png')
	cnv_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	x_train, x_val, y_train, y_val = train_test_split(x_train_cnv, y_train_cnv, test_size=0.2,stratify=y_train_cnv)
	cnv_model.fit(x_train_cnv, y_train_cnv, epochs=epochs,validation_data=(x_val,y_val), batch_size=8,verbose=2)
	cnv_scores = cnv_model.evaluate(x_test_cnv, y_test_cnv,verbose=2)
	print("%s: %.2f%%" % (cnv_model.metrics_names[1], cnv_scores[1]*100))
	cvscores_cnv.append(cnv_scores[1] * 100)	
print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores_cnv), numpy.std(cvscores_cnv)))


dataset_exp = numpy.loadtxt("/home/nikhil/Desktop/Project/nik/Data/METABRIC_gene_exp_1980.txt", delimiter=" ")
# split into input (X) and output (Y) variables
X_exp = dataset_exp[:,0:400]
Y_exp = dataset_exp[:,400]
#Spliting the data set into training and testing
kfold = StratifiedKFold(n_splits=10, shuffle=False, random_state=1)
cvscores_exp = []
i=1
for train_index, test_index in kfold.split(X_exp, Y_exp):
	print(i,'th Fold *******************************')
	i=i+1
	#Spliting the data set into training and testing
	x_train_exp, x_test_exp=X_exp[train_index],X_exp[test_index]	
	y_train_exp, y_test_exp = Y_exp[train_index],Y_exp[test_index] 
	x_train_exp = numpy.expand_dims(x_train_exp, axis=2)
	x_test_exp = numpy.expand_dims(x_test_exp, axis=2)
	
	# first CNN EXP Model
	init=initializers.glorot_normal(seed=1)
	bias_init =initializers.Constant(value=0.1)
	main_input1 = Input(shape=(400,1))
	conv1 = Conv1D(filters=4,kernel_size=15,strides=2,activation='tanh',padding='same',kernel_initializer=init,bias_initializer=bias_init)(main_input1)
	flat1 = Flatten()(conv1)
	dropout1 = Dropout(0.50)(flat1)
	dense1 = Dense(150,activation='tanh',name='dense1',kernel_initializer=init,bias_initializer=bias_init)(dropout1)
	dropout2 = Dropout(0.25,name='dropout2')(dense1)
	output = Dense(1, activation='sigmoid',activity_regularizer= regularizers.l2(0.01),kernel_initializer=init,bias_initializer=bias_init)(dropout2)
	exp_model =	Model(inputs=main_input1, outputs=output)
	# summarize layers
	#print(model.summary())
	# plot graph
	#plot_model(model, to_file='/home/nikhil/Desktop/Project/nik/Code/Submodels/Model Design/CNN Gene Exp.png')
	exp_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	x_train, x_val, y_train, y_val = train_test_split(x_train_exp, y_train_exp, test_size=0.2,stratify=y_train_exp)	
	exp_model.fit(x_train, y_train, epochs=epochs, batch_size=8,verbose=2,validation_data=(x_val,y_val))
	exp_scores = exp_model.evaluate(x_test_exp, y_test_exp,verbose=2)
	print("%s: %.2f%%" % (exp_model.metrics_names[1], exp_scores[1]*100))
	cvscores_exp.append(exp_scores[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores_exp), numpy.std(cvscores_exp)))
i=1
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
	model_dnnclinical = Model(inputs = main_input1, outputs = output )
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
	model_dnnclinical.compile(loss='binary_crossentropy', optimizer=adams, metrics=['accuracy'])
	x_train, x_val, y_train, y_val = train_test_split(x_train_clinical, y_train_clinical, test_size=0.2,stratify = y_train_clinical)
	# Fit the model
	model_dnnclinical.fit(x_train, y_train, epochs=40, batch_size=64,verbose=2,validation_data=(x_val,y_val),callbacks=[lrate])
	# evaluate the model
	clinical_scores = model_dnnclinical.evaluate(x_test_clinical, y_test_clinical, verbose=0)
	print("%s: %.2f%%" % (model_dnnclinical.metrics_names[1], clinical_scores[1]*100))
	
i=1
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
	model_dnncnv = Model(inputs = main_input1, outputs = output )
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

	model_dnncnv.compile(loss='binary_crossentropy', optimizer=adams, metrics=['accuracy'])
	x_train, x_val, y_train, y_val = train_test_split(x_train_cnv, y_train_cnv, test_size=0.2,stratify = y_train_cnv)
	# Fit the model
	model_dnncnv.fit(x_train, y_train, epochs=epochs, batch_size=64,verbose=2,validation_data=(x_val,y_val),callbacks=[lrate]	)
	#model_cnv.fit(x_train_cnv, y_train_cnv, epochs=epochs, batch_size=8,verbose=2,validation_split=0.20)
	# evaluate the model
	cnv_scores = model_dnncnv.evaluate(x_test_cnv, y_test_cnv,verbose=0)
	print("%s: %.2f%%" % (model_dnncnv.metrics_names[1], cnv_scores[1]*100))
i=1
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
	model_dnnexp = Model(inputs = main_input1, outputs = output )
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
	model_dnnexp.compile(loss='binary_crossentropy', optimizer=adams, metrics=['accuracy'])
	
	x_train, x_val, y_train, y_val = train_test_split(x_train_exp, y_train_exp, test_size=0.2,stratify = y_train_exp)
	# Fit the model
	model_dnnexp.fit(x_train, y_train, epochs=epochs, batch_size=64,verbose=2,validation_data=(x_val,y_val),callbacks=[lrate])
	# evaluate the model
	exp_scores = model_dnnexp.evaluate(x_test_exp, y_test_exp,verbose=0)
	print("%s: %.2f%%" % (model_dnnexp.metrics_names[1], exp_scores[1]*100))




#Plotting
X_train_clinical, X_test_clinical, y_train_clinical, y_test_clinical = train_test_split(X_clinical, Y_clinical, test_size=0.2,stratify=Y_clinical)
X_train_cnv, X_test_cnv, y_train_cnv, y_test_cnv = train_test_split(X_cnv, Y_cnv, test_size=0.2,stratify=Y_cnv)
X_train_exp, X_test_exp, y_train_exp, y_test_exp = train_test_split(X_exp, Y_exp, test_size=0.2,stratify=Y_exp)
pred_dnnclinical = model_dnnclinical.predict(X_test_clinical)
pred_dnncnv = model_dnncnv.predict(X_test_cnv)
pred_dnnexp = model_dnnexp.predict(X_test_exp)
X_test_clinical=numpy.expand_dims(X_test_clinical, axis=2)
X_test_cnv=numpy.expand_dims(X_test_cnv, axis=2)
X_test_exp=numpy.expand_dims(X_test_exp, axis=2)
pred_clinical = clinical_model.predict(X_test_clinical)
pred_cnv = cnv_model.predict(X_test_cnv)
pred_exp = exp_model.predict(X_test_exp)
fpr_clinical, tpr_clinical, threshold_clinical = roc_curve(y_test_clinical, pred_clinical,pos_label=1)
fpr_cnv, tpr_cnv, threshold_cnv = roc_curve(y_test_cnv, pred_cnv,pos_label=1)
fpr_exp, tpr_exp, threshold_exp = roc_curve(y_test_exp, pred_exp,pos_label=1)
fpr_dnnclinical, tpr_dnnclinical, threshold_dnnclinical = roc_curve(y_test_clinical, pred_dnnclinical,pos_label=1)
fpr_dnncnv, tpr_dnncnv, threshold_dnncnv = roc_curve(y_test_cnv, pred_dnncnv,pos_label=1)
fpr_dnnexp, tpr_dnnexp, threshold_dnnexp = roc_curve(y_test_exp, pred_dnnexp,pos_label=1)
roc_auc_dnnclinical = auc(fpr_dnnclinical, tpr_dnnclinical)
roc_auc_dnncnv = auc(fpr_dnncnv, tpr_dnncnv)
roc_auc_dnnexp = auc(fpr_dnnexp, tpr_dnnexp)
roc_auc_clinical = auc(fpr_clinical, tpr_clinical)
roc_auc_cnv = auc(fpr_cnv, tpr_cnv)
roc_auc_exp = auc(fpr_exp, tpr_exp)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr_clinical, tpr_clinical, 'r', label = 'CNN-Clinical = %0.2f' % roc_auc_clinical)
plt.plot(fpr_cnv, tpr_cnv, 'b', label = 'CNN-CNA = %0.2f' % roc_auc_cnv)
plt.plot(fpr_exp, tpr_exp, 'g', label = 'CNN-Expr = %0.2f' % roc_auc_exp)
plt.plot(fpr_dnnclinical, tpr_dnnclinical, 'y', label = 'DNN-Clinical = %0.2f' % roc_auc_dnnclinical)
plt.plot(fpr_dnncnv, tpr_dnncnv, 'o', label = 'DNN-CNA = %0.2f' % roc_auc_dnncnv)
plt.plot(fpr_dnnexp, tpr_dnnexp, 'c', label = 'DNN-Expr = %0.2f' % roc_auc_dnnexp)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate(Sn)')
plt.xlabel('Flase Positive Rate(1-Sp)')
plt.show()




