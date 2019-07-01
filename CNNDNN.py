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
import keras.backend as K
import numpy as np

epochs = 20
epochs_dnn = 40
Sp_value = 0.99


def precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def sensitivity(y_true, y_pred):
    #y_pred = K.tf.convert_to_tensor(y_pred, np.float32) #Converting y_pred from numpy to tensor 
    #y_true = K.tf.convert_to_tensor(y_true, np.float32) #Converting y_true from numpy to tensor
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    Sn=(true_positives / (possible_positives + K.epsilon()))
    #with K.tf.Session() as sess: 	#Converting Sn 
        #Sn=sess.run(Sn)		# from tensor  to numpy
    return Sn

def specificity(y_true, y_pred):
    #y_pred = K.tf.convert_to_tensor(y_pred, np.float32) #Converting y_pred from numpy to tensor 
    #y_true = K.tf.convert_to_tensor(y_true, np.float32) #Converting y_true from numpy to tensor
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    Sp= (true_negatives / (possible_negatives + K.epsilon()))
    #with K.tf.Session() as sess: 	#Converting Sp 
       # Sp=sess.run(Sp)		# from tensor  to numpy
    return Sp

def sensitivity_at_specificity(specificity, **kwargs):
    def Sn(labels, predictions):
        # any tensorflow metric
        value, update_op = K.tf.metrics.sensitivity_at_specificity(labels, predictions, specificity, **kwargs)

        # find all variables created for this metric
        metric_vars = [i for i in K.tf.local_variables() if 'sensitivity_at_specificity' in i.name.split('/')[2]]

        # Add metric variables to GLOBAL_VARIABLES collection.
        # They will be initialized for new session.
        for v in metric_vars:
            K.tf.add_to_collection(K.tf.GraphKeys.GLOBAL_VARIABLES, v)

        # force to update metric values
        with K.tf.control_dependencies([update_op]):
            value = K.tf.identity(value)
            return value
    return Sn


def matthews_correlation(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())


# fix random seed for reproducibility
numpy.random.seed(1)

# load Clinical dataset
'''dataset_clinical = numpy.loadtxt("/home/nikhil/Desktop/Project/nik/Data/tcga_cln.csv",delimiter=",")
X_clinical = dataset_clinical[:,0:10]
Y_clinical = dataset_clinical[:,10]'''

dataset_clinical = numpy.loadtxt("/home/nikhil/Desktop/Project/nik/Data/METABRIC_clinical_1980.txt", delimiter=" ")#change the path to your local system	
# split into input (X) and output (Y) variables
X_clinical = dataset_clinical[:,0:25]
Y_clinical = dataset_clinical[:,25]

# load CNV dataset
#dataset_cnv = numpy.loadtxt("/home/nikhil/Desktop/Project/nik/Data/tcga_cna_200.csv",delimiter=",")
dataset_cnv = numpy.loadtxt("/home/nikhil/Desktop/Project/nik/Data/METABRIC_cnv_1980.txt", delimiter=" ") #change the path to your local system
# split into input (X) and output (Y) variables
X_cnv = dataset_cnv[:,0:200]
Y_cnv = dataset_cnv[:,200]

#dataset_exp = numpy.loadtxt("/home/nikhil/Desktop/Project/nik/Data/tcga_exp_400.csv",delimiter=",")
dataset_exp = numpy.loadtxt("/home/nikhil/Desktop/Project/nik/Data/METABRIC_gene_exp_1980.txt", delimiter=" ")#change the path to your local system
# split into input (X) and output (Y) variables
X_exp = dataset_exp[:,0:400]
Y_exp = dataset_exp[:,400]


print('Training the Clinical CNN')
kfold = StratifiedKFold(n_splits=10, shuffle=False, random_state=1)
acc_clinical = []
Pr_clinical = []
Sn_clinical = []
Mcc_clinical = []
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
	clinical_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy',precision,sensitivity_at_specificity(Sp_value), matthews_correlation])
	x_train, x_val, y_train, y_val = train_test_split(x_train_clinical, y_train_clinical, test_size=0.2,stratify=y_train_clinical)
	clinical_model.fit(x_train, y_train, epochs=epochs, batch_size=8,verbose=2,validation_data=(x_val,y_val))	

	clinical_scores = clinical_model.evaluate(x_test_clinical, y_test_clinical,verbose=2)
	print("%s: %.2f%%" % (clinical_model.metrics_names[1], clinical_scores[1]*100))
	print("%s: %.2f%%" % (clinical_model.metrics_names[2], clinical_scores[2]*100))
	print("%s: %.2f%%" % (clinical_model.metrics_names[3], clinical_scores[3]*100))
	print("%s: %.2f%%" % (clinical_model.metrics_names[4], clinical_scores[4]*100))
	acc_clinical.append(clinical_scores[1] * 100)
	Pr_clinical.append(clinical_scores[2] * 100)
	Sn_clinical.append(clinical_scores[3] * 100)
	Mcc_clinical.append(clinical_scores[4] * 100)
	intermediate_layer_clinical = Model(inputs=main_input1,outputs=dense1)
print("Accuracy = %.2f%% (+/- %.2f%%)\n" % (numpy.mean(acc_clinical), numpy.std(acc_clinical)))
print("Precision = %.2f%% (+/- %.2f%%)\n" % (numpy.mean(Pr_clinical), numpy.std(Pr_clinical)))
print("Sensitivity = %.2f%% (+/- %.2f%%)\n" % (numpy.mean(Sn_clinical), numpy.std(Sn_clinical)))
print("Mcc = %.2f%% (+/- %.2f%%)\n" % (numpy.mean(Mcc_clinical), numpy.std(Mcc_clinical)))

print('Training the CNA CNN')
kfold = StratifiedKFold(n_splits=10, shuffle=False, random_state=1)
acc_cnv = []
Pr_cnv = []
Sn_cnv = []
Mcc_cnv = []
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
	cnv_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy',precision,sensitivity_at_specificity(Sp_value), matthews_correlation])
	x_train, x_val, y_train, y_val = train_test_split(x_train_cnv, y_train_cnv, test_size=0.2,stratify=y_train_cnv)
	cnv_model.fit(x_train_cnv, y_train_cnv, epochs=epochs,validation_data=(x_val,y_val), batch_size=8,verbose=2)
	cnv_scores = cnv_model.evaluate(x_test_cnv, y_test_cnv,verbose=2)
	print("%s: %.2f%%" % (cnv_model.metrics_names[1], cnv_scores[1]*100))
	print("%s: %.2f%%" % (cnv_model.metrics_names[2], cnv_scores[2]*100))
	print("%s: %.2f%%" % (cnv_model.metrics_names[3], cnv_scores[3]*100))
	print("%s: %.2f%%" % (cnv_model.metrics_names[4], cnv_scores[4]*100))
	acc_cnv.append(cnv_scores[1] * 100)
	Pr_cnv.append(cnv_scores[2] * 100)
	Sn_cnv.append(cnv_scores[3] * 100)
	Mcc_cnv.append(cnv_scores[4] * 100)
	intermediate_layer_cnv = Model(inputs=main_input1,outputs=dropout2)
print("Accuracy = %.2f%% (+/- %.2f%%)\n" % (numpy.mean(acc_cnv), numpy.std(acc_cnv)))
print("Precision = %.2f%% (+/- %.2f%%)\n" % (numpy.mean(Pr_cnv), numpy.std(Pr_cnv)))
print("Sensitivity = %.2f%% (+/- %.2f%%)\n" % (numpy.mean(Sn_cnv), numpy.std(Sn_cnv)))
print("Mcc = %.2f%% (+/- %.2f%%)\n" % (numpy.mean(Mcc_cnv), numpy.std(Mcc_cnv)))



print('Training the Expr CNN')
kfold = StratifiedKFold(n_splits=10, shuffle=False, random_state=1)
acc_exp = []
Pr_exp = []
Sn_exp = []
Mcc_exp = []
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
	exp_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy',precision,sensitivity_at_specificity(Sp_value), matthews_correlation])
	x_train, x_val, y_train, y_val = train_test_split(x_train_exp, y_train_exp, test_size=0.2,stratify=y_train_exp)	
	exp_model.fit(x_train, y_train, epochs=epochs, batch_size=8,verbose=2,validation_data=(x_val,y_val))
	exp_scores = exp_model.evaluate(x_test_exp, y_test_exp,verbose=2)
	print("%s: %.2f%%" % (exp_model.metrics_names[1], exp_scores[1]*100))
	print("%s: %.2f%%" % (exp_model.metrics_names[2], exp_scores[2]*100))
	print("%s: %.2f%%" % (exp_model.metrics_names[3], exp_scores[3]*100))
	print("%s: %.2f%%" % (exp_model.metrics_names[4], exp_scores[4]*100))
	acc_exp.append(exp_scores[1] * 100)
	Pr_exp.append(exp_scores[2] * 100)
	Sn_exp.append(exp_scores[3] * 100)
	Mcc_exp.append(exp_scores[4] * 100)
	intermediate_layer_exp = Model(inputs=main_input1,outputs=dropout2)
print("Accuracy = %.2f%% (+/- %.2f%%)\n" % (numpy.mean(acc_exp), numpy.std(acc_exp)))
print("Precision = %.2f%% (+/- %.2f%%)\n" % (numpy.mean(Pr_exp), numpy.std(Pr_exp)))
print("Sensitivity = %.2f%% (+/- %.2f%%)\n" % (numpy.mean(Sn_exp), numpy.std(Sn_exp)))
print("Mcc = %.2f%% (+/- %.2f%%)\n" % (numpy.mean(Mcc_exp), numpy.std(Mcc_exp)))

print('Training the Clinical DNN')
i=1
acc_dnnclinical = []
Pr_dnnclinical = []
Sn_dnnclinical = []
Mcc_dnnclinical = []
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


	def exp_decay(epoch):
		initial_lrate = 0.001
		k = 0.1
		lrate = initial_lrate * math.exp(-k*epoch)
		return lrate
	lrate = LearningRateScheduler(exp_decay)

	adams=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	model_dnnclinical.compile(loss='binary_crossentropy', optimizer=adams, metrics=['accuracy',precision,sensitivity_at_specificity(Sp_value), matthews_correlation])
	x_train, x_val, y_train, y_val = train_test_split(x_train_clinical, y_train_clinical, test_size=0.2,stratify = y_train_clinical)
	# Fit the model
	model_dnnclinical.fit(x_train, y_train, epochs=epochs_dnn, batch_size=64,verbose=2,validation_data=(x_val,y_val),callbacks=[lrate])
	# evaluate the model
	clinical_dnnscores = model_dnnclinical.evaluate(x_test_clinical, y_test_clinical, verbose=0)
	print("%s: %.2f%%" % (model_dnnclinical.metrics_names[1], clinical_dnnscores[1]*100))
	print("%s: %.2f%%" % (model_dnnclinical.metrics_names[2], clinical_dnnscores[2]*100))
	print("%s: %.2f%%" % (model_dnnclinical.metrics_names[3], clinical_dnnscores[3]*100))
	print("%s: %.2f%%" % (model_dnnclinical.metrics_names[4], clinical_dnnscores[4]*100))
	acc_dnnclinical.append(clinical_dnnscores[1] * 100)
	Pr_dnnclinical.append(clinical_dnnscores[2] * 100)
	Sn_dnnclinical.append(clinical_dnnscores[3] * 100)
	Mcc_dnnclinical.append(clinical_dnnscores[4] * 100)
print("Accuracy = %.2f%% (+/- %.2f%%)\n" % (numpy.mean(acc_dnnclinical), numpy.std(acc_dnnclinical)))
print("Precision = %.2f%% (+/- %.2f%%)\n" % (numpy.mean(Pr_dnnclinical), numpy.std(Pr_dnnclinical)))
print("Sensitivity = %.2f%% (+/- %.2f%%)\n" % (numpy.mean(Sn_dnnclinical), numpy.std(Sn_dnnclinical)))
print("Mcc = %.2f%% (+/- %.2f%%)\n" % (numpy.mean(Mcc_dnnclinical), numpy.std(Mcc_dnnclinical)))

print('Training the CNA DNN')	
i=1
acc_dnncnv = []
Pr_dnncnv = []
Sn_dnncnv = []
Mcc_dnncnv = []
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


	def exp_decay(epoch):
		initial_lrate = 0.00001
		k = 0.1
		lrate = initial_lrate * math.exp(-k*epoch)
		return lrate
	lrate = LearningRateScheduler(exp_decay)
	adams=optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

	model_dnncnv.compile(loss='binary_crossentropy', optimizer=adams, metrics=['accuracy',precision,sensitivity_at_specificity(Sp_value), matthews_correlation])
	x_train, x_val, y_train, y_val = train_test_split(x_train_cnv, y_train_cnv, test_size=0.2,stratify = y_train_cnv)
	# Fit the model
	model_dnncnv.fit(x_train, y_train, epochs=epochs, batch_size=64,verbose=2,validation_data=(x_val,y_val),callbacks=[lrate]	)
	#model_cnv.fit(x_train_cnv, y_train_cnv, epochs=epochs, batch_size=8,verbose=2,validation_split=0.20)
	# evaluate the model
	cnv_dnnscores = model_dnncnv.evaluate(x_test_cnv, y_test_cnv,verbose=0)
	print("%s: %.2f%%" % (model_dnncnv.metrics_names[1], cnv_dnnscores[1]*100))
	print("%s: %.2f%%" % (model_dnncnv.metrics_names[2], cnv_dnnscores[2]*100))
	print("%s: %.2f%%" % (model_dnncnv.metrics_names[3], cnv_dnnscores[3]*100))
	print("%s: %.2f%%" % (model_dnncnv.metrics_names[4], cnv_dnnscores[4]*100))
	acc_dnncnv.append(cnv_dnnscores[1] * 100)
	Pr_dnncnv.append(cnv_dnnscores[2] * 100)
	Sn_dnncnv.append(cnv_dnnscores[3] * 100)
	Mcc_dnncnv.append(cnv_dnnscores[4] * 100)
print("Accuracy = %.2f%% (+/- %.2f%%)\n" % (numpy.mean(acc_dnncnv), numpy.std(acc_dnncnv)))
print("Precision = %.2f%% (+/- %.2f%%)\n" % (numpy.mean(Pr_dnncnv), numpy.std(Pr_dnncnv)))
print("Sensitivity = %.2f%% (+/- %.2f%%)\n" % (numpy.mean(Sn_dnncnv), numpy.std(Sn_dnncnv)))
print("Mcc = %.2f%% (+/- %.2f%%)\n" % (numpy.mean(Mcc_dnncnv), numpy.std(Mcc_dnncnv)))

print('Training the Expr DNN')
i=1
acc_dnnexp = []
Pr_dnnexp = []
Sn_dnnexp = []
Mcc_dnnexp = []
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

	def exp_decay(epoch):
		initial_lrate = 0.00001
		k = 0.1
		lrate = initial_lrate * math.exp(-k*epoch)
		return lrate
	lrate = LearningRateScheduler(exp_decay)

	adams=optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	model_dnnexp.compile(loss='binary_crossentropy', optimizer=adams, metrics=['accuracy',precision,sensitivity_at_specificity(Sp_value), matthews_correlation])
	
	x_train, x_val, y_train, y_val = train_test_split(x_train_exp, y_train_exp, test_size=0.2,stratify = y_train_exp)
	# Fit the model
	model_dnnexp.fit(x_train, y_train, epochs=epochs, batch_size=64,verbose=2,validation_data=(x_val,y_val),callbacks=[lrate])
	# evaluate the model
	exp_dnnscores = model_dnnexp.evaluate(x_test_exp, y_test_exp,verbose=0)
	print("%s: %.2f%%" % (model_dnnexp.metrics_names[1], exp_dnnscores[1]*100))
	print("%s: %.2f%%" % (model_dnnexp.metrics_names[2], exp_dnnscores[2]*100))
	print("%s: %.2f%%" % (model_dnnexp.metrics_names[3], exp_dnnscores[3]*100))
	print("%s: %.2f%%" % (model_dnnexp.metrics_names[4], exp_dnnscores[4]*100))
	acc_dnnexp.append(exp_dnnscores[1] * 100)
	Pr_dnnexp.append(exp_dnnscores[2] * 100)
	Sn_dnnexp.append(exp_dnnscores[3] * 100)
	Mcc_dnnexp.append(exp_dnnscores[4] * 100)
print("Accuracy = %.2f%% (+/- %.2f%%)\n" % (numpy.mean(acc_dnnexp), numpy.std(acc_dnnexp)))
print("Precision = %.2f%% (+/- %.2f%%)\n" % (numpy.mean(Pr_dnnexp), numpy.std(Pr_dnnexp)))
print("Sensitivity = %.2f%% (+/- %.2f%%)\n" % (numpy.mean(Sn_dnnexp), numpy.std(Sn_dnnexp)))
print("Mcc = %.2f%% (+/- %.2f%%)\n" % (numpy.mean(Mcc_dnnexp), numpy.std(Mcc_dnnexp)))


X_clinical_ = numpy.expand_dims(X_clinical, axis=2)
# for extracting final layer features 
#y_pred_ =  clinical_model.predict(X_clinical_)
# for extracting one layer before final layer features
y_pred_clinical = intermediate_layer_clinical.predict(X_clinical_)

X_cnv_ = numpy.expand_dims(X_cnv, axis=2)
# for extracting final layer features 
#y_pred_ =  cnv_model.predict(X_cnv_)
# for extracting one layer before final layer features
y_pred_cnv = intermediate_layer_cnv.predict(X_cnv_)

X_exp_ = numpy.expand_dims(X_exp, axis=2)
# for extracting final layer features 
#y_pred_ =  exp_model.predict(X_exp_)
# for extracting one layer before final layer features
y_pred_exp = intermediate_layer_exp.predict(X_exp_)

stacked_feature=numpy.concatenate((y_pred_clinical,y_pred_cnv,y_pred_exp,Y_clinical[:,None]),axis=1)
with open('/home/nikhil/Desktop/Project/nik/Code/Submodels/CNN/stacked_metadata.csv', 'w') as f:    #change the path to your local system
	for item_clinical in stacked_feature:
		for elem in item_clinical:
			f.write(str(elem)+'\t')
		f.write('\n')

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




