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
epochs = 40

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
	model_clinical.fit(x_train, y_train, epochs=epochs, batch_size=64,verbose=2,validation_data=(x_val,y_val),callbacks=[lrate])
	# evaluate the model
	clinical_scores = model_clinical.evaluate(x_test_clinical, y_test_clinical, verbose=0)
	print("%s: %.2f%%" % (model_clinical.metrics_names[1], clinical_scores[1]*100))
	cvscores_clinical.append(clinical_scores[1] * 100)	
	intermediate_layer_model = Model(inputs=main_input1,outputs=dense4)
print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores_clinical), numpy.std(cvscores_clinical)))

#Plotting
X_train, X_test, y_train, y_test = train_test_split(X_clinical, Y_clinical, test_size=0.2,stratify = Y_clinical)
pred_clinical = model_clinical.predict(X_test)
fpr, tpr, threshold = roc_curve(y_test, pred_clinical)
roc_auc = auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate(Sn)')
plt.xlabel('Flase Positive Rate(1-Sp)')
plt.show()

#pred_clinical=numpy.argmax(pred_clinical,axis=1)
#print(confusion_matrix(y_test,pred_clinical),"\n")
#print(classification_report(y_test,pred_clinical))

# for extracting final layer features 
#y_pred =  model.predict(X_clinical)
# for extracting one layer before final layer features
y_pred = intermediate_layer_model.predict(X_clinical)
feature_target=numpy.concatenate((y_pred,Y_clinical[:,None]),axis=1)
with open('/home/nikhil/Desktop/Project/nik/Code/Submodels/DNN/clinical_metadata.csv', 'w') as f:
	for item_clinical in feature_target:
		#f.write("%s\n"%item_clinical)
		for elem in item_clinical:
			f.write(str(elem)+'\t')
		f.write('\n')




