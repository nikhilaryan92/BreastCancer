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
	model = Model(inputs=main_input1, outputs=output)
	# summarize layers
	#print(model.summary())
	# plot graph
	plot_model(model, to_file='/home/nikhil/Desktop/Project/nik/Code/Submodels/Model Design/CNN Clinical.png')

	def exp_decay(epoch):
		initial_lrate = 0.001
		k = 0.1
		lrate = initial_lrate * math.exp(-k*epoch)
		return lrate
	lrate = LearningRateScheduler(exp_decay)
	adams=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	model.compile(loss='binary_crossentropy', optimizer=adams, metrics=['accuracy'])
	#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	x_train, x_val, y_train, y_val = train_test_split(x_train_clinical, y_train_clinical, test_size=0.2,stratify=y_train_clinical)
	model.fit(x_train, y_train, epochs=epochs, batch_size=8,verbose=2,validation_data=(x_val,y_val),callbacks=[lrate])
	#model.fit(x_train, y_train, epochs=epochs, batch_size=8,verbose=2,validation_data=(x_val,y_val))	

	clinical_scores = model.evaluate(x_test_clinical, y_test_clinical,verbose=2)
	print("%s: %.2f%%" % (model.metrics_names[1], clinical_scores[1]*100))
	cvscores_clinical.append(clinical_scores[1] * 100)
	intermediate_layer_model = Model(inputs=main_input1,outputs=dense1)
print("Accuracy = %.2f%% (+/- %.2f%%)\n" % (numpy.mean(cvscores_clinical), numpy.std(cvscores_clinical)))
#Plotting
X_train, X_test, y_train, y_test = train_test_split(X_clinical, Y_clinical, test_size=0.5,stratify=Y_clinical)
X_test=numpy.expand_dims(X_test, axis=2)
pred_clinical = model.predict(X_test)
fpr, tpr, threshold = roc_curve(y_test, pred_clinical,pos_label=1)
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

X_clinical = numpy.expand_dims(X_clinical, axis=2)
# for extracting final layer features 
y_pred =  model.predict(X_clinical)
# for extracting one layer before final layer features
#y_pred = intermediate_layer_model.predict(X_clinical)
stacked_feature=numpy.concatenate((y_pred,Y_clinical[:,None]),axis=1)
with open('/home/nikhil/Desktop/Project/nik/Code/Submodels/CNN/clinical_metadata.csv', 'w') as f:
	for item_clinical in stacked_feature:
		#f.write("%s\n"%item_clinical)
		for elem in item_clinical:
			f.write(str(elem)+'\t')
		f.write('\n')
