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
epochs = 20

# fix random seed for reproducibility
numpy.random.seed(1)
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
	model =	Model(inputs=main_input1, outputs=output)
	# summarize layers
	#print(model.summary())
	# plot graph
	plot_model(model, to_file='/home/nikhil/Desktop/Project/nik/Code/Submodels/Model Design/CNN CNV.png')
	def exp_decay(epoch):
		initial_lrate = 0.01
		k = 0.1
		lrate = initial_lrate * math.exp(-k*epoch)
		return lrate
	lrate = LearningRateScheduler(exp_decay)
	adams=optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	model.compile(loss='binary_crossentropy', optimizer=adams, metrics=['accuracy'])
	#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	x_train, x_val, y_train, y_val = train_test_split(x_train_cnv, y_train_cnv, test_size=0.2,stratify=y_train_cnv)
	model.fit(x_train, y_train, epochs=epochs, batch_size=8,verbose=2,validation_data=(x_val,y_val),callbacks=[lrate])
	#model.fit(x_train_cnv, y_train_cnv, epochs=epochs,validation_data=(x_val,y_val), batch_size=8,verbose=2)
	cnv_scores = model.evaluate(x_test_cnv, y_test_cnv,verbose=2)
	print("%s: %.2f%%" % (model.metrics_names[1], cnv_scores[1]*100))
	cvscores_cnv.append(cnv_scores[1] * 100)	
	intermediate_layer_model = Model(inputs=main_input1,outputs=dropout2)
print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores_cnv), numpy.std(cvscores_cnv)))
#Plotting
X_train, X_test, y_train, y_test = train_test_split(X_cnv, Y_cnv, test_size=0.2,stratify=Y_cnv)
X_test=numpy.expand_dims(X_test, axis=2)
pred_cnv = model.predict(X_test)
fpr, tpr, threshold = roc_curve(y_test, pred_cnv)
roc_auc = auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate(Sn)')
plt.xlabel('False Positive Rate(1-Sp)')
plt.show()

X_cnv = numpy.expand_dims(X_cnv, axis=2)
# for extracting final layer features 
#y_pred =  model.predict(X_cnv)
# for extracting one layer before final layer features
y_pred = intermediate_layer_model.predict(X_cnv)
stacked_feature=numpy.concatenate((y_pred,Y_cnv[:,None]),axis=1)
with open('/home/nikhil/Desktop/Project/nik/Code/Submodels/CNN/cnv_metadata.csv', 'w') as f:
	for item_cnv in stacked_feature:
		for elem in item_cnv:
			f.write(str(elem)+'\t')
		f.write('\n')

