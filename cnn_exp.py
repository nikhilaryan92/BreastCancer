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
# load Gene Exp dataset
#dataset_exp = numpy.loadtxt("/home/nikhil/Desktop/Project/MDNNMD/data/gene.csv", delimiter=",")
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
	model =	Model(inputs=main_input1, outputs=output)
	# summarize layers
	#print(model.summary())
	# plot graph
	plot_model(model, to_file='/home/nikhil/Desktop/Project/nik/Code/Submodels/Model Design/CNN Gene Exp.png')
	def exp_decay(epoch):
		initial_lrate = 0.001
		k = 0.1
		lrate = initial_lrate * math.exp(-k*epoch)
		return lrate
	lrate = LearningRateScheduler(exp_decay)
	adams=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.compile(loss='binary_crossentropy', optimizer=adams, metrics=['accuracy'])
	x_train, x_val, y_train, y_val = train_test_split(x_train_exp, y_train_exp, test_size=0.2,stratify=y_train_exp)	
	#model.fit(x_train_exp, y_train_exp, epochs=epochs,validation_split=0.20, batch_size=8,verbose=2)
	model.fit(x_train, y_train, epochs=epochs, batch_size=8,verbose=2,validation_data=(x_val,y_val),callbacks=[lrate])
	exp_scores = model.evaluate(x_test_exp, y_test_exp,verbose=2)
	print("%s: %.2f%%" % (model.metrics_names[1], exp_scores[1]*100))
	cvscores_exp.append(exp_scores[1] * 100)
	intermediate_layer_model = Model(inputs=main_input1,outputs=dropout2)	
print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores_exp), numpy.std(cvscores_exp)))
#Plotting
x_train_exp, x_test_exp, y_train_exp, y_test_exp = train_test_split(X_exp, Y_exp, test_size = 0.20,stratify=Y_exp)
x_test_exp=numpy.expand_dims(x_test_exp, axis=2)
pred_cnv = model.predict(x_test_exp)
fpr, tpr, threshold = roc_curve(y_test_exp, pred_cnv)
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


X_exp = numpy.expand_dims(X_exp, axis=2)
# for extracting final layer features 
#y_pred =  model.predict(X_exp)
# for extracting one layer before final layer features
y_pred = intermediate_layer_model.predict(X_exp)
#feature_target=numpy.concatenate((y_pred,X_exp[:,None]),axis=1)
with open('/home/nikhil/Desktop/Project/nik/Code/Submodels/CNN/exp_metadata.csv', 'w') as f:
	for item_exp in y_pred:
		for elem in item_exp:
			f.write(str(elem)+'\t')
		f.write('\n')

