from keras.layers import Dense,Dropout,Input
from sklearn.model_selection import train_test_split,StratifiedKFold
import numpy,math
from sklearn.metrics import roc_curve,auc
from keras.models import Model
from keras.utils import plot_model
from keras import regularizers,initializers
from keras import optimizers
import matplotlib.pyplot as plt
from keras.callbacks import LearningRateScheduler


# fix random seed for reproducibility
numpy.random.seed(1)
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



#Plotting
X_train, X_test, y_train, y_test = train_test_split(X_exp, Y_exp, test_size=0.2,stratify=Y_exp)
pred_exp = model_exp.predict(X_test)
fpr, tpr, threshold = roc_curve(y_test, pred_exp)
roc_auc = auc(fpr,tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr,tpr,'b',label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# for extracting final layer features 
#y_pred =  model_cnv.predict(X_exp)
# for extracting one layer before final layer features
y_pred = intermediate_layer_model.predict(X_exp)
feature_target=numpy.concatenate((y_pred,Y_exp[:,None]),axis=1)
with open('/home/nikhil/Desktop/Project/nik/Code/Submodels/DNN/exp_metadata.csv', 'w') as f:
	for item_exp in feature_target:
		for elem in item_exp:
			f.write(str(elem)+'\t')
		f.write('\n')

