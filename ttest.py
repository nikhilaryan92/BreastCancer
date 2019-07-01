from scipy import stats
import numpy
dataset_clinical = numpy.loadtxt("/home/nikhil/Desktop/Project/nik/T test/ttest_metbric.csv", delimiter=" ")
a = dataset_clinical[:,0:1]
b = dataset_clinical[:,1]

t2, p2 = stats.ttest_ind(a,b)
print("t = " + str(t2))
print("p = " + str(2*p2))
