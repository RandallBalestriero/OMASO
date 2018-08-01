from pylab import *
execfile('lasagne_tf.py')


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis



datasets = [make_moons(noise=0.3, random_state=0),
            make_circles(noise=0.2, factor=0.5, random_state=1),
            linearly_separable
            ]


h = .02  # step size in the mesh

cpt = 0
for ds_cnt, ds in enumerate(datasets):
	cpt+=1
	X, y = ds
	X = StandardScaler().fit_transform(X)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)
	x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
	y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))

	ax = subplot(len(datasets),3,cpt)
	ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
               edgecolors='k')
	ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,edgecolors='k')
	ax.set_xlim(xx.min(), xx.max())
	ax.set_ylim(yy.min(), yy.max())
	ax.set_xticks(())
	ax.set_yticks(())
	ax = subplot(len(datasets),3,cpt*len(datasets))

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.log_device_placement=True

	session = tf.Session(config=config)
	model1  = DNNClassifier(input_shape,smallQCNN('none'))
	model1.fit(X_train,Y_train,session,X_test,Y_test)
	Z = model1.predict(c_[xx.ravel(), yy.ravel()])[:,1]
	Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,edgecolors='k')
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,edgecolors='k', alpha=0.6)
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())


show()



