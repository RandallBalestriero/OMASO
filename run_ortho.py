from pylab import *
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

execfile('utils.py')
execfile('models.py')
execfile('lasagne_tf.py')

DATASET = sys.argv[-4]
lr      = float(sys.argv[-3])

if(int(sys.argv[-2])==0):
	m = smallCNN
	m_name = 'smallCNN'
elif(int(sys.argv[-2])==1):
	m = largeCNN
	m_name = 'largeCNN'

elif(int(sys.argv[-2])==2):
        m = resnet_large
        m_name = 'resnetLarge'


x_train,x_test,y_train,y_test,c,n_epochs,input_shape=load_utility(DATASET)
n_epochs=120
for kk in xrange(10):
	all_train = []
	all_test  = []
	all_W     = []
	for coeff in linspace(0,1,4):
        	name = DATASET+'_'+m_name+'_lr'+str(lr)+'_run'+str(kk)+'_c'+str(coeff)
		model1  = DNNClassifier(input_shape,m(bn=1,n_classes=c),lr=lr,gpu=int(sys.argv[-1]),Q=coeff)
		train_loss,test_loss,W = model1.fit(x_train,y_train,x_test,y_test,n_epochs=n_epochs)
		all_train.append(train_loss)
		all_test.append(test_loss)
		all_W.append(W)
		f = open('/mnt/project2/rb42Data/OMASO/'+name,'wb')
		cPickle.dump([all_train,all_test,all_W],f)
		f.close()




