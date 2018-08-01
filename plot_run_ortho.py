import cPickle
from pylab import *
import glob
import matplotlib as mpl
label_size = 36
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size

#models = ['smallCNN']
#lrs = ['0.0001','0.0005','0.001']
#C= linspace(0,2,15)
#C=(C*100).astype('int32').astype('float32')/100.0

def load_files(DATASET,model,lr):
	"""Returns [array of shape (C,RUNS,BATCHES),(C,RUNS,EPOCHS)]"""
	all_train = []
        all_test  = []
	Cs        = []
	files     = sort(glob.glob('/mnt/project2/rb42Data/OMASO/'+DATASET+'*'+model+'_lr'+lr+'_run0_c*'))
	print "ALL FILES",files
	for f,cc in zip(files,xrange(len(files))):
		trainc = []
		testc  = []
                Cs.append(float(f.split('c')[-1][:4]))
		subfiles = glob.glob(f.replace('run0','run*'))
		print subfiles
		print int(subfiles[0].split('run')[1].split('_')[0])
                lists = subfiles
                for ff in lists:
			print 'dogin',ff
			fff = open(ff,'rb')
                        content = cPickle.load(fff)
			train = content[0]
			test  = content[1]
                        fff.close()
			print shape(train),shape(test)
                        trainc.append(train[cc])#find(Cs[-1]==C)[0]])
                        testc.append(test[cc])#find(Cs[-1]==C)[0]])
		all_train.append(asarray(trainc))
                all_test.append(asarray(testc))
		print shape(all_train[-1])
	return all_train,all_test,Cs



def load_Ws(DATASET,model,lr):
        """Returns [array of shape (C,RUNS,BATCHES),(C,RUNS,EPOCHS)]"""
        Ws        = []
        Cs = []
        files     = sort(glob.glob('../../SAVE/QUADRATIC/'+DATASET+'*'+model+'_lr'+lr+'_run0_c2*'))
        print files
        for f,cc in zip(files,xrange(len(files))):
                print f
                trainc = []
                testc  = []
                Cs.append(float(f.split('c')[-1][:4]))
                ff = open(f,'rb')
                content = cPickle.load(ff)
                Ws = content[2]#[cc]
                print shape(Ws)
                ff.close()
#                Ws.append(W)
        return asarray(Ws)[:,-1,:,:],(linspace(0,2,10)*100).astype('int32').astype('float32')/100.0



def compute_mean_std_max(data):
	return asarray([d[:,-3:].mean() for d in data]),asarray([d[:,-3:].mean(1).std() for d in data]),asarray([d.max() for d in data])


def plot_files(models,lrs,DATASET):
        for model in models:
		all_train = []
		all_test  = []
		figure(figsize=(18,8))
		cpt=1
		for lr in lrs:
			train,test,Cs = load_files(DATASET,model,lr)
			all_train.append([d.mean(0) for d in train])
			all_test.append([d.mean(0) for d in test])
			#
			dmean,dstd,dmax = compute_mean_std_max(test)
			print dmean
                	subplot(1,len(lrs),cpt)
                	plot(Cs,100*dmax,'bo')
                	plot(Cs,100*dmean,'ko')
                	fill_between(Cs,100*dmean+100*dstd,100*dmean-100*dstd,alpha=0.5,facecolor='gray')
                        title('Learning Rate:'+lr,fontsize=32)
                	xlabel(r'$\lambda$',fontsize=37)
                	if(lr==lrs[0]):
                	        ylabel('Test Accuracy',fontsize=37)
                	cpt+=1
#        	suptitle(DATASET+' '+model,fontsize=18)
		savefig(DATASET+'_'+model+'_histo.png')
		close()
		figure(figsize=(18,8))
		cpt=1
                for lr,i in zip(lrs,xrange(len(lrs))):
			subplot(2,len(lrs),cpt)
			semilogy(all_train[i][0],'b',alpha=0.5)
                        semilogy(all_train[i][7],'k',alpha=0.5)
#			xlabel('Batch',fontsize=19)
                        if(lr==lrs[0]):
                                ylabel(r'$\log (\mathcal{L}_{CE})$',fontsize=37)
			title('Learning Rate:'+lr,fontsize=32)
                        subplot(2,len(lrs),len(lrs)+cpt)
                        plot(all_test[i][0]*100,color='b',alpha=0.5,linewidth=3)
                        plot(all_test[i][7]*100,color='k',alpha=0.5,linewidth=3)
			if(lr==lrs[0]):
                                ylabel('Test Accuracy',fontsize=37)
#                        xlabel('Epoch',fontsize=19)
			cpt+=1
#                suptitle(DATASET+' '+model,fontsize=18)
                savefig(DATASET+'_'+model+'_loss.png')
		close()
#	show()	

def normalize(x):
    return (x-x.min())/(x.max()-x.min())

def plot_Ws(models,lrs,DATASET):
        for model in models:
		all_train = []
		all_test  = []
		figure(figsize=(25,9))
		cpt=1
		for lr in lrs:
			Ws,Cs = load_Ws(DATASET,model,lr)
                        plot([])
                        for W,i in zip(Ws,range(len(Ws))):
                	    subplot(3,len(Ws),1+i)
                            K=(dot(W.T,W)-diag((W**2).sum(0)))**2
                	    imshow(normalize(K),interpolation='nearest',aspect='auto')
                            xticks([])
                            yticks([])
                            subplot(3,len(Ws),1+i+len(Ws))
                            hist(log(K[K.nonzero()]).flatten(),200)
#                            xticks([])
#                            yticks([])
                        subplot(313)
                        semilogy(Cs,[mean((dot(W.T,W)-diag((W**2).sum(0)))**2) for W in Ws])
                        ylabel(r'$\log(\mathcal{L}_\lambda(W^{(L)})$',fontsize=22)
                        xlabel(r'$\lambda$',fontsize=22)
        	suptitle(DATASET+' '+model,fontsize=25)
                savefig(DATASET+'_'+model+'_Ws.png')
#show()



#plot_files(['resnetLarge'],lrs = ['0.0005'],DATASET='CIFAR100')
plot_files(['smallCNN'],lrs = ['0.0001'],DATASET='CIFAR')
#plot_files(['smallCNN'],lrs = ['0.0001','0.0005','0.001'],DATASET='CIFAR')
#plot_files(['largeCNN'],lrs = ['0.0005'],DATASET='CIFAR100')
###plot_files(['largeCNN'],lrs = ['0.0005'],DATASET='CIFAR')
###plot_files(['largeCNN'],lrs = ['0.0005'],DATASET='SVHN')



#plot_Ws(['largeCNN'],lrs = ['0.0005'],DATASET='CIFAR100')

