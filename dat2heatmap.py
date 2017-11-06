#BIG CHENG, 2017/10/31
#BIG CHENG, 2017/11/06, refactoring & simplized for heatmap only


import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import OneHotEncoder

from mpl_toolkits.mplot3d import Axes3D

import matplotlib.animation as animation
import seaborn as sns

import os.path


class dat2heatmap:

    n_img_inf = 2979
    n_img_reg = 1057
    n_img_all = 4036
    n_people = 49

    fname_frinf2reg = 'fr_inf2reg.csv'
    fname_fd4img = 'fd4img.dat'
    
    @staticmethod
    def plot_heatmap(hist2d, labels, func='plot', fname=None):
        #sns.set()

        fig = plt.figure()
        ax = plt.axes()

        # Draw a heatmap with the numeric values in each cell
        xticks = np.linspace(0,1,11)
        yticks = np.linspace(0,0.9,10)  ## special
        sns.heatmap(hist2d, linewidths=.5, xticklabels=xticks, yticklabels=yticks, ax = ax, annot=True, fmt="d")
        ax.set_title(labels[0])
        ax.set_xlabel(labels[1])
        ax.set_ylabel(labels[2])

        if func == 'plot':
            plt.show()
        elif func == 'save':
            plt.savefig(fname)
            print "%s saved" % (fname)


    def help_datasets(self):
        print ("Please download & gunzip following datafile before running this program.")
        print ("http://10.70.70.45/data/qt3/fr_inf2reg.csv.gz")
        print ("http://10.70.70.45/data/qt3/fd4img.dat.gz")


    ## load fr_all data
    def load_topn(self):
        if not os.path.isfile(dat2heatmap.fname_frinf2reg):
            self.help_datasets()
            exit()

        df = pd.read_csv(dat2heatmap.fname_frinf2reg, header=None)    ## already sorted
        # 0 for gt, 1-49 for "every pid", 50-99 for "every score"
        self.fr_all = df.as_matrix()


    ## load fd data
    def load_fr(self):
        if not os.path.isfile(dat2heatmap.fname_fd4img):
            self.help_datasets()
            exit()
        ## fd filter
        df2 = pd.read_csv(dat2heatmap.fname_fd4img, header=None)
        self.fd = df2[0].as_matrix()


    def gen_topn_ok(self, n_rank = 2, fd_filter=False):

        self.load_topn()
        self.load_fr()

        if fd_filter:
            fd = self.fd[:dat2heatmap.n_img_inf] ## only use inf img here
            fr_all = self.fr_all[np.where(fd==1)]
        else:
            fr_all = self.fr_all

        fr_gt = fr_all[:,0]
        
        fr_tops = []
        for i in xrange(n_rank):
            fr_tops += [fr_all[:,i+1]]

        fr_scores = []
        for i in xrange(n_rank):
            fr_scores += [fr_all[:,i+50]]

        def gen_ok_score(fr_gt, fr_top, fr_score):
            ok_top = np.equal(fr_top, fr_gt)    ## true/false 
            ok_score = fr_score[np.where(ok_top == True)]
            failed_score = fr_score[np.where(ok_top == False)]
            return ok_score, failed_score, fr_score

        ## ok1 group:
        ok_score11, failed_score11, total_score11 = gen_ok_score(fr_gt, fr_tops[0], fr_scores[0])
        ok_score12, failed_score12, total_score12 = gen_ok_score(fr_gt, fr_tops[0], fr_scores[1])

        ## ok2 group:
        ok_score21, failed_score21, total_score21 = gen_ok_score(fr_gt, fr_tops[1], fr_scores[0])
        ok_score22, failed_score22, total_score22 = gen_ok_score(fr_gt, fr_tops[1], fr_scores[1])

        return [ok_score11, ok_score12, ok_score21, ok_score22], [failed_score11, failed_score12, failed_score21, failed_score22], [total_score11, total_score12, total_score21, total_score22]

    def plot_topn_heatmap_all(self):
        ok_scores, failed_scores, total_scores = self.gen_topn_ok()
        ok_scoresf, failed_scoresf, total_scoresf = self.gen_topn_ok(2, True) ## fd filter
        ok_score11, ok_score12, ok_score21, ok_score22 = ok_scores
        ok_score11f, ok_score12f, ok_score21f, ok_score22f = ok_scoresf
        total_score11, total_score12, total_score21, total_score22 = total_scores
        total_score11f, total_score12f, total_score21f, total_score22f = total_scoresf

        def plot_topn_heatmap(x, y, labels, func, fname):
            ## post process        
            print x.shape, y.shape

            w_bin = 0.1
            bins = np.arange(0,1.01,w_bin)
            print "bins=", bins
            hist, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[[0, 1], [0, 1]])
            print hist.shape, xedges.shape, yedges.shape

            dat2heatmap.plot_heatmap(hist.astype(int), labels, func, fname)

        func = 'plot'
        plot_topn_heatmap(ok_score11, ok_score12, ['heatmap for rank1 is correct','rank2','rank1'], func, 'heatmap_rank12_ok1.png')
        plot_topn_heatmap(ok_score21, ok_score22, ['heatmap for rank2 is correct','rank2','rank1'], func, 'heatmap_rank12_ok2.png')

        plot_topn_heatmap(ok_score11f, ok_score12f, ['heatmap for rank1 is correct (fd filtered)','rank2','rank1'], func, 'heatmap_rank12_ok1f.png')
        plot_topn_heatmap(ok_score21f, ok_score22f, ['heatmap for rank2 is correct (fd filtered)','rank2','rank1'], func, 'heatmap_rank12_ok2f.png')

        plot_topn_heatmap(total_score11, total_score12, ['heatmap(total#) for rank1 is correct','rank2','rank1'], func, 'heatmap_total_rank12_ok1.png')
        plot_topn_heatmap(total_score21, total_score22, ['heatmap(total#) for rank2 is correct','rank2','rank1'], func, 'heatmap_total_rank12_ok2.png')

        plot_topn_heatmap(total_score11f, total_score12f, ['heatmap(total#) for rank1 is correct (fd filtered)','rank2','rank1'], func, 'heatmap_total_rank12_ok1f.png')
        plot_topn_heatmap(total_score21f, total_score22f, ['heatmap(total#) for rank2 is correct (fd filtered)','rank2','rank1'], func, 'heatmap_total_rank12_ok2f.png')


    def plot_topn_heatmap_all_prob(self):
        ok_scores, failed_scores, total_scores = self.gen_topn_ok()
        ok_scoresf, failed_scoresf, total_scoresf = self.gen_topn_ok(2, True) ## fd filter
        ok_score11, ok_score12, ok_score21, ok_score22 = ok_scores
        ok_score11f, ok_score12f, ok_score21f, ok_score22f = ok_scoresf
        total_score11, total_score12, total_score21, total_score22 = total_scores
        total_score11f, total_score12f, total_score21f, total_score22f = total_scoresf

        def plot_topn_heatmap_prob(x, y, x_total, y_total, labels, func, fname):
            w_bin = 0.1
            bins = np.arange(0,1.01,w_bin)

            hist, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[[0, 1], [0, 1]])

            hist_total, xedges, yedges = np.histogram2d(x_total, y_total, bins=bins, range=[[0, 1], [0, 1]])

            hist_prob = hist/(hist_total.astype(float)+np.ones((10, 10))*0.00001)
            hist_prob_percentage = (hist_prob*100).astype(int)
            print hist_prob_percentage ## simple output 
            dat2heatmap.plot_heatmap(hist_prob_percentage, labels, func, fname)

        func = 'plot'
        plot_topn_heatmap_prob(ok_score11, ok_score12, total_score11, total_score12, ['heatmap(acc%) for rank1 is correct','rank2','rank1'], func, 'heatmap_acc_rank12_ok1.png')
        plot_topn_heatmap_prob(ok_score21, ok_score22, total_score21, total_score22, ['heatmap(acc%) for rank2 is correct','rank2','rank1'], func, 'heatmap_acc_rank12_ok2.png')

        plot_topn_heatmap_prob(ok_score11f, ok_score12f, total_score11f, total_score12f, ['heatmap(acc%) for rank1 is correct (fd filtered)','rank2','rank1'], func, 'heatmap_acc_rank12_ok1f.png')
        plot_topn_heatmap_prob(ok_score21f, ok_score22f, total_score21f, total_score22f, ['heatmap(acc%) for rank2 is correct (fd filtered)','rank2','rank1'], func, 'heatmap_acc_rank12_ok2f.png')


if __name__ == '__main__':

    inst1 = dat2heatmap()
    inst1.plot_topn_heatmap_all()
    inst1.plot_topn_heatmap_all_prob() # prob

    


    

