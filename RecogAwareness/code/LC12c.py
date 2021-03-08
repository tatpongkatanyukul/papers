"""
Investigate LC interpretation
I.e.,
(1) generate data with know probabilities p(y) and p(x|y);
(2) calculate p(y|x), which is what convention takes for softmax output denoted yk;
(3) calculate p(y|x,s), which is what LC interpretes for yk;
(4) train a model on the generated data and compare its yk to p(y|x) and p(y|x,s).
See LC08.ipynb, LC09a.ipynb and LC10.ipynb for development.
"""

# Required computing modules
import numpy as np
import scipy
from scipy.stats import multivariate_normal

# Required visualizing modules
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# Required training models
import tensorflow as tf
from tensorflow import keras



def prepare_meshdata(prob_fn, row_lim=(-6, 6), col_lim=(-6, 6), res=50):
    Xr = np.linspace(row_lim[0], row_lim[1], res)
    Xc = np.linspace(col_lim[0], col_lim[1], res)
    xr, xc = np.meshgrid(Xr, Xc, sparse=False, indexing='ij')

    zp = np.zeros((res, res))
    for i in range(res):
        for j in range(res):
            x = np.array([[xr[i, j], xc[i, j]]])
            y = prob_fn(x)
            zp[i, j] = y

            # Plot

    # Make data.
    X = xc
    Y = xr
    Z = zp

    return X, Y, Z

def plot_p(prob_fn, row_lim=(-6, 6), col_lim=(-6, 6), res=50, zlim=(0, 1)):
    '''
    plot utility
    '''
    X, Y, Z = prepare_meshdata(prob_fn, row_lim, col_lim, res)

    # Plot the surface.
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(zlim[0], zlim[1])
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    #ax.margins(*margins, x=None, y=None, tight=True)

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


class guruCalc:
    '''
    Compute inference probabilities
    Dist: tuple or list of dicts each with keys 'label', 'prob', 'mean', 'conv'.
    prob ->  p(y)
    mean, conv -> discretized multivariate normal dist of p(x|y)
    '''

    def __init__(self, Dist=None, seen_classes=None):
        self.__Dist = Dist
        self._num_classes = len(Dist)
        self._Omega = range(self._num_classes)
        self._seen_classes = seen_classes

    def set_Dist(self, Dist):
        self.__Dist = Dist
        self._num_classes = len(Dist)
        self._Omega = range(self._num_classes)

    def get_Dist(self):
        return self.__Dist

    def get_num_classes(self):
        return self._num_classes

    def get_Omega(self):
        return self._Omega

    def get_seen_classes(self):
        return self._seen_classes


    def py(self, i):
        '''
        prior p(y)
        '''
        return self.__Dist[i]['prob']

    def print_py(self):
        for i in range(self._num_classes):
            print('p(y={}) = {}'.format(i, self.py(i)))

    def pxCy(self, x, i):
        '''
        Likelihood PMF p(x|y)
        ~ i.i.d of discretized normal distribution
        1 dim: p(x|y=i) = cdf_i(x + 1) - cdf_i(x)
        2 dim: p(x|y=i) = cdf_i(x + [1,1]) - cdf_i(x + [1,0]) - cdf_i(x + [0,1]) + cdf_i(x + [0,0])
        N dim: ?
        *** Now, it only works for 2-dim x.
        '''

        cdf = lambda x: multivariate_normal.cdf(x,
                                                mean=self.__Dist[i]['mean'], cov=self.__Dist[i]['cov'])

        p22 = cdf(x + [1, 1])
        p21 = cdf(x + [1, 0])
        p12 = cdf(x + [0, 1])
        p11 = cdf(x + [0, 0])

        return p22 - p21 - p12 + p11

    def show_pxCy(self, row_lim=(-6, 6), col_lim=(-6, 6), res=50, zlim=(0, 1)):
        for i in range(self._num_classes):
            pf = lambda x: self.pxCy(x, i)
            plot_p(pf, row_lim, col_lim, res, zlim)

    def px(self, x):
        '''
        Marginal p(x)
        '''
        marginal = 0
        for c in self._Omega:
            marginal += self.pxCy(x, c) * self.py(c)

        return marginal

    def show_px(self, row_lim=(-6, 6), col_lim=(-6, 6), res=50, zlim=(0, 0.1)):
        plot_p(self.px, row_lim, col_lim, res, zlim)

    def pyCx(self, i, x):
        '''
        Posterior p(y|x)
        '''
        return self.pxCy(x, i) * self.py(i) / self.px(x)

    def show_pyCx(self, row_lim=(-6, 6), col_lim=(-6, 6), res=50, zlim=(0, 1)):
        for i in range(self._num_classes):
            pf = lambda x: self.pyCx(i, x)
            plot_p(pf, row_lim, col_lim, res, zlim)

    def ps(self):
        '''
        Latent p(s)
        '''
        latent = 0
        for i in self._seen_classes:
            latent += self.py(i)

        return latent

    def print_ps(self):
        print(self.ps())

    def pxy(self, x, i):
        '''
        Joint p(x,y)
        '''
        return self.pxCy(x, i) * self.py(i)

    def show_pxy(self, row_lim=(-6, 6), col_lim=(-6, 6), res=50, zlim=(0, 1)):
        for i in range(self._num_classes):
            pf = lambda x: self.pxy(x, i)
            plot_p(pf, row_lim, col_lim, res, zlim)

    def pxCs(self, x):
        '''
        Latent joint p(x|s)
        '''
        numer = 0
        for i in self._seen_classes:
            numer += self.pxy(x, i)

        return numer / self.ps()

    def show_pxCs(self, row_lim=(-6, 6), col_lim=(-6, 6), res=50, zlim=(0, 1)):
        plot_p(self.pxCs, row_lim, col_lim, res, zlim)

    def pxyCs(self, x, i):
        '''
        Latent joint p(x, y|s)
        '''
        joint = 0
        if i in self._seen_classes:
            joint = self.pxCy(x, i) * self.py(i) / self.ps()

        return joint

    def show_pxyCs(self, row_lim=(-6, 6), col_lim=(-6, 6), res=50, zlim=(0, 1)):
        for i in range(self._num_classes):
            pf = lambda x: self.pxyCs(x, i)
            plot_p(pf, row_lim, col_lim, res, zlim)

    def pyCxs(self, i, x):
        '''
        Latent posterior p(y|x,s)
        '''
        return self.pxyCs(x, i) / self.pxCs(x)

    def show_pyCxs(self, row_lim=(-6, 6), col_lim=(-6, 6), res=50, zlim=(0, 1)):
        for i in range(self._num_classes):
            pf = lambda x: self.pyCxs(i, x)
            plot_p(pf, row_lim, col_lim, res, zlim)

    def _contour_over_classes(self, pfs, row_lim=(-5, 14), col_lim=(-5, 12), res=50,
                     levels=np.linspace(0, 1, 20), alpha=0.8,
                     xlab='x1', ylab='x2', title_struc='prob %d'):

        fig = plt.figure()
        fig.subplots_adjust(hspace=0.6, wspace=0.6,
                            left=0.1, right=0.92, bottom=0.1, top=0.92)

        for i in range(len(pfs)):
            X, Y, Z = prepare_meshdata(pfs[i], row_lim, col_lim, res)

            plt.subplot(2, 2, i+1)
            CS = plt.contour(X, Y, Z, levels,
                          alpha=alpha,
                          cmap=plt.cm.coolwarm,  # plt.cm.bone,
                          origin='lower')

            plt.title(title_struc % i)
            plt.xlabel(xlab)
            plt.ylabel(ylab)
            #cbar = plt.colorbar(CS)

        plt.show()


    def contour_pxCy(self, row_lim=(-5, 14), col_lim=(-5, 12), res=50,
                     levels=np.linspace(0, 1, 20), alpha=0.8,
                     xlab='x1', ylab='x2', title_struc='p(x|y=%d)'):

        pfs = [lambda x: self.pxCy(x, 0), lambda x: self.pxCy(x, 1),
               lambda x: self.pxCy(x, 2), lambda x: self.pxCy(x, 3)]

        self._contour_over_classes(pfs, row_lim, col_lim, res,
                     levels, alpha, xlab, ylab, title_struc)


    def contour_pyCx(self, row_lim=(-5, 14), col_lim=(-5, 12), res=50,
                     levels=np.linspace(0, 1, 20), alpha=0.8,
                     xlab='x1', ylab='x2', title_struc='p(y=%d|x)'):

        pfs = [lambda x: self.pyCx(0, x), lambda x: self.pyCx(1, x),
               lambda x: self.pyCx(2, x), lambda x: self.pyCx(3, x)]

        self._contour_over_classes(pfs, row_lim, col_lim, res,
                     levels, alpha, xlab, ylab, title_struc)

    def contour_pyCxs(self, row_lim=(-5, 14), col_lim=(-5, 12), res=50,
                     levels=np.linspace(0, 1, 20), alpha=0.8,
                     xlab='x1', ylab='x2', title_struc='p(y=%d|x,s)'):

        pfs = [lambda x: self.pyCxs(0, x), lambda x: self.pyCxs(1, x),
               lambda x: self.pyCxs(2, x)]

        self._contour_over_classes(pfs, row_lim, col_lim, res,
                     levels, alpha, xlab, ylab, title_struc)

    def posterior_diff(self, row_lim = (-5, 14), col_lim = (-5, 12), res = 50,
        levels = np.linspace(0, 1, 20), alpha = 0.8,
        xlab = 'x1', ylab = 'x2'):

        fig = plt.figure()
        fig.subplots_adjust(hspace=0.6, wspace=0.6,
                            left=0.1, right=0.92, bottom=0.1, top=0.92)

        diff = []
        rel_diff = []
        for i in range(self._num_classes):
            pconv = lambda x: self.pyCx(i, x)
            placz = lambda x: self.pyCxs(i, x)
            X, Y, Z_conv = prepare_meshdata(pconv, row_lim, col_lim, res)
            X, Y, Z_lacz = prepare_meshdata(placz, row_lim, col_lim, res)
            se = (Z_conv - Z_lacz)**2
            diff.append( se )

            epsilon = 1e-6
            reld = np.abs(Z_conv - Z_lacz + epsilon)/(Z_conv + Z_lacz + epsilon)
            rel_diff.append(reld)

            # Plot
            plt.subplot(2, 2, i + 1)
            CS = plt.contour(X, Y, reld, levels,
                             alpha=alpha,
                             cmap=plt.cm.coolwarm,  # plt.cm.bone,
                             origin='lower')

            plt.title('(p(y=%d|x) - p(y=%d|x,s))^2' % (i,i))
            plt.xlabel(xlab)
            plt.ylabel(ylab)
            #cbar = plt.colorbar(CS)

        plt.show()

        return diff, rel_diff


    def explore_probs(self):
        '''
        print and plot all probabilities
        '''
        print('p(y)')
        self.print_py()

        print('p(x|y)')
        self.show_pxCy(row_lim=(-3, 12), col_lim=(-3, 12), zlim=(0, 0.02))

        self.contour_pxCy(levels=20)

        print('p(x)')
        self.show_px(row_lim=(-3, 12), col_lim=(-3, 12), zlim=(0, 0.02))

        print('p(s)')
        self.print_ps()

        print('p(x|s)')
        self.show_pxCs(row_lim=(-3, 12), col_lim=(-3, 12), zlim=(0, 0.02))

        print('p(x,y)')
        self.show_pxy(row_lim=(-3, 12), col_lim=(-3, 12), zlim=(0, 0.02))

        print('p(x,y|s)')
        self.show_pxyCs(row_lim=(-3, 12), col_lim=(-3, 12), zlim=(0, 0.02))

        print('p(y|x')
        self.show_pyCx(row_lim=(-3, 12), col_lim=(-3, 12), zlim=(0, 1))

        self.contour_pyCx(levels=20)

        print('p(y|x,s')
        self.show_pyCxs(row_lim=(-3, 12), col_lim=(-3, 12), zlim=(0, 1))

        self.contour_pyCxs(levels=20)


class guruSim(guruCalc):

    def __init__(self, Dist=None, seen_classes=None,
                 Train_Data=None, Test_Data=None):
        super().__init__(Dist, seen_classes)

        self.All_Data = None
        self.Train_Data = Train_Data
        self.Test_Data = Test_Data

    def _gen_data(self, num_points=500):

        data_sim = np.zeros((1, len(super().get_Dist()[0]['mean']) + 1))

        for r in super().get_Dist():
            n = round(num_points * r['prob'])
            x = np.random.multivariate_normal(r['mean'], r['cov'], n)

            xd = np.floor(x)

            # d: Num x Dim, 3 x 2
            y = np.ones((n, 1)) * r['label']

            data_sim = np.vstack((data_sim, np.hstack((xd, y))))


        return data_sim[1:]

    def plot_data2D(self, dat, syms=['r+', 'gx', 'b*', 'ko'],
                   sym_legend=['C0', 'C1', 'C2', 'C3'],
                   title='Data Points', xlabel='x1', ylabel='x2'):
        '''
        :param dat as 2D np.array of size Num x 3,
            for 2-dim X and 1-dim Y
        '''

        nrow = len(dat)

        unique_ys = set(dat[:,2])

        plt_handle = []

        for y in sorted(unique_ys):
            yi = int(y)
            ids = np.where(dat[:,2] == yi)
            ph = plt.plot(dat[ids,0], dat[ids,1], syms[yi-1])
            plt_handle.append(ph[0])

        plt.axis('equal')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend(plt_handle, sym_legend, loc=0)
        plt.show()

    def prepare_data(self, seen_train_size=1500, seen_test_size=500, unseen_class=3):
        assert unseen_class not in super().get_seen_classes() \
                and unseen_class < super().get_num_classes()

        seen_prob = 0
        for i in super().get_seen_classes():
            seen_prob += super().get_Dist()[i]['prob']

        Train_Size = int(np.ceil(seen_train_size / seen_prob))
        Test_Size = int(np.ceil(seen_test_size / seen_prob))

        all_data1 = self._gen_data(num_points=Train_Size)
        all_data2 = self._gen_data(num_points=Test_Size)

        self.All_Data = [all_data1, all_data2]

        # Select only seen classes to the training set
        train_ids = np.where(all_data1[:, -1]  != unseen_class)[0]
        train_x = all_data1[train_ids, :-1]
        train_y = all_data1[train_ids, -1]

        # Shuffle
        train_N = len(train_ids)

        train_sids = np.random.choice(train_N, train_N, replace=False)
        train_x = train_x[train_sids, :]
        train_y = train_y[train_sids]

        print('# train (seen) = ', train_N)
        print('# unseen in train_y:', unseen_class in train_y)

        # Select only seen data for test
        test_ids = np.where(all_data2[:, -1]  != unseen_class)[0]
        test_x = all_data2[test_ids, :-1]
        test_y = all_data2[test_ids, -1]

        print('# test = ', len(test_y))
        print('# unseen in test_y:', unseen_class in test_y)

        self.Train_Data = {'x': train_x, 'y': train_y}
        self.Test_Data = {'x':test_x, 'y': test_y}

    def plot_Data(self, dat, syms=['r+', 'gx', 'b*', 'ko'],
                   sym_legend=['C0', 'C1', 'C2'],
                   title='Train Data', xlabel='x1', ylabel='x2'):
        '''
        :param dat as dict{ 'x': 2D np.array of size Num x 2,
                            'y': 1D np.array of Num }
        '''

        unique_ys = set(dat['y'])

        plt_handle = []

        for y in sorted(unique_ys):
            yi = int(y)
            ids = np.where(dat['y'] == yi)[0]


            ph = plt.plot(dat['x'][ids,0], dat['x'][ids,1], syms[yi-1])
            plt_handle.append(ph[0])

        plt.axis('equal')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend(plt_handle, sym_legend, loc=0)
        plt.show()


if __name__ == '__main__':
    seen_classes = [0, 1, 2]
    unseen_class = 3

    Dist = (
        {'label': 0, 'prob': 0.2, 'mean': [0, 0], 'cov': [[9, -1], [-1, 11]]},
        {'label': 1, 'prob': 0.2, 'mean': [6, 9], 'cov': [[9, 0], [0, 12]]},
        {'label': 2, 'prob': 0.2, 'mean': [12, 0], 'cov': [[11, 1], [1, 9]]},
        {'label': 3, 'prob': 0.4, 'mean': [6, 4.5], 'cov': [[5, 0], [0, 5]]})

    ####################################
    # # # Test guruCalc class
    ####################################

    gu = guruCalc(Dist, seen_classes)
    # #gu.explore_probs()
    #
    # # # Difference
    # diff, rel_diff = gu.posterior_diff(row_lim=(-3, 12), col_lim=(-3, 12), res=10)
    #
    # # diff : list of 4 of [res x res]
    # for d in diff:
    #     print(np.average(d))
    #
    # # Turn diff to 2D array of 4 x (res*res)
    # adiff = np.array([diff[0].ravel(), diff[1].ravel(), diff[2].ravel()])
    #
    # # multiple box plots on one figure
    # plt.figure()
    # plt.boxplot(np.transpose(adiff))
    # plt.show()

    # # Relative difference
    # arel = np.array([rel_diff[0].ravel(), rel_diff[1].ravel(), rel_diff[2].ravel()]) * 100
    #
    # # multiple box plots on one figure
    # plt.figure()
    # plt.boxplot(np.transpose(arel))
    # plt.title('Relative Difference Percentage')
    # # |p(y|x) - p(y|x,s)|/|p(y|x) + p(y|x,s)| * 100%
    # plt.show()


    ####################################
    # # Test guruSim class
    ####################################

    # gs = guruSim(Dist, seen_classes)
    # # print(gs.get_Dist())
    # # print(gs.get_num_classes())
    # # print(gs.get_Omega())
    # # print(gs.Data)
    # # data = gs._gen_data(500)
    # # gs.plot_data2D(data)
    # # print(gs.Train_Data)
    # # print(gs.Test_Data)
    #
    # gs.prepare_data(1500, 500)
    # # gs.plot_data2D(gs.All_Data[0], title='Generated Data 1')
    # # gs.plot_data2D(gs.All_Data[1], title='Generated Data 2')
    # # gs.plot_Data(gs.Train_Data, title='Train Data')
    # # gs.plot_Data(gs.Test_Data, title='Test Data')
    #
    # # Define model
    # model = keras.Sequential([
    #     keras.layers.Dense(16, activation=tf.nn.relu),
    #     keras.layers.Dense(3, activation=tf.nn.softmax)
    # ])
    #
    # model.compile(optimizer='adam',
    #               loss='sparse_categorical_crossentropy',
    #               metrics=['accuracy'])
    #
    # # Train the model
    # import time
    # start = time.time()
    # model.fit(gs.Train_Data['x'], gs.Train_Data['y'], epochs=40)
    # end = time.time()
    # print('Elapse time:{}'.format(end - start))
    #
    # # Test the model
    #
    # test_loss, test_acc = model.evaluate(gs.Test_Data['x'], gs.Test_Data['y'])
    # print('\n\nTest accuracy:', test_acc)
    #
    # # Examine output
    # results = []
    # for x in gs.Test_Data['x']:
    #     _softmax  = model.predict(np.array([x]))
    #     _conv     = [gu.pyCx(0, x), gu.pyCx(1, x), gu.pyCx(2, x)]
    #     _lacz     = [gu.pyCxs(0, x), gu.pyCxs(1, x), gu.pyCxs(2, x)]
    #     results.append({'softmax': _softmax[0], 'conv': _conv, 'lacz': _lacz})
    #
    #np.save('LC12a_190210a.npy', results)

    gu.explore_probs()