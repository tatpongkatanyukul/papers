#Copyright (c) 2010, Roland Memisevic
#All rights reserved.
#
#Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
#    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#
#            THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from numpy import zeros, ones, newaxis, array, random, double, dot, concatenate, exp, log, sum, nan, inf, argmax, ndarray, mod, abs
from numpy.random import randn, rand
import cudamat


def logsumexp(x, dim=-1):
    """Compute log(sum(exp(x))) in a numerically stable way.
    
       Use second argument to specify along which dimensions the logsumexp
       shall be computed. If -1 (which is also the default), logsumexp is 
       computed along the last dimension. 
    """
    if len(x.shape) < 2:  #only one possible dimension to sum over?
        if x.shape[0] == 0:
            return x.copy()
        xmax = x.max()
        return xmax + log(sum(exp(x-xmax)))
    else:
        if dim != -1:
            x = x.transpose(range(dim) + range(dim+1, len(x.shape)) + [dim])
        lastdim = len(x.shape)-1
        xmax = x.max(lastdim)
        return xmax + log(sum(exp(x-xmax[...,newaxis]),lastdim))


class GatedSoftmax(object):

    def __init__(self,numin, numclasses, numhid, batchsize=100, input_density='gaussian'):
        self.numin   = numin
        self.numclasses  = numclasses
        self.numhid  = numhid
        self.batchsize = batchsize
        self.input_density = input_density
        self.numparams = numin*numhid*numclasses+numclasses+numhid+numin
        self.gparams = cudamat.CUDAMatrix(randn(1, self.numparams)*0.01)
        self.wxyh = [None] * self.numclasses
        for c in range(self.numclasses):
            self.wxyh[c] = self.gparams.get_col_slice(c*numin*numhid, 
                                                    (c+1)*numin*numhid).\
                                            reshape((numin, numhid))
        self.wy = self.gparams.get_col_slice(numin*numclasses*numhid,
                              numin*numclasses*numhid+numclasses)
        self.wh = self.gparams.get_col_slice(numin*numclasses*numhid+numclasses,
                              numin*numclasses*numhid+numclasses+numhid)
        self.wx = self.gparams.get_col_slice(numin*numclasses*numhid+numclasses+numhid,
                              numin*numclasses*numhid+numclasses+numhid+numin)
        self.wy.assign(0.0)
        self.wh.assign(0.0)
        self._wyh = [None] * self.numclasses
        for c in range(self.numclasses):
            self._wyh[c] = cudamat.empty((self.batchsize, numhid))
            self._wyh[c].assign(0.0)
        self._wyh_tmp = cudamat.empty((self.batchsize, numhid))
        self._wyh_tmp.assign(0.0)
        self.grad_wx = cudamat.empty(self.wx.shape)
        self.grad_wx.assign(0.0)
        self.grad_wy = cudamat.empty(self.wy.shape)
        self.grad_wy.assign(0.0)
        self.grad_wh = cudamat.empty(self.wh.shape)
        self.grad_wh.assign(0.0)
        self.grad_wyh = [None] * self.numclasses
        self.grad_wxyh = [None] * self.numclasses
        for c in range(self.numclasses):
            self.grad_wyh[c] = cudamat.empty(self._wyh[c].shape)
            self.grad_wyh[c].assign(0.0)
            self.grad_wxyh[c] = cudamat.empty(self.wxyh[c].shape)
            self.grad_wxyh[c].assign(0.0)
        self._grad_wxyh_tmp = cudamat.empty(self.wxyh[c].shape)
        self._grad_wxyh_tmp.assign(0.0)
        self.ginputs = cudamat.empty((self.numin, self.batchsize))
        self.ginputs.assign(0.0)
        self.ginputs.copy_to_host()
        self.outputs = cudamat.empty((self.batchsize, self.numclasses))
        self.outputs.assign(0.0)
        self.outputs.copy_to_host()
        self.outputs_minus_probs = cudamat.empty((self.batchsize, self.numclasses))
        self.outputs_minus_probs.assign(0.0)
        self.outputs_minus_probs.copy_to_host()
        self._unnormalized = cudamat.empty((self.batchsize, self.numclasses))
        self._unnormalized.assign(0.0)
        self._unnormalized.copy_to_host()
        self.hidfact = cudamat.empty((self.batchsize, self.numhid))
        self.hidfact.assign(0.0)
        self.hidfact2 = cudamat.empty((self.batchsize, self.numhid))
        self.hidfact2.assign(0.0)
        self.hidfactgreater0 = cudamat.empty((self.batchsize, self.numhid))
        self.hidfactgreater0.assign(0.0)
        self.probs = cudamat.empty((self.batchsize, self.numclasses))
        self.probs.assign(0.0)
        self.probs.copy_to_host()
        self.logprobs = cudamat.empty((self.batchsize, self.numclasses))
        self.logprobs.assign(0.0)
        self._lse = cudamat.empty((1, self.batchsize))
        self.gparams.copy_to_host()
        self.params = self.gparams.numpy_array
        self._hprobs = zeros((self.numhid, self.batchsize), 'single')
        self._hidprobs = cudamat.empty((self.batchsize, self.numhid))
        self._hidprobs2 = cudamat.empty((self.batchsize, self.numhid))
        self._av_hidprob = cudamat.empty((1, self.numhid))
        self._av_hidprob2 = cudamat.empty((1, self.numhid))
        self._grad = zeros((self.numparams), 'single')

        # settings for training:
        self.stepsize = 10.0**-3.0
        self.inc = zeros(self.numparams, 'single')
        self.momentum = 0.9
        self.oldcost = inf
        self.firstcall = True

        # allocate memory for generative training:
        self.inputs_tmp = cudamat.empty((self.batchsize, self.numin))
        self.input_recs = cudamat.empty((self.batchsize, self.numin))
        self.input_recs_ = cudamat.empty((self.batchsize, self.numin))
        self.input_recs__ = cudamat.empty((self.batchsize, self.numin))
        self.input_recs___ = cudamat.empty((self.batchsize, self.numin))
        self.input_recs____ = cudamat.empty((self.batchsize, self.numin))
        self.input_rand = cudamat.empty((self.batchsize, self.numin))
        self.input_rand_ = cudamat.empty((self.batchsize, self.numin))
        self.large_a_results = cudamat.empty((self.batchsize, self.numin))
        self.hy = cudamat.empty((self.batchsize, self.numhid))
        self.h_rand = cudamat.empty((self.batchsize, self.numhid))
        self.wxyh_tmp = cudamat.empty((self.numin, self.numhid))

    def hidprobs(self, inputs, outputs):
        numdims, numcases = inputs.shape
        self._hidprobs.assign(0.0)
        self._hidprobs.add_row_vec(self.wh)
        if type(outputs) == ndarray:
            self.outputs.copy_to_host()
            self.outputs.numpy_array[:,:] = outputs.T
            self.outputs.copy_to_device()
        else:
            outputs.transpose(self.outputs)
        self.modulateweights(inputs)
        for c in range(self.numclasses):
            self._wyh_tmp.assign(self._wyh[c])
            self._wyh_tmp.mult_by_col(self.outputs.get_col_slice(c, c+1))
            self._hidprobs.add(self._wyh_tmp)
        self._hidprobs.apply_sigmoid()
        return self._hidprobs

    def hidprobsT(self, inputs):
        """ Compute hidprobs for already initialized inputs and outputs."""
        self._hidprobs.assign(0.0)
        self._hidprobs.add_row_vec(self.wh)
        for c in range(self.numclasses):
            cudamat.dot(inputs.T, self.wxyh[c], target=self._wyh[c])
        for c in range(self.numclasses):
            self._wyh_tmp.assign(self._wyh[c])
            self._wyh_tmp.mult_by_col(self.outputs.get_col_slice(c, c+1))
            self._hidprobs.add(self._wyh_tmp)
        self._hidprobs.apply_sigmoid()
        return self._hidprobs

    def hidprobs_rec(self, inputs):
        """ Compute hidprobs for already initialized inputs and outputs."""
        self._hidprobs.assign(0.0)
        self._hidprobs.add_row_vec(self.wh)
        for c in range(self.numclasses):
            cudamat.dot(inputs, self.wxyh[c], target=self._wyh[c])
        for c in range(self.numclasses):
            self._wyh_tmp.assign(self._wyh[c])
            self._wyh_tmp.mult_by_col(self.outputs.get_col_slice(c, c+1))
            self._hidprobs.add(self._wyh_tmp)
        self._hidprobs.apply_sigmoid()
        return self._hidprobs

    def compute_logprobs(self, unnormalized):
        self.logprobs.assign(unnormalized)
        self.logprobs.transpose().max(0, target=self._lse)
        self._lse.mult(-1.0)
        self.logprobs.add_col_vec(self._lse.transpose()) 
        cudamat.exp(self.logprobs)
        self._lse.mult(-1.0)
        self._lse.add(cudamat.log(self.logprobs.transpose().sum(0)))
        self._lse.mult(-1.0)
        self.logprobs.assign(unnormalized)
        self.logprobs.add_col_vec(self._lse.transpose())

    def classify_nocuda(self, inputs):
        numdims, numcases = inputs.shape
        _wxyh = [self.wxyh[c].asarray() for c in range(self.numclasses)]
        _wh = self.wh.asarray().flatten()
        _wy = self.wy.asarray().flatten()
        result = [None] * self.numclasses
        for c in range(self.numclasses):
            result[c] = dot(_wxyh[c].T, inputs) + _wh[:, newaxis]
            result[c] = logsumexp(
                          concatenate(
                                (zeros((self.numhid,numcases,1) , 'single'),
                                        result[c][:,:,newaxis]), 2)
                                , 2)
            result[c] = result[c].sum(0)
        result = array(result) + _wy[:,newaxis]
        return (result==result.max(0)).astype(int)

    def classify(self, inputs):
        numdims, numcases = inputs.shape
        result = zeros((self.numclasses, numcases), 'int')
        if type(inputs) == ndarray:
            for batch in range(numcases/self.batchsize):
                result[:, batch*self.batchsize : (batch+1)*self.batchsize] = \
             self.classifybatch(inputs[:, batch*self.batchsize:(batch+1)*self.batchsize])
        else:
            for batch in range(numcases/self.batchsize):
                result[:, batch*self.batchsize : (batch+1)*self.batchsize] = \
             self.classifybatch(inputs.get_col_slice(batch*self.batchsize,(batch+1)*self.batchsize))
        rest = mod(numcases, self.batchsize)
        if rest > 0:
            if type(inputs) == ndarray:
                result[:, -rest:] = self.classify_nocuda(inputs[:, -rest:])
            else:
                result[:, -rest:] = self.classify_nocuda(
             inputs.get_col_slice(batch*self.batchsize, inputs.shape[1]))
        return (result==result.max(0)).astype(int)

    def classifybatch(self, inputs):
        numdims, numcases = inputs.shape
        assert numcases == self.batchsize
        self.modulateweights(inputs)
        self._unnormalized.assign(0.0)
        self._unnormalized.add_row_vec(self.wy)
        result = [None] * self.numclasses
        for c in range(self.numclasses):
            self.hidfact.assign(self._wyh[c])
            self.hidfact.add_row_vec(self.wh)
            self.hidfact.greater_than(0.0, target=self.hidfactgreater0)
            self.hidfactgreater0.mult(self.hidfact)
            self.hidfact.subtract(self.hidfactgreater0)
            cudamat.exp(self.hidfact)
            self.hidfact2.assign(0.0)
            self.hidfact2.subtract(self.hidfactgreater0)
            cudamat.exp(self.hidfact2)
            self.hidfact.add(self.hidfact2)
            cudamat.log(self.hidfact)
            self.hidfact.add(self.hidfactgreater0)
            self._unnormalized.get_col_slice(c,c+1).add_sums(self.hidfact, 1)
        unnormalized = self._unnormalized.asarray().T
        return (unnormalized==unnormalized.max(0)).astype(int)

    def cost(self, inputs, outputs, weightcost):
        numdims, numcases = inputs.shape
        assert numcases == self.batchsize
        self.modulateweights(inputs)
        self._unnormalized.assign(0.0)
        self._unnormalized.add_row_vec(self.wy)
        for c in range(self.numclasses):
            self.hidfact.assign(self._wyh[c])
            self.hidfact.add_row_vec(self.wh)
            self.hidfact.greater_than(0.0, target=self.hidfactgreater0)
            self.hidfactgreater0.mult(self.hidfact)
            self.hidfact.subtract(self.hidfactgreater0)
            cudamat.exp(self.hidfact)
            self.hidfact2.assign(0.0)
            self.hidfact2.subtract(self.hidfactgreater0)
            cudamat.exp(self.hidfact2)
            self.hidfact.add(self.hidfact2)
            cudamat.log(self.hidfact)
            self.hidfact.add(self.hidfactgreater0)
            self._unnormalized.get_col_slice(c,c+1).add_sums(self.hidfact, 1)
        self.compute_logprobs(self._unnormalized)
        if type(outputs) == ndarray:
            cost = -(self.logprobs.asarray().T*outputs).sum()/double(self.batchsize)
        else:
            cost = -(self.logprobs.asarray().T*outputs.asarray()).sum()/double(self.batchsize)
        for c in range(self.numclasses):
            cost += weightcost * (self.wxyh[c].asarray()**2).sum()
        cost += weightcost * (self.wh.asarray()**2).sum()
        return cost

    def grad(self, inputs, outputs, weightcost):
        numdims, numcases = inputs.shape
        assert numcases == self.batchsize
        if type(outputs) == ndarray:
            self.outputs.copy_to_host()
            self.outputs.numpy_array[:,:] = outputs.T
            self.outputs.copy_to_device()
        else:
            outputs.transpose(self.outputs)
        self.modulateweights(inputs)
        self._unnormalized.assign(0.0)
        self._unnormalized.add_row_vec(self.wy)
        for c in range(self.numclasses):
            self.hidfact.assign(self._wyh[c])
            self.hidfact.add_row_vec(self.wh)
            self.hidfact.greater_than(0.0, target=self.hidfactgreater0)
            self.hidfactgreater0.mult(self.hidfact)
            self.hidfact.subtract(self.hidfactgreater0)
            cudamat.exp(self.hidfact)
            self.hidfact2.assign(0.0)
            self.hidfact2.subtract(self.hidfactgreater0)
            cudamat.exp(self.hidfact2)
            self.hidfact.add(self.hidfact2)
            cudamat.log(self.hidfact)
            self.hidfact.add(self.hidfactgreater0)
            self._unnormalized.get_col_slice(c,c+1).add_sums(self.hidfact, 1)
        self.compute_logprobs(self._unnormalized)
        self.probs.assign(self.logprobs)
        cudamat.exp(self.probs)
        self.outputs_minus_probs.assign(self.outputs)
        self.outputs_minus_probs.subtract(self.probs)
        self.grad_wy.assign(0.0)
        self.grad_wy.add_sums(self.outputs_minus_probs, 0)
        self.grad_wy.divide(double(self.batchsize))
        self.grad_wh.assign(0.0)
        for c in range(self.numclasses):
            self.grad_wyh[c].assign(0.0)
            self.grad_wyh[c].add_row_vec(self.wh)
            self.grad_wyh[c].add(self._wyh[c])
            self.grad_wyh[c].apply_sigmoid()
            self.grad_wyh[c].mult_by_col(self.outputs_minus_probs.get_col_slice(c, c+1))
            cudamat.dot(self.inputs, self.grad_wyh[c], target=self.grad_wxyh[c]) 
            self.grad_wxyh[c].divide(double(self.batchsize))
            self.grad_wxyh[c].add_mult(self.wxyh[c], -2.0*weightcost)
            self.grad_wh.add_sums(self.grad_wyh[c],0)
        self.grad_wh.divide(double(self.batchsize))
        self.grad_wh.add_mult(self.wh, -2.0*weightcost)

        self.grad_wx.assign(0.0)
        self._grad[:] = -concatenate(([g.asarray().T.flatten() for g in self.grad_wxyh]
                                        + [self.grad_wy.asarray().flatten()]
                                        + [self.grad_wh.asarray().flatten()]
                                        + [self.grad_wx.asarray().flatten()]))
        return self._grad

    def grad_generative(self, inputs, outputs, weightcost):
        numdims, numcases = inputs.shape
        assert numcases == self.batchsize
        if type(inputs) == ndarray:
            self.ginputs.numpy_array[:, :] = inputs
            self.ginputs.copy_to_device()
            self.inputs = self.ginputs
        else:
            self.inputs = inputs
        if type(outputs) == ndarray:
            self.outputs.copy_to_host()
            self.outputs.numpy_array[:,:] = outputs.T
            self.outputs.copy_to_device()
        else:
            outputs.transpose(self.outputs)

        #erase old grad-values:
        for c in range(self.numclasses):
            self.grad_wxyh[c].assign(0.0)
        self.grad_wh.assign(0.0)
        self.grad_wx.assign(0.0)
        # positive phase:
        hidprobs = self.hidprobsT(self.inputs)
        self.h_rand.fill_with_rand()
        # add positive gradients:
        for c in range(self.numclasses):
            self.inputs.transpose(self.inputs_tmp)
            self.inputs_tmp.mult_by_col(self.outputs.get_col_slice(c,c+1))
            self.grad_wxyh[c].add_dot(self.inputs_tmp.T, hidprobs)
        self.grad_wh.add_sums(hidprobs, axis=0)
        self.grad_wx.add_sums(self.inputs_tmp, axis=0)
        hidprobs.greater_than(self.h_rand)
        # negative phase:
        self.input_recs.assign(0.0)
        for c in range(self.numclasses):
            self.hy.assign(hidprobs)
            self.hy.mult_by_col(self.outputs.get_col_slice(c, c+1))
            self.wxyh_tmp.assign(self.wxyh[c])
            self.input_recs.add_dot(self.hy, self.wxyh_tmp.T)
        self.input_recs.add_row_vec(self.wx)
        if self.input_density=='bernoulli':
            self.input_recs.apply_sigmoid()
            self.input_rand.fill_with_rand()
            self.input_recs.greater_than(self.input_rand)
        elif self.input_density=='gaussian':
            self.input_rand.fill_with_randn()
            self.input_recs.add(self.input_rand)
        else:
            assert False, 'unknown input density'
        # re-compute hidprobs 
        hidprobs = self.hidprobs_rec(self.input_recs)
        # add negative gradients
        for c in range(self.numclasses):
            self.inputs_tmp.assign(self.input_recs)
            self.inputs_tmp.mult_by_col(self.outputs.get_col_slice(c,c+1))
            self.grad_wxyh[c].subtract_dot(self.inputs_tmp.T, hidprobs)
        self.grad_wh.add_sums(hidprobs, axis=0, mult=-1.0)
        self.grad_wx.add_sums(self.input_recs, axis=0, mult=-1.0)

        for c in range(self.numclasses):
            self.grad_wxyh[c].divide(double(self.batchsize))
        self.grad_wh.divide(double(self.batchsize))
        self.grad_wx.divide(double(self.batchsize))

        # add weightcost gradient
        for c in range(self.numclasses):
            self.grad_wxyh[c].add_mult(self.wxyh[c], -2.0*weightcost)

        self.grad_wh.add_mult(self.wh, -2.0*weightcost)

        self._grad[:] = -concatenate((
                                [g.asarray().T.flatten() for g in self.grad_wxyh]
                                        + [self.grad_wy.asarray().flatten()*0.0]
                                        + [self.grad_wh.asarray().flatten()]
                                        + [self.grad_wx.asarray().flatten()]))
        return self._grad

    def zeroone(self, inputs, labels):
        if type(labels) != ndarray:
            labels = labels.asarray()
        return 1.0 - (self.classify(inputs)*labels).sum().sum()/\
                                                      double(inputs.shape[1])

    def modulateweights(self, inputs):
        numin, numcases = inputs.shape
        assert numcases == self.batchsize
        #cuda-version:
        if type(inputs)==ndarray:
            self.ginputs.numpy_array[:, :] = inputs
            self.ginputs.copy_to_device()
            self.inputs = self.ginputs
        else:
            #self.inputs.assign(inputs)
            self.inputs = inputs
        for c in range(self.numclasses):
            cudamat.dot(self.inputs.T, self.wxyh[c], target=self._wyh[c])
        #non-cuda-version:
        #for c in range(self.numclasses):
        #    _wxh = self.wxyh[c].asarray().flatten().reshape(self.numin, self.numhid)
        #    self._wyh[c] = cudamat.CUDAMatrix(dot(inputs.T, _wxh))

    def f(self, x, inputs, outputs, weightcost):
        """Wrapper function around cost function to check grads, etc."""
        numdims, numcases = inputs.shape
        self.gparams.copy_to_host()
        xold = self.gparams.numpy_array.copy()
        self.updateparams(x.copy().flatten())
        if inputs.shape[1]==self.batchsize:
            result = self.cost(inputs, outputs, weightcost)
        else:
            result = 0.0
            #since we're calling the cost-function mulitple times, we have to adjust the weightcost
            weightcostfactor = double(numcases)/self.batchsize
            if type(inputs)==ndarray:
                for batch in range(numcases/self.batchsize):
                    result += self.cost(inputs[:, batch*self.batchsize:
                                                  (batch+1)*self.batchsize], 
                                  outputs[:, batch*self.batchsize:
                                                  (batch+1)*self.batchsize],
                                  weightcost/weightcostfactor)
            else:
                for batch in range(numcases/self.batchsize):
                    result += self.cost(inputs.get_col_slice(batch*self.batchsize, 
                                                  (batch+1)*self.batchsize), 
                                  outputs.get_col_slice(batch*self.batchsize, 
                                                  (batch+1)*self.batchsize),
                                  weightcost/weightcostfactor)
                
        self.updateparams(xold.copy())
        return result

    def g(self, x, inputs, outputs, weightcost):
        """Wrapper function around gradient to check grads, etc."""
        numdims, numcases = inputs.shape
        self.gparams.copy_to_host()
        xold = self.gparams.numpy_array.copy()
        self.updateparams(x.copy().flatten())
        if inputs.shape[1]==self.batchsize:
            result = self.grad(inputs, outputs, weightcost)
        else:
            #since we're calling the cost-function mulitple times, we have to adjust the weightcost
            weightcostfactor = double(numcases)/self.batchsize
            result = zeros(self.numparams, 'single')
            if type(inputs)==ndarray:
                for batch in range(numcases/self.batchsize):
                    result += self.grad(inputs[:, batch*self.batchsize:
                                                  (batch+1)*self.batchsize], 
                                  outputs[:, batch*self.batchsize:
                                                  (batch+1)*self.batchsize],
                                  weightcost/weightcostfactor)
            else:
                for batch in range(numcases/self.batchsize):
                    result += self.grad(inputs.get_col_slice(batch*self.batchsize, 
                                                  (batch+1)*self.batchsize),
                                  outputs.get_col_slice(batch*self.batchsize, 
                                                  (batch+1)*self.batchsize),
                                  weightcost/weightcostfactor)
        self.updateparams(xold.copy())
        return result

    def updateparams(self,newparams):
        self.gparams.copy_to_host()
        self.gparams.numpy_array[:, :] = newparams.copy()[newaxis, :]
        self.gparams.copy_to_device()
        self.params[:,:] = self.gparams.numpy_array

    def train_gendisc(self, inputs, outputs, weightcost, numsteps, weight_generative=0.5, weight_discriminative=0.5):
        """ Train using both discriminative and generative gradients. 

            Uses simple gradient steps. Generative gradients are approximated using 
            contrastive divergence.
        """

        numdims, numcases = inputs.shape
        if type(inputs) == ndarray:
            for step in range(numsteps):
                print 'gradstep ', step
                for batch in range(numcases/self.batchsize):
                    g = weight_discriminative * self.grad(inputs[:, batch*self.batchsize:
                                            (batch+1)*self.batchsize], 
                                 outputs[:, batch*self.batchsize:
                                            (batch+1)*self.batchsize], 
                                  weightcost)
                    g += weight_generative * self.grad_generative(inputs[:, batch*self.batchsize:
                                            (batch+1)*self.batchsize], 
                                 outputs[:, batch*self.batchsize:
                                            (batch+1)*self.batchsize], 
                                  weightcost)
                    self.inc[:] = self.momentum*self.inc - self.stepsize * g 
                    self.updateparams(self.params + self.inc)
        else:
            for step in range(numsteps):
                print 'gradstep ', step
                for batch in range(numcases/self.batchsize):
                    g = weight_discriminative * self.grad(inputs.get_col_slice(batch*self.batchsize,
                                            (batch+1)*self.batchsize),
                                 outputs.get_col_slice(batch*self.batchsize,
                                            (batch+1)*self.batchsize),
                                 weightcost)
                    g += weight_generative * self.grad_generative(inputs.get_col_slice(batch*
                                        self.batchsize,(batch+1)*self.batchsize),
                                 outputs.get_col_slice(batch*self.batchsize,
                                                        (batch+1)*self.batchsize),
                                 weightcost)
                    self.inc[:] = self.momentum*self.inc - self.stepsize * g 
                    self.updateparams(self.params + self.inc)

    def train_disc(self, inputs, outputs, weightcost, numsteps):
        """ Train discriminatively using simple gradient steps. 

        """
        numdims, numcases = inputs.shape
        if type(inputs) == ndarray:
            for step in range(numsteps):
                print 'gradstep ', step
                for batch in range(numcases/self.batchsize):
                    g = self.grad(inputs[:, batch*self.batchsize:
                                            (batch+1)*self.batchsize], 
                                 outputs[:, batch*self.batchsize:
                                            (batch+1)*self.batchsize], 
                                  weightcost)
                    self.inc[:] = self.momentum*self.inc - self.stepsize * g 
                    self.updateparams(self.params + self.inc)
        else:
            for step in range(numsteps):
                print 'gradstep ', step
                for batch in range(numcases/self.batchsize):
                    g = self.grad(inputs.get_col_slice(batch*self.batchsize,
                                            (batch+1)*self.batchsize),
                                 outputs.get_col_slice(batch*self.batchsize,
                                            (batch+1)*self.batchsize),
                                 weightcost)
                    self.inc[:] = self.momentum*self.inc - self.stepsize * g 
                    self.updateparams(self.params + self.inc)

    def train_bolddriver(self, inputs, outputs, weightcost, numsteps):
        """ Train with step-size adaptive gradient descent (AKA the "bold driver" algorithm). 

        """
        for step in range(numsteps):
            if self.firstcall:
                self.firstcall = False
                self.oldcost = self.f(self.params, inputs, outputs, weightcost)
                #print "initial cost: %f " % self.oldcost
            g = self.g(self.params, inputs, outputs, weightcost)
            self.inc[:] = self.momentum*self.inc - self.stepsize * g 
            self.updateparams(self.params + self.inc)
            self.newcost = self.f(self.params, inputs, outputs, weightcost)
            if self.newcost <= self.oldcost:
                print "cost: %f " % self.newcost
                print "increasing step-size to %f" % self.stepsize
                self.oldcost = self.newcost
                self.stepsize = self.stepsize * 1.1
            else:
                print "cost: %f " % self.newcost
                print "decreasing step-size to %f" % self.stepsize
                #roll back changes to parameters and increments:
                self.updateparams(self.params - self.inc)
                if self.momentum > 0.0:
                    self.inc[:] = (self.inc + self.stepsize * g) / self.momentum
                else:
                    self.inc *= 0.0
                self.newcost = self.oldcost
                self.stepsize = self.stepsize * 0.5
                if self.stepsize < 10.0**-8.0:
                    print 'stepssize < ', 10**.0-8.0, ' exiting.'

    def train_cg(self, inputs, outputs, weightcost, maxnumlinesearch=100):
        """ Train with conjugate gradients. 

            This method makes use of the external minimize module.  
        """

        from minimize import minimize
        numdims, numcases = inputs.shape
        assert mod(numcases, self.batchsize) == 0, 'input size must be multiple of model batchsize'
        p, g, numlinesearches = minimize(self.params.copy(), self.f, self.g, (inputs, outputs, weightcost), maxnumlinesearch)
        self.updateparams(p)
        return numlinesearches

 
if __name__=='__main__':
    #INITIALIZE GPU
    import cudamat 
    cudamat.CUDAMatrix.init_random()
    cudamat.init()

    #INSTANTIATE MODEL
    model = GatedSoftmax(50, 2, 20, batchsize=100)
    #MAKE RANDOM (NONSENSE) TRAINING DATA
    inputs = randn(50, 100)
    outputs = (rand(1, 100)>0.5).astype(int)
    outputs = concatenate((outputs, 1-outputs),0)
    #TRAIN MODEL ON NONSENSE DATA USING GENERATIVE/DISCRIMINATIVE TRAINING
    model.train_gendisc(inputs, outputs, weightcost=0.001, numsteps=100)
    #CLASSIFY TRAINING CASES AND PRINT ZEROONE-COST
    print 'classification:', model.classify(inputs)
    print 'training cost:', model.zeroone(inputs, outputs)


