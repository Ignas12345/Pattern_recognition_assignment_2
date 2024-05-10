import numpy as np
from .DiscreteD import DiscreteD
from .GaussD import GaussD
from .MarkovChain import MarkovChain


class HMM:
    """
    HMM - class for Hidden Markov Models, representing
    statistical properties of random sequences.
    Each sample in the sequence is a scalar or vector, with fixed DataSize.
    
    Several HMM objects may be collected in a single multidimensional array.
    
    A HMM represents a random sequence(X1,X2,....Xt,...),
    where each element Xt can be a scalar or column vector.
    The statistical dependence along the (time) sequence is described
    entirely by a discrete Markov chain.
    
    A HMM consists of two sub-objects:
    1: a State Sequence Generator of type MarkovChain
    2: an array of output probability distributions, one for each state
    
    All states must have the same class of output distribution,
    such as GaussD, GaussMixD, or DiscreteD, etc.,
    and the set of distributions is represented by an object array of that class,
    although this is NOT required by general HMM theory.
    
    All output distributions must have identical DataSize property values.
    
    Any HMM output sequence X(t) is determined by a hidden state sequence S(t)
    generated by an internal Markov chain.
    
    The array of output probability distributions, with one element for each state,
    determines the conditional probability (density) P[X(t) | S(t)].
    Given S(t), each X(t) is independent of all other X(:).
    
    
    References:
    Leijon, A. (20xx) Pattern Recognition. KTH, Stockholm.
    Rabiner, L. R. (1989) A tutorial on hidden Markov models
    	and selected applications in speech recognition.
    	Proc IEEE 77, 257-286.
    
    """
    def __init__(self, mc, distributions):

        self.stateGen = mc
        self.outputDistr = distributions

        self.nStates = mc.nStates
        self.dataSize = distributions[0].dataSize
    
    def rand(self, nSamples):
        """
        [X,S]=rand(self,nSamples); generates a random sequence of data
        from a given Hidden Markov Model.
        
        Input:
        nSamples=  maximum no of output samples (scalars or column vectors)
        
        Result:
        X= matrix or row vector with output data samples
        S= row vector with corresponding integer state values
          obtained from the self.StateGen component.
          nS= length(S) == size(X,2)= number of output samples.
          If the StateGen can generate infinite-duration sequences,
              nS == nSamples
          If the StateGen is a finite-duration MarkovChain,
              nS <= nSamples
        """
        
        #*** Insert your own code here and remove the following error message 
        S = self.stateGen.rand(nSamples)
        l = len(S)
        X = np.zeros((self.dataSize, l))
        for i in range(l):
          X[:,i] = self.outputDistr[S[i] - 1].rand(1).flatten()
        
        
        return X, S
        
    def viterbi(self):
        pass

    def train(self):
        pass

    def stateEntropyRate(self):
        pass

    def setStationary(self):
        pass

    def logprob(self, pX, scale_factors):
        alpha, c = self.stateGen.forward(pX)
        logprob = np.sum(np.log(c))
        logP = logprob + np.sum(np.log(scale_factors))
        return logP

    def adaptStart(self):
        pass

    def adaptSet(self):
        pass

    def adaptAccum(self):
        pass
