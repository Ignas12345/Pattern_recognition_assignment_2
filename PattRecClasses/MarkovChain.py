import numpy as np
from .DiscreteD import DiscreteD

class MarkovChain:
    """
    MarkovChain - class for first-order discrete Markov chain,
    representing discrete random sequence of integer "state" numbers.
    
    A Markov state sequence S(t), t=1..T
    is determined by fixed initial probabilities P[S(1)=j], and
    fixed transition probabilities P[S(t) | S(t-1)]
    
    A Markov chain with FINITE duration has a special END state,
    coded as nStates+1.
    The sequence generation stops at S(T), if S(T+1)=(nStates+1)
    """
    def __init__(self, initial_prob, transition_prob):

        self.q = initial_prob  #InitialProb(i)= P[S(1) = i]
        self.A = transition_prob #TransitionProb(i,j)= P[S(t)=j | S(t-1)=i]


        self.nStates = transition_prob.shape[0]

        self.is_finite = False
        if self.A.shape[0] != self.A.shape[1]:
            self.is_finite = True


    def probDuration(self, tmax):
        """
        Probability mass of durations t=1...tMax, for a Markov Chain.
        Meaningful result only for finite-duration Markov Chain,
        as pD(:)== 0 for infinite-duration Markov Chain.
        
        Ref: Arne Leijon (201x) Pattern Recognition, KTH-SIP, Problem 4.8.
        """
        pD = np.zeros(tmax)

        if self.is_finite:
            pSt = (np.eye(self.nStates)-self.A.T)@self.q

            for t in range(tmax):
                pD[t] = np.sum(pSt)
                pSt = self.A.T@pSt

        return pD

    def probStateDuration(self, tmax):
        """
        Probability mass of state durations P[D=t], for t=1...tMax
        Ref: Arne Leijon (201x) Pattern Recognition, KTH-SIP, Problem 4.7.
        """
        t = np.arange(tmax).reshape(1, -1)
        aii = np.diag(self.A).reshape(-1, 1)
        
        logpD = np.log(aii)*t+ np.log(1-aii)
        pD = np.exp(logpD)

        return pD

    def meanStateDuration(self):
        """
        Expected value of number of time samples spent in each state
        """
        return 1/(1-np.diag(self.A))
    
    def rand(self, tmax):
        """
        S=rand(self, tmax) returns a random state sequence from given MarkovChain object.
        
        Input:
        tmax= scalar defining maximum length of desired state sequence.
           An infinite-duration MarkovChain always generates sequence of length=tmax
           A finite-duration MarkovChain may return shorter sequence,
           if END state was reached before tmax samples.
        
        Result:
        S= integer row vector with random state sequence,
           NOT INCLUDING the END state,
           even if encountered within tmax samples
        If mc has INFINITE duration,
           length(S) == tmax
        If mc has FINITE duration,
           length(S) <= tmaxs
        """
        
        #*** Insert your own code here and remove the following error message 
        if self.is_finite == False:
          S = []
          S.append(np.random.choice(np.arange(1, self.nStates+1), p = self.q))
          if tmax > 1:
            for i in np.arange(1, tmax):
              next_state = np.random.choice(np.arange(1, self.nStates+1), p = self.A[S[i-1] - 1])
              S.append(next_state)
            return S

        if self.is_finite == True:
          S = []
          S.append(np.random.choice(np.arange(1, self.nStates+1), p = self.q))
          if tmax > 1:
            for i in np.arange(1, tmax):
              next_state = np.random.choice(np.arange(1, self.nStates+2), p = self.A[S[i-1] - 1])
              if next_state == (self.nStates + 1):
                return S
              else:
                S.append(next_state)
            return S

    def viterbi(self):
        pass
    
    def stationaryProb(self):
        pass
    
    def stateEntropyRate(self):
        pass
    
    def setStationary(self):
        pass

    def logprob(self):
        pass

    def join(self):
        pass

    def initLeftRight(self):
        pass
    
    def initErgodic(self):
        pass

    def forward(self, pX):
      #pX is assummed to be n_states x n_observations matrix
      #a_temp s the n_states x n_observations (or n_states x 1)
      #c is nobservations x 1
      #a_hat is n_states x n_observations
      #initialization
      if self.is_finite == True:

        nStates, nObservations = np.shape(pX)
        assert nStates == self.nStates
        a_hat = np.ones((nStates, nObservations))
        c = np.ones((nObservations + 1, 1))

        a_temp = self.q * pX[:, 0]
        c[0] = np.sum(a_temp)
        a_hat[:,0] = a_temp / c[0]

        #forward steps

        for i in np.arange(1, nObservations):
          a_temp = pX[:, i] * (a_hat[:,i-1] @ self.A[:,:-1])
          c[i] = np.sum(a_temp)
          a_hat[:,i] = a_temp / c[i]

        #exit for finite 
        c[nObservations] = a_hat[:,nObservations - 1] @ self.A[:,-1]
        
        return a_hat, c, a_temp

      else:
        nStates, nObservations = np.shape(pX)
        assert nStates == self.nStates
        a_hat = np.ones((nStates, nObservations))
        c = np.ones((nObservations, 1))

        a_temp = self.q * pX[:, 0]
        c[0] = np.sum(a_temp)
        a_hat[:,0] = a_temp / c[0]

        #forward steps

        for i in np.arange(1, nObservations):
          a_temp = pX[:, i] * (a_hat[:,i-1] @ self.A[:,])
          c[i] = np.sum(a_temp)
          a_hat[:,i] = a_temp / c[i]
        
        return a_hat, c

    def finiteDuration(self):
        pass
    
    def backward(self):
        pass

    def adaptStart(self):
        pass

    def adaptSet(self):
        pass

    def adaptAccum(self):
        pass
