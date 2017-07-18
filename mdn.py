import keras.backend as K
from keras.models import Model
from keras.layers import Input, LSTM, Dense, merge
import numpy as np
from tqdm import tqdm
from scipy.stats import distributions

def make_slices(x, lookback_length, random_offset=True):
    ''' Convenience function for selecting random slices along the time axis '''
    t0 = np.random.randint(x.shape[1] - lookback_length) if random_offset else 0
    x_slice = x[:, t0:t0 + lookback_length, :]
    y = x[:, t0+lookback_length, :]
    return x_slice, y

def normal(mu, sigma):
    ''' Gaussian PDF using keras' backend abstraction '''
    def f(y):
        pdf = y - mu
        pdf = pdf / sigma
        pdf = - K.square(pdf) / 2.
        return K.exp(pdf) / sigma
    return f

def mdn_likelihood(mu, sigma, pi):
    ''' Likelihood can be computed separately from the loss (NLL) as the loss will average acrsoos the batch'''
    def f(y):
        mixture = normal(mu, sigma)(y)
        mixture = mixture * pi
        likelihood = K.sum(mixture, axis=1)
        return likelihood
    return f

def mdn_loss(likelihood):
    def f(y, y_pred):
        ''' The loss just needs the function signature f(y, y_pred) for keras' training helper
        the loss is computed by manipulating mu, sigma, pi tensors directly; y_pred is just a
        concatenation of these three variables so it can be safely ignored '''
        log_loss = - K.log(likelihood(y))
        return K.mean(log_loss)
    return f

class MixtureDensityNetwork1D(object):
    def __init__(self, input_dim, mixture_components, lstm_dim, nb_layers, dropout_U=0., dropout_W=0.):
        self.input_dim = input_dim
        self.mixture_components = mixture_components
        self.lstm_dim = lstm_dim
        self.nb_layers = nb_layers
        self.dropout_U = dropout_U
        self.dropout_W = dropout_W

        self._build_graph()

    def _build_graph(self):
        ''' Constructs the tf/Keras computation graph using Keras' functional API '''
        self._inputs = Input(shape=(None, self.input_dim))
        h = self._inputs

        for l in range(0, self.nb_layers-1):
            h = LSTM(self.lstm_dim, return_sequences=True,
                     dropout_U=self.dropout_U, dropout_W=self.dropout_W)(h)
        # final LSTM layer maps to a single vector
        self._lstm = LSTM(self.lstm_dim, dropout_U=self.dropout_U, dropout_W=self.dropout_W)(h)

        # map to mu, sigma, pi parameterizing mixture components
        self._mu = Dense(self.mixture_components)(self._lstm)  # linear mean estimation
        self._sigma = Dense(self.mixture_components, activation=K.exp)(self._lstm)  # exponential variance
        self._pi = Dense(self.mixture_components, activation=K.softmax)(self._lstm)  # softmax mixture

        self._gmm = merge([self._mu, self._sigma, self._pi], mode='concat')
        self.mdn = Model(self._inputs, self._gmm)
        self.mu_model = Model(self._inputs, self._mu)
        self.sigma_model = Model(self._inputs, self._sigma)
        self.pi_model = Model(self._inputs, self._pi)

        self._y = K.placeholder((None, 1))
        likelihood_fn = mdn_likelihood(self._mu, self._sigma, self._pi)
        self._likelihood = likelihood_fn(self._y)

        self._loss = mdn_loss(likelihood_fn)
        self.mdn.compile(optimizer='rmsprop', loss=self._loss)

    def _sample_mixture(self, gmm_params):
        ''' Samples from a mixture distribution parameterized by gmm_params. Model parameters are
        arranged in a single vector of length 3 * K whith means mu, standard deviations sigma and
        mixing coefficients pi concatenated for the K mixture components. The mixture is a
        a linear combination of gaussian kernels defined by gmm_params. '''
        mu = gmm_params[:, :self.mixture_components]
        sigma = gmm_params[:, self.mixture_components: 2 * self.mixture_components]
        pi = gmm_params[:, 2 * self.mixture_components: 3 * self.mixture_components]

        # renormalize pi (precision errors can cause probablities to not sum to unity)
        pi = pi / pi.sum(axis=-1, keepdims=True)

        # sample k components with probability pi
        K = np.array([int(np.random.choice(range(self.mixture_components), p=p)) for p in pi])
        sample_mu = np.array([m[k] for m, k in zip(mu, K)])
        sample_sigma = np.array([s[k] for s, k in zip(sigma, K)])

        # vectorization trick, compute batch of samples from a unit normal; scale by sigma and offset means
        N = np.random.randn(mu.shape[0])

        return np.expand_dims(sample_sigma * N + sample_mu, -1)

    def _pdf(self, gmm_params, linspace):
        ''' Computes the complete pdf as a linear combination of gaussians '''
        gmm_params = np.expand_dims(gmm_params, -1)
        mu = gmm_params[:, :self.mixture_components]
        sigma = gmm_params[:, self.mixture_components: 2 * self.mixture_components]
        pi = gmm_params[:, 2 * self.mixture_components: 3 * self.mixture_components]

        # renormalize pi (precision errors can cause probablities to not sum to unity)
        pi = pi / pi.sum(axis=1, keepdims=True)

        # sampling array
        X = np.repeat(np.expand_dims(linspace, axis=0), repeats=mu.shape[1], axis=0)
        X = np.repeat(np.expand_dims(X, axis=0), repeats=mu.shape[0], axis=0)

        # compute normal
        N = np.exp(-(X - mu)**2 / (2 * sigma**2)) / np.sqrt(2 * np.pi * sigma**2)

        # take a weighted sum of gaussian kernels
        density = np.sum(pi * N, 1)
        return density

    def _p(self, gmm_params, x):
        ''' Evaluates P(X|x0...xt) '''
        gmm_params = np.expand_dims(gmm_params, -1)
        mu = gmm_params[:, :self.mixture_components]
        sigma = gmm_params[:, self.mixture_components: 2 * self.mixture_components]
        pi = gmm_params[:, 2 * self.mixture_components: 3 * self.mixture_components]

        # renormalize pi (precision errors can cause probablities to not sum to unity)
        pi = pi / pi.sum(axis=1, keepdims=True)

        # sampling array
        X = np.expand_dims(x, axis=-1)

        # compute normal
        N = np.exp(-(X - mu)**2 / (2 * sigma**2)) / np.sqrt(2 * np.pi * sigma**2)

        # take a weighted sum of gaussian kernels
        p = np.sum(pi * N, 1)
        return p

    def _lstsq(self, gmm_params):
        ''' Computes the least-squares solution as a weight sum of mixture component means. Weights are
        the conditional prior probabilities of each component ie the mixing coefficients. See Bishop '94 eq. 44, 45 '''
        mu = gmm_params[:, :self.mixture_components]
        pi = gmm_params[:, 2 * self.mixture_components: 3 * self.mixture_components]

        # renormalize pi (precision errors can cause probablities to not sum to unity)
        pi = pi / pi.sum(axis=-1, keepdims=True)

        return np.sum(pi * mu, axis=-1)

    def _most_likely(self, gmm_params):
        ''' Approximates maximum of the mixture density parameterized by gmm_params. The approximation selects the
        expected value of the component with the largest central value. See Bishop '94 eq. 48 '''
        mu = gmm_params[:, :self.mixture_components]
        sigma = gmm_params[:, self.mixture_components: 2 * self.mixture_components]
        pi = gmm_params[:, 2 * self.mixture_components: 3 * self.mixture_components]

        # renormalize pi (precision errors can cause probablities to not sum to unity)
        pi = pi / pi.sum(axis=-1, keepdims=True)

        # compute argmax of central value over components
        K = np.argmax(pi / sigma, axis=-1)

        return mu[:, K]

    def fit(self, X, lookback_length, X_validate=None, **kwargs):
        ''' This is a convenience function that slices the input data according to some lookback length and
        trains the LSTM using keras' training logic '''
        x_slice, y = make_slices(X, lookback_length, random_offset=True)
        if X_validate is not None:
            kwargs['validation_data'] = (make_slices(X_validate, lookback_length))

        return self.mdn.fit(x_slice, y, **kwargs)


    def lstsq_sequence(self, X, max_lookback=None):
        ''' Computes the least-squares solution over a sequence (the conditional average < Yt | {Xt-1 ... Xt0} >)
        from the mixture model parameters. (see MixtureDensity1D._lstsq for more on how this is computed)
        This method recovers a sequence that is (in principal) equivalent to the least-squares solution
        (the output of a model we would obtain if trained to minimize mean-squared error on the same data) '''
        if max_lookback is None:
            max_lookback = X.shape[1]

        X_pred = np.zeros_like(X)

        for t in tqdm(range(1, X.shape[1])):
            if t < max_lookback:
                gmm_params = self.mdn.predict(X[:, :t])
            else:
                gmm_params = self.mdn.predict(X[:, t - max_lookback: t])

            X_pred[:, t] = self._lstsq(gmm_params)

        return X_pred

    def ml_sequence(self, X, max_lookback=None):
        ''' Estimates the most likely solution over a sequence. The true value is the MAX{p(Xt | Xt-1...Xt0)}
        however as the pdf is non-convex, this function uses the approximation given in Bishop '94 eq. 48 '''
        if max_lookback is None:
            max_lookback = X.shape[1]

        X_pred = np.zeros_like(X)

        for t in tqdm(range(1, X.shape[1])):
            if t < max_lookback:
                gmm_params = self.mdn.predict(X[:, :t])
            else:
                gmm_params = self.mdn.predict(X[:, t - max_lookback: t])

            X_pred[:, t] = self._most_likely(gmm_params)

        return X_pred

    def sample_from_ground_truth(self, X, max_lookback=None):
        ''' Runs the model over an input sequence obtaining a sequence sample; P(Xt|Xt-1 ... Xt0)
        where the conditional on Xt-1 ... Xt0 refers to actual (ground truth) samples from X '''
        if max_lookback is None:
            max_lookback = X.shape[1]

        X_pred = np.zeros_like(X)

        for t in tqdm(range(1, X.shape[1])):
            if t < max_lookback:
                gmm_params = self.mdn.predict(X[:, :t])
            else:
                gmm_params = self.mdn.predict(X[:, t - max_lookback: t])

            X_pred[:, t] = self._sample_mixture(gmm_params)

        return X_pred

    def sample_from_seed(self, seed, output_len=1):
        ''' Runs the model from an input seed obtaining a sequence sample of arbitrary length;
        P(Xt|X't-1 ... X't0) where the conditional on X't-1 ... X't0 refers to samples from X'
        (ie samples the model itself has generated) '''
        X_pred = np.zeros((seed.shape[0], output_len, seed.shape[2]))
        X = np.copy(seed)

        for t in tqdm(range(output_len)):
            gmm_params = self.mdn.predict(X)
            x_prime = self._sample_mixture(gmm_params)

            X_pred[:, t] = x_prime

            # update the seed (shift time axis and add new sample)
            X = np.roll(X, shift=-1, axis=1)
            X[:, -1] = x_prime

        return X_pred

    def get_density(self, X, linspace, max_lookback=None):
        ''' Compute the conditional density '''
        if max_lookback is None:
            max_lookback = X.shape[1]

        density = np.zeros((X.shape[0], X.shape[1], linspace.shape[0]))

        for t in tqdm(range(1, X.shape[1])):
            if t < max_lookback:
                gmm_params = self.mdn.predict(X[:, :t])
            else:
                gmm_params = self.mdn.predict(X[:, t - max_lookback: t])

            density[:, t, :] = self._pdf(gmm_params, linspace)

        return density

    def P(self, X, max_lookback=None):
        ''' Compute the conditional density '''
        if max_lookback is None:
            max_lookback = X.shape[1]

        p = np.zeros_like(X)

        for t in tqdm(range(1, X.shape[1])):
            if t < max_lookback:
                gmm_params = self.mdn.predict(X[:, :t])
            else:
                gmm_params = self.mdn.predict(X[:, t - max_lookback: t])

            p[:, t, :] = self._p(gmm_params, X[:, t])

        return p


    def eval_likelihood(self, X, max_lookback=None):
        ''' evaluate the likelihood function for a given X '''
        if max_lookback is None:
            max_lookback = X.shape[1]

        likelihood = list()

        for t in tqdm(range(1, X.shape[1])):
            if t < max_lookback:
                l = self._likelihood.eval(feed_dict={K.learning_phase(): 0,
                                                     self._inputs: X[:, :t],
                                                     self._y: X[:, t]}, session=K.get_session())
            else:
                l = self._likelihood.eval(feed_dict={K.learning_phase(): 0,
                                                     self._inputs: X[:, t - max_lookback: t],
                                                     self._y: X[:, t]}, session=K.get_session())
            likelihood.append(l)

        return np.array(likelihood).T

    @property
    def load_weights(self):
        return self.mdn.load_weights

    @property
    def save_weights(self):
        return self.mdn.save_weights


