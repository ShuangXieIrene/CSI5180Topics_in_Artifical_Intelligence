import tensorflow as tf
import numpy as np 

class TFNaiveBayes:
    def fit_NormalDistribution(self, X, y):
        # Separate training points by class (nb_classes * nb_samples * nb_features)
        unique_y = np.unique(y)
        feature_by_class = np.array([
            [x for x, t in zip(X, y) if t == c]
            for c in unique_y])

        # len_x = [len(feature) for feature in feature_by_class]
        # feature_by_class_resize = np.array([feature[:min(len_x)] for feature in feature_by_class]).astype(np.float32)

        # Estimate mean and variance for each class / feature
        # shape: nb_classes * nb_features
        # mean, var = tf.nn.moments(tf.constant(feature_by_class_resize), axes=[1])
        mean, var = [tf.nn.moments(tf.constant(feature), axes=0) for feature in feature_by_class]

        # # Create a 3x2 univariate normal distribution with the 
        # # known mean and variance
        self.dist = tf.distributions.Normal(loc=mean, scale=tf.sqrt(var))

    def predict(self, X):
        assert self.dist is not None
        nb_classes, nb_features = map(int, self.dist.scale.shape)
        # Conditional probabilities log P(x|c) with shape
        # (nb_samples, nb_classes)
        cond_probs = tf.reduce_sum(
            self.dist.log_prob(
                tf.reshape(
                    tf.tile(X, [1, nb_classes]), [-1, nb_classes, nb_features])),
            axis=2)
        print tf.reshape(tf.tile(X, [1, nb_classes]),  [-1, nb_classes, nb_features]).eval()


        # uniform priors
        priors = np.log(np.array([1. / nb_classes] * nb_classes)).astype(np.float32)

        # posterior log probability, log P(c) + log P(x|c)
        joint_likelihood = tf.add(priors, cond_probs)
        

        # normalize to get (log)-probabilities
        norm_factor = tf.reduce_logsumexp(
            joint_likelihood, axis=1, keepdims=True)
        log_prob = joint_likelihood - norm_factor
        # exp to get the actual probabilities
        return tf.exp(log_prob)