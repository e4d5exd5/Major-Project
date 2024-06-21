import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model


def calc_euclidian_dists(x, y):
    """
    calc_euclidian_dists: Calculates the euclidian distance between two tensors
    :param x: Tensor of shape (n, d)
    :param y: Tensor of shape (m, d)
    :return: Tensor of shape (n, m) with euclidian distances
    """
    n = x.shape[0]
    m = y.shape[0]
    x = tf.tile(tf.expand_dims(x, 1), [1, m, 1])
    y = tf.tile(tf.expand_dims(y, 0), [n, 1, 1])
    return tf.reduce_mean(tf.math.pow(x - y, 2), 2) 


class Prototypical(Model):
    def __init__(self, model, w, h, d, c, MC_LOSS_WEIGHT, TAU, N_TIMES):
        '''
        model: encoder model
        w: width of input image
        h: height of input image
        d: depth of input image
        c: number of channels of input image
        '''
        super(Prototypical, self).__init__()
        self.w, self.h, self.d, self.c = w, h, d, c
        self.encoder = model
        self.MC_LOSS_WEIGHT = MC_LOSS_WEIGHT
        self.n_times = N_TIMES

    def call(self, support, query, support_labels, query_labels, K, C, N,training=True, TAU=1):
        '''                                                     
        support: support images (25, 11, 11, 30, 1)
        query: query images (75, 11, 11, 30, 1)
        supppor_labels: support labels (25, 5)
        query_labels: query labels (75, 5)
        K: number of support images per class
        C: number of classes
        N: number of query images per class
        n_times: number of times to pass the query images for variance calculation
        training: True if training, False if testing
        Tau: Temperature Scaling
        '''
        cat = tf.concat([support,query], axis=0)
        loss = 0
        all_predictions = []
        y = np.zeros((len(query_labels),C))
        for i in range(len(query_labels)) :
            if query_labels[i] != 0:
                x = support_labels.index(query_labels[i])
                y[i][x] = 1

            
        for i in range(self.n_times) :
            # Pass through encoder to get embeddings.
            z = self.encoder(cat)

            # Reshape embeddings to separate support and query embeddings.
            # For prototypes, we reshape (C * K) x D embeddings to C x K x D, ie. each class has K examples, each of D dimensions.
            z_support = tf.reshape(z[:C * K],[C, K, z.shape[-1]])
            # For query, we simply take the remaining embeddings.
            z_query = z[C * K:]

            # The prototypes are simply the mean of the support embeddings.
            z_prototypes = tf.math.reduce_mean(z_support, axis=1)

            # Take the euclidian distance between the query embeddings and the prototypes.
            distances = calc_euclidian_dists(z_query, z_prototypes)

            # Calculate the log softmax of the distances. These are the preictions for the current pass.
            log_predictions = tf.    nn.log_softmax(-distances, axis=-1)
                
            # Calculate the loss for the current pass and add it to the total loss.
            loss += - tf.reduce_mean((tf.reduce_sum(tf.multiply(y, log_predictions), axis=-1)))

            # If testing then apply Temprature Scaling
            if not training:
                distances = distances / TAU
            
            # Calculate predictions by applying softmax on distances
            predictions = tf.nn.softmax(-distances, axis=-1)
            
            # Append the predictions for the current pass to the list of predictions.
            all_predictions.append(predictions)
        
        
        if training:
            # Convert the list of predictions to a tensor.
            predictions = tf.convert_to_tensor(np.reshape(np.asarray(all_predictions),(self.n_times,int(C*N),C)))

            # Calculate the standard deviation of the predictions.
            std_predictions = tf.math.reduce_std(predictions,axis=0)

            # Calculate the standard deviation of the true labels.
            std = tf.reduce_sum(tf.reduce_sum(tf.multiply(std_predictions,y),axis=1))

            # Add the standard deviation to the loss.
            loss += self.MC_LOSS_WEIGHT*std

            # Calculate the mean prediction.
            mean_predictions = tf.reduce_mean(predictions,axis=0)

            # Calculate the accuracy for each query patch.
            # Check if the index of max probability is equal to the true class index.
            mean_eq = tf.cast(tf.equal( tf.cast(tf.argmax(mean_predictions, axis=-1), tf.int32), tf.cast(tf.argmax(y,axis=-1), tf.int32)), tf.float32)
            
            # Calculate the mean accuracy.
            mean_accuracy = tf.reduce_mean(mean_eq)
            
            return loss, mean_accuracy, mean_predictions

        else:
            # Calculate the mean predictions.
            mean_predictions = tf.reduce_mean(all_predictions,axis=0)
            
            # Get the index of the max probability for each query patch.
            mean_predictions_indices = tf.argmax(mean_predictions,axis=1)
            
            # Calculate the accuracy for each query patch.
            # Check if the index of max probability is equal to the true class index.
            mean_eq = tf.cast(tf.equal(tf.cast(tf.argmax(mean_predictions, axis= -1), tf.int32), tf.cast(tf.argmax(y, axis=-1), tf.int32)), tf.float32)
            
            # Calculate the mean accuracy.
            mean_accuracy = tf.reduce_mean(mean_eq)
            
            
            '''
            Explanation for classwise_mean_acc:
            
            What we need to get?
                We need to get the mean accuracy for each class.
            
            How to get it?
                We loop over all the query patches.
                x is the index of the true class of the current query patch.
                We check if the index of max probability is equal to the true class index.
                if yes, we append 1 to the list of correct predictions for the current class.
                if no, we append 0 to the list of correct predictions for the current class.
                
                After the loop, we calculate the mean accuracy for each class.
                To do this, we simply divide the number of correct predictions for each class by the total number of predictions for each class.
                    where sum(acc_list) represents the number of correct predictions for the current class.
                    and len(acc_list) represents the total number of predictions for the current class.
            
            And done we have the mean accuracy for each class.
            '''
            classwise_mean_acc = [[] for _ in range(C)]
            std = 0
            for i in range(int(len(query_labels))):
                #  Classwise mean accuracy
                if(query_labels[i] != 0):
                    x = support_labels.index(query_labels[i])
                    
                    is_correct = (mean_predictions_indices[i] == x)
                    classwise_mean_acc[x].append(int(is_correct))
                
                    # Get all the predictions for the current query patch from all the n_times.
                    p_i = np.array([p[i, :] for p in all_predictions])

                    # Get the standard deviation of the predictions for the current query patch.
                    std += tf.math.reduce_std(p_i, axis=0)[x]

            # Calculate the mean accuracy for each class
            classwise_mean_acc = [sum(acc_list) / len(acc_list)
                                if acc_list else 0 for acc_list in classwise_mean_acc]
            

            loss += self.MC_LOSS_WEIGHT*std
            
            return loss, mean_predictions, mean_accuracy, classwise_mean_acc, y


    def save(self, model_path):
        self.encoder.save_weights(model_path)

    def load(self, model_path):
        self.encoder(tf.zeros([1, self.w, self.h, self.c]))
        self.encoder.load_weights(model_path)

