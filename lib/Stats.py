# Python 3.10.1
# Aditya Sawant

import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score


class Stats:

    def __init__(self, mc_predictions1, mc_predictions2, y1, y2, target_names):
        self.mc_predictions1 = mc_predictions1
        self.mc_predictions2 = mc_predictions2
        self.y1 = y1
        self.y2 = y2
        self.target_names = target_names
    def get_accuracy(self):
        # Accuracy
        mean_predictions1 =  tf.reduce_mean(self.mc_predictions1,axis=0)
        mean_predictions2 =  tf.reduce_mean(self.mc_predictions2,axis=0)
        overall_predictions = tf.concat([mean_predictions1,mean_predictions2],axis=0)
        overall_true_labels = tf.concat([self.y1,self.y2],axis=0)
        correct_pred = tf.cast(tf.equal(                                             # accuracy for the current pass
                    tf.cast(tf.argmax(overall_predictions, axis=-1), tf.int32), 
                    tf.cast(tf.argmax(overall_true_labels,axis=-1), tf.int32)), tf.float32)
        o_acc = tf.reduce_mean(correct_pred) 
        print(f"Overall accuracy:{o_acc.numpy():.3f}")
        
        mean_predictions1 =  tf.reduce_mean(self.mc_predictions1,axis=0)
        cm_pred1 = tf.argmax(mean_predictions1, axis=-1)
        mean_predictions2 =  tf.reduce_mean(self.mc_predictions2,axis=0)
        cm_pred2 = tf.argmax(mean_predictions2, axis=-1) + 3
        overall_predictions = tf.concat([cm_pred1,cm_pred2],axis=0)
        cm_true1 = tf.argmax(self.y1,axis=-1)
        cm_true2 = tf.argmax(self.y2,axis=-1) + 3
        overall_true_labels = tf.concat([cm_true1,cm_true2],axis=0)
        results = confusion_matrix(overall_true_labels,overall_predictions) 
        print ('Confusion Matrix :')
        print(results) 
        print ('Report : ')
        print (classification_report(overall_true_labels, overall_predictions, target_names=self.target_names))    
        print("Cohen's Kappa Score: ", cohen_kappa_score(overall_true_labels, overall_predictions))