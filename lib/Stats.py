# Python 3.10.1
# Aditya Sawant

import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import os
import csv
from operator import truediv

class Stats:

    def __init__(self, mc_predictions1, mc_predictions2, y1, y2):
        self.mean_predictions1 = mc_predictions1
        self.mean_predictions2 = mc_predictions2
        self.y1 = y1
        self.y2 = y2
        self.get_accuracy()


    def get_accuracy(self):
        # Accuracy
        try:
            self.y_test = tf.concat([self.mean_predictions1,self.mean_predictions2],axis=0)
            self.y_pred = tf.concat([self.y1,self.y2],axis=0)
        except Exception as e:
            print('There is only one mean prediction. Skipping Concat')
            self.y_test = self.mean_predictions1
            self.y_pred = self.y1
        correct_pred = tf.cast(tf.equal(                                             # accuracy for the current pass
                    tf.cast(tf.argmax(self.y_test, axis=-1), tf.int32), 
                    tf.cast(tf.argmax(self.y_pred,axis=-1), tf.int32)), tf.float32)
        o_acc = tf.reduce_mean(correct_pred) 
        
        try:
            cm_pred1 = tf.argmax(self.mean_predictions1, axis=-1)
            cm_pred2 = tf.argmax(self.mean_predictions2, axis=-1) + 3
            self.y_test = tf.concat([cm_pred1,cm_pred2],axis=0)
            cm_true1 = tf.argmax(self.y1,axis=-1)
            cm_true2 = tf.argmax(self.y2,axis=-1) + 3
            self.y_pred = tf.concat([cm_true1,cm_true2],axis=0)
        except:
            cm_pred1 = tf.argmax(self.mean_predictions1, axis=-1)
            self.y_test = cm_pred1
            cm_true1 = tf.argmax(self.y1,axis=-1)
            self.y_pred = cm_true1

        self.classification = classification_report(self.y_pred, self.y_test)    
        self.oa = accuracy_score(self.y_pred, self.y_test)*100
        self.confusion = confusion_matrix(self.y_pred,self.y_test)
        self.each_aa, self.aa = self.AA_andEachClassAccuracy()
        self.kappa = cohen_kappa_score(self.y_test, self.y_pred)*100
    
    def printReport(self):
        classification = str(self.classification)
        confusion = str(self.confusion)
        print('{} Kappa accuracy (%)'.format(self.kappa))
        print('\n')
        print('{} Overall accuracy (%)'.format(self.oa))
        print('\n')
        print('{} Average accuracy (%)'.format(self.aa))
        print('\n')
        print('\n')
        print('{}'.format(classification))
        print('\n')
        print('{}'.format(confusion))
    
    def saveReport(self, PATH, FINAL_REPORT_PATH, dataset, n_times, tau, run, encoder):
        classification = str(self.classification)
        confusion = str(self.confusion)
        print(f'OA {self.oa:.2f} | KA {self.kappa:.2f} | AA {self.aa:.2f}')
        os.makedirs(os.path.dirname(PATH), exist_ok=True)
        with open(PATH, 'w') as x_file:
            
            x_file.write('Dataset : {}'.format(dataset))
            x_file.write('\n')
            x_file.write('N times : {}'.format(n_times))
            x_file.write('\n')
            x_file.write('TAU : {}'.format(tau))
            x_file.write('\n')
            x_file.write('{} Overall accuracy (%)'.format(self.oa))
            x_file.write('\n')
            x_file.write('{} Kappa accuracy (%)'.format(self.kappa))
            x_file.write('\n')
            x_file.write('{} Average accuracy (%)'.format(self.aa))
            x_file.write('\n')
            x_file.write('\n')
            x_file.write('{}'.format(classification))
            x_file.write('\n')
            x_file.write('{}'.format(confusion))
            x_file.close()
        
        with open(FINAL_REPORT_PATH, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([dataset, encoder, tau, n_times, run+1, f'{self.oa:.2f}', f'{self.kappa:.2f}', f'{self.aa:.2f}'])
        
    def AA_andEachClassAccuracy(self):
        counter = self.confusion.shape[0]
        list_diag = np.diag(self.confusion)
        list_raw_sum = np.sum(self.confusion, axis=1)
        each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
        average_acc = np.mean(each_acc)
        return each_acc*100, average_acc*100
        