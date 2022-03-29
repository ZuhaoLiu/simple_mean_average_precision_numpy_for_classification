import numpy as np

def make_one_hot(input_array, num_class):
    out_array = np.eye(num_class)[input_array]
    return out_array

def mean_average_precision(all_probs, all_labels):
    '''
    Args:
        all_probs: 2D numpy.ndarray. The first dimension is number of samples, and the second one is the number of class
        all_labels: 1D numpy.ndarray. Save the index of labels.
    '''
    n_sample, num_class = all_probs.shape
    
    all_labels = make_one_hot(all_labels, num_class)
    n_precision = []
    for each_class in range(num_class):
        probs = all_probs[:,each_class]
        labels = all_labels[:,each_class]
        order = np.argsort(-probs) # Sort by confidence from largest to smallest
        probs = probs[order]
        labels = labels[order]
        precision = []
        recall = []
        for i in range(n_sample):
            pos_pred_label = labels[0:i+1]
            neg_pred_label = labels[i+1:]
            tp = np.sum(pos_pred_label)
            fp = len(pos_pred_label) - tp
            fn = np.sum(neg_pred_label)
            P = tp / (tp + fp + 1e-10)
            R = tp / (tp + fn + 1e-10)
            precision.append(P)
            recall.append(R)
        recall_change_index_0 = [] # The same recall value may correspond multiple precision values. So we take the largest precision value.
        for i in range(n_sample-1):
            if recall[i] != recall[i+1]:
                recall_change_index_0.append(i+1)
        recall_change_index_1 = recall_change_index_0[0:]
        recall_change_index_0.insert(0, 0)
        recall_change_index_1.append(n_sample)
        precision = np.array(precision)
        recall = np.array(recall)
        for i in range(len(recall_change_index_1)):
            index_0 = recall_change_index_0[i]
            index_1 = recall_change_index_1[i]
            precision[index_0:index_1] = np.max(precision[index_0:])
        unique_precision = []
        unique_precision.append(precision[0])
        for i in range(n_sample-1):
            if recall[i] != recall[i+1]:  # Only take precision when recall changes
                unique_precision.append(precision[i+1])
        n_precision.append(np.mean(unique_precision))
    
    mAP = np.mean(np.array(n_precision))
    return mAP

if __name__ == "__main__":
    probs = np.random.rand(100,10) # 100 samples, 10 classes
    labels = np.random.randint(0,9,(100))
    mAP = mean_average_precision(probs, labels)
    print(mAP)
