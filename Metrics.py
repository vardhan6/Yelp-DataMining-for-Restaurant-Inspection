def print_classifier_performance_metrics(name, predictions):
    target_names = ['1 star', '2 star', '3 star', '4 star', '5 star']
    
    print "MODEL: %s" % name
    print

    print 'Precision: ' + str(metrics.precision_score(test_y, predictions))
    print 'Recall: ' + str(metrics.recall_score(test_y, predictions))
    print 'F1: ' + str(metrics.f1_score(test_y, predictions))
    print 'Accuracy: ' + str(metrics.accuracy_score(test_y, predictions))

    print
    print 'Classification Report:'
    print classification_report(test_y, predictions, target_names=target_names)
    
    print
    print 'Precision variance: %f' % np.var(precision_score(test_y, predictions, average=None), ddof=len(target_names)-1)
    
    print
    print 'Recall variance: %f' % np.var(recall_score(test_y, predictions, average=None), ddof=len(target_names)-1)

print_classifier_performance_metrics('Multinomial Naive Bayes', multinomial_nb_prediction)	