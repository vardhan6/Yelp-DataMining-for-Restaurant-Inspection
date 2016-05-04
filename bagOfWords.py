bow_feature_matrix_train = bag_of_words_vectorizer.fit_transform(train_X)
bow_feature_matrix_test = bag_of_words_vectorizer.transform(test_X)
bow_feature_matrix_train, bow_feature_matrix_test

multinomial_nb_classifier = MultinomialNB()
multinomial_nb_classifier.fit(bow_feature_matrix_train, train_y)
multinomial_nb_prediction = multinomial_nb_classifier.predict(bow_feature_matrix_test)

def make_confusion_matrix_relative(confusion_matrix):
    star_category_classes = [1, 2, 3, 4, 5]
    N = map(lambda clazz : sum(test_y == clazz), star_category_classes)
    relative_confusion_matrix = np.empty((len(star_category_classes), len(star_category_classes)))
    
    for j in range(0, len(star_category_classes)):
        if N[j] > 0:
            relative_frequency = confusion_matrix[j, :] / float(N[j])
            relative_confusion_matrix[j, :] = relative_frequency
            
    return relative_confusion_matrix
	
	
multinomial_confusion_matrix = confusion_matrix(test_y, multinomial_nb_prediction)
print make_confusion_matrix_relative(multinomial_confusion_matrix)	