bernoulli_feature_matrix_train = binary_vectorizer.fit_transform(train_X)
bernoulli_feature_matrix_test = binary_vectorizer.transform(test_X)
bernoulli_feature_matrix_train, bernoulli_feature_matrix_test

bernoulli_nb_classifier = BernoulliNB().fit(bernoulli_feature_matrix_train, train_y)
bernoulli_nb_prediction = bernoulli_nb_classifier.predict(bernoulli_feature_matrix_test)

bernoulli_confusion_matrix = confusion_matrix(test_y, bernoulli_nb_prediction)

print_classifier_performance_metrics('Bernoulli Naive Bayes', bernoulli_nb_prediction)