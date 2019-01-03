import read
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

TUNING = True  # Set this to False if you don't want to tune

print ("Reading data ...")
x_all, y_all = read.read(LOAD_DATA=False)
x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.3, random_state=42)
print (x_train.shape, y_train.shape)
print (x_test.shape, y_test.shape)

if TUNING:

    tuned_parameters = [{'gamma': [1e-3, 1e-2, 1/8],
                         'C': [1, 10, 100, 1000]}]

    scores = ["accuracy", "f1"]

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print("")

        clf = GridSearchCV(SVC(kernel="rbf"), tuned_parameters, cv=5, scoring=score, n_jobs=2)
        clf.fit(x_train, y_train)

        print("Best parameters set found on development set:")
        print("")
        print(clf.best_estimator_)
        print (clf.best_params_)
        print("")
        print("Grid scores on development set:")
        print("")
        print (clf.best_score_)
        print("")

        print("Detailed classification report:")
        print("")
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print("")
        y_true, y_pred = y_test, clf.predict(x_test)
        print(classification_report(y_true, y_pred))
        print("")

else:
    models = [SVC(),
              SVC(C=1000, cache_size=200, class_weight=None, coef0=0.0,
                  decision_function_shape='ovr', degree=3, gamma=0.01, kernel='rbf',
                  max_iter=-1, probability=False, random_state=None, shrinking=True,
                  tol=0.001, verbose=False)
              ]
    for model in models:
        print ("Fitting SVM ...")
        model.fit(x_train, y_train)

        print ("Evaluating ...")
        y_pred = model.predict(x_test)

        print ("Accuracy is %f." % accuracy_score(y_test, y_pred))
        print (confusion_matrix(y_test, y_pred))
        print ("Precision score is %f." % precision_score(y_test, y_pred))
        print ("Recall score is %f." % recall_score(y_test, y_pred))
        print ("F1 score is %f." % f1_score(y_test, y_pred))
        print ("-----------------------------------")


