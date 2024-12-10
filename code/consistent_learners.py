import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron, LogisticRegression



def train_classifier(X, y, classifier_name='svm'):

    classifier = None
    if(classifier_name=='svm'):
        classifier = SVC(kernel='linear')
    elif(classifier_name=='perceptron'):
        classifier = Perceptron(max_iter=100000, tol=1e-5)
    elif(classifier_name=='logistic'):
        classifier = LogisticRegression(max_iter=10000, tol=1e-5)
    else:
        raise ValueError("wrong classifier type!")
    
    classifier.fit(X, y)
    predictions = classifier.predict(X)
    training_loss = np.sum(predictions != y)
    print("Training loss ", classifier_name, " :", training_loss)

    return classifier

def evaluate_classifier(classifier, S, A, Phi_dic, pi):
    # Extract the weight vector w from the trained SVM classifier
    w = classifier.coef_[0]

    # Predict actions for the entire set S
    predicted_actions = []
    for s_id in range(len(S)):
        scores = {a: np.dot(w, Phi_dic[(s_id, a)]) for a in A}
        predicted_action = max(scores, key=scores.get)
        predicted_actions.append(predicted_action)

    # Calculate the error
    error = sum(1 for s_id in range(len(S)) if predicted_actions[s_id] != pi[s_id])

    return error


