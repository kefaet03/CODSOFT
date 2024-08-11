import pickle

with open("F:\\He_is_enough03 X UniqoXTech X Dreams\\Machine Learning\\movie genre classification\\tfidf_vectorizer.pkl", 'rb') as file:
    tfidf_vectorizer = pickle.load(file)

with open("F:\\He_is_enough03 X UniqoXTech X Dreams\\Machine Learning\\movie genre classification\\nb_classifier.pkl", 'rb') as file:
    nb_classifier = pickle.load(file)

def predict_genre(description):
    description_tfidf = tfidf_vectorizer.transform([description])     # Transform the input description using the loaded TF-IDF vectorizer
    
    predicted_genre = nb_classifier.predict(description_tfidf)     # Predict the genre using the loaded classifier
    
    return predicted_genre[0]

new_description = "In pursuit of a serial killer, an FBI agent uncovers a series of occult clues that she must solve to end his terrifying killing spree."

predicted_genre = predict_genre(new_description)

print(f"The predicted genre is: {predicted_genre}")
