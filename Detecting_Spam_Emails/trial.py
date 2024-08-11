import pickle
import tensorflow as tf

# Load the TfidfVectorizer
with open("F:\\He_is_enough03 X UniqoXTech X Dreams\\Machine Learning\\Detecting_Spam_Emails\\tfidf_vec.pkl", 'rb') as file:
    tfidf_vec = pickle.load(file)

# Load the MultinomialNB model
with open("F:\\He_is_enough03 X UniqoXTech X Dreams\\Machine Learning\\Detecting_Spam_Emails\\baseline_model.pkl", 'rb') as file:
    baseline_model = pickle.load(file)

# Example input for prediction
new_message = "Oh my gosh!!! I am SO excited for tomorrow’s (Tuesday 7/23) launch party in Oakland…! I am expecting TONS of fabulous people, some fascinating discussions, and an overall total “lovefest”—Marie style ;)Come hang out, get your book signed, take selfies, hear me talk from the heart about compersion, and get inspired by a panel of several wise individuals.I couldn’t be more honored to be joined by Dossie Easton (The Ethical Slut’s co-author), William Winters (Bonobo Network founder & co-lead), Emma & Fin (Normalizing Non-Monogamy Podcast hosts & community leaders), and Alexandra Ballensweig (humhum founder & facilitator), in addition to several other pioneers, beloved friends, and colleagues.Event schedule:6:30-7:30: Mingling, Book Signing7:30-8:00: Marie's Speech & Book Reading + Q&A8:00-8:30: Expert Panel w/Dossie, William, Emma & Fin (moderated by Alexandra) + Q&A8:30-9:30: Mingling, Book Signing... and Dancing, if the mood is right!Please come & help me celebrate the launch of my book! 😎There’s no cost to attend, but please sign up HERE so I know how many people to expect (and how many snacks to buy).Can’t wait to see you there!Love, Marie xo"

# Transform the input using the TfidfVectorizer
new_message_tfidf = tfidf_vec.transform([new_message])

# Predict using the MultinomialNB model
predicted_label_nb = baseline_model.predict(new_message_tfidf)
print("ham: 0 || spam: 1")
print(f"Predicted label (NB): {predicted_label_nb[0]}")

