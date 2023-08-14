# Resume-Screnner

Here I have trained the dataset with two models.
1. OneVsRestClassifier with KNeighborsClassifier as the estimator
2. Multinomial Naive Bayes classifier

It seems that MultinomialNB has achieved a higher training score and a nearly the same validation score compared to the OnevsRestClassifier and that is why i decided to save the Multinomial model and use it for writing the script.
By thorough and repetitive experimentations, using RandomizedSearchCV or GridSearchCV, the hyperparameters could be finetuned by adjusting the various parameters of the model to see whehther there is a rise in the training and validation scores. As I am trying to predict multiple classes, therefore I went with these two models as they are well known for multi class classification.

To run the script2.py file create a folder for resumes named resume_dir and a folder named csv_ouput. Then run:

  *python script2.py --path to the resume directory --path to the csv output directory*


I have also created a streamlit app to categorize the resume and show the predicted category. By running
  *streamlit run streamlit_app.py*

you would be directed to the local app in your browser.
