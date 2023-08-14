import streamlit as st
import pickle
import re
import nltk

nltk.download('punkt')
nltk.download('stopwords')

#loading models
clf = pickle.load(open('D:\pythonProject\ML_task\saved_model\model2.pkl','rb'))
tfidfd = pickle.load(open('D:\pythonProject\ML_task\saved_model\word_vectorizer.pkl','rb'))

def clean_resume(resume_text):
    clean_text = re.sub('http\S+\s*', ' ', resume_text)
    clean_text = re.sub('RT|cc', ' ', clean_text)
    clean_text = re.sub('#\S+', '', clean_text)
    clean_text = re.sub('@\S+', '  ', clean_text)
    clean_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', r' ', clean_text)
    clean_text = re.sub('\s+', ' ', clean_text)
    return clean_text
# web app
def main():
    st.title("Resume Screening App")
    uploaded_file = st.file_uploader('Upload Resume', type=['txt','pdf'])

    if uploaded_file is not None:
        try:
            resume_bytes = uploaded_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            # If UTF-8 decoding fails, try decoding with 'latin-1'
            resume_text = resume_bytes.decode('latin-1')

        cleaned_resume = clean_resume(resume_text)
        input_features = tfidfd.transform([cleaned_resume])
        prediction_id = clf.predict(input_features)[0]
        st.write(prediction_id)

        # Map category ID to category name
        category_mapping = {
            19: "HR",
            13: "DESIGNER",
            20: "INFORMATION-TECHNOLOGY",
            23: "TEACHER",
            1: "ADVOCATE",
            9: "BUSINESS-DEVELOPMENT",
            18: "HEALTHCARE",
            17: "FITNESS",
            2: "AGRICULTURE",
            8: "BPO",
            22: "SALES",
            12: "CONSULTANT",
            14: "DIGITAL-MEDIA",
            5: "AUTOMOBILE",
            10: "CHEF",
            16: "FINANCE",
            3: "APPAREL",
            15: "ENGINEERING",
            0: "ACCOUNTANT",
            11: "CONSTRUCTION",
            21: "PUBLIC-RELATIONS",
            7: "BANKING",
            4: "ARTS",
            6: "AVIATION",
            
        }

        category_name = category_mapping.get(prediction_id, "Unknown")

        st.write("Predicted Category:", category_name)



# python main
if __name__ == "__main__":
    main()