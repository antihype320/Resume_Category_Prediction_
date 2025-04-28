# Resume Category Prediction

## Project Overview
This project involves building a machine learning model to automatically classify resumes into different categories based on their content. The system uses natural language processing (NLP) techniques, including text preprocessing and feature extraction, to transform raw resume data into a format suitable for machine learning algorithms. We then apply multiple classifiers, including K-Nearest Neighbors, Support Vector Machines, and Random Forests, to predict the category of a given resume.

<img 
  src="https://github.com/user-attachments/assets/96b0533d-bfc6-44da-b773-c8cd305d6841" 
  style="border: 4px solid white; border-radius: 8px; box-shadow: 0 0 12px rgba(255, 255, 255, 0.2); background: rgba(255, 255, 255, 0.1); padding: 5px"
/>

![image2](https://github.com/user-attachments/assets/04a447da-85a1-4b86-bf73-5e7e0e031511)



### Key Features:
- Text preprocessing to clean and normalize resume data.
- TF-IDF vectorization to transform resumes into numerical features.
- Multiple classification algorithms to predict resume categories.
- Model training and evaluation with accuracy, confusion matrix, and classification reports.
- Model persistence using **pickle** for easy loading and predictions.

Additionally, a **Streamlit-based web application** has been developed to allow users to upload resumes in various formats (PDF, DOCX, CSV) and receive real-time predictions on the category of the resume. The app also extracts contact information such as email and phone numbers from the resumes.

---

## Project Flow

1. **Data Loading**  
   The dataset, **UpdatedResumeDataSet.csv**, contains resume information with the corresponding **Category** labels (e.g., "Engineering", "Marketing").

2. **Data Preprocessing**  
   - Clean text data by removing URLs, special characters, and unnecessary symbols.
   - Encode the categorical labels into numeric format using **LabelEncoder**.

3. **Feature Extraction**  
   - Transform resume text into numerical features using **TF-IDF Vectorizer**.

4. **Model Training**  
   - Split the data into training and testing sets using **train_test_split**.
   - Train multiple machine learning models including **K-Nearest Neighbors (KNN)**, **Support Vector Classifier (SVC)**, and **Random Forest**.
   - Evaluate the models using accuracy scores, confusion matrices, and classification reports.

5. **Model Persistence**  
   - Save the trained vectorizer, classifier, and label encoder to disk using **pickle**.

6. **Prediction Function**  
   - A function to predict the category of a new resume by preprocessing, vectorizing, and using the trained classifier.

7. **Web Application**  
   - **Streamlit** app allows users to upload resumes in PDF, DOCX, or CSV format.
   - Extracted resume text is processed and classified in real-time.
   - The app displays the predicted category along with any contact information (email, phone) found in the resume.

---

## Technologies Used

- **Pandas**
- **NumPy**
- **Matplotlib & Seaborn**
- **Scikit-learn**
  - **KNeighborsClassifier**
  - **SVC** 
  - **RandomForestClassifier**
- **Pickle**
- **Regular Expressions (re)**
- **Streamlit**
- **PyPDF2**
- **python-docx**
