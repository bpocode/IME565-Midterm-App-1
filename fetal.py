import streamlit as st
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Set up the title and description of the app
st.title('Fetal Health Classification: A Machine Learning App') 
df = pd.read_csv('fetal_health.csv')


# Display an image of penguins
st.image('fetal_health_image.gif', width = 600)
st.write('Utilize our advanced Machine Learning application to predict fetla health classifications.')

dt_pickle = open('dt_clf.pickle', 'rb') 
ada_pickle = open('ada_clf.pickle', 'rb')
rf_pickle = open('rf_clf.pickle', 'rb')
voting_pickle = open('voting_clf.pickle', 'rb')

dt_clf = pickle.load(dt_pickle) 
ada_clf = pickle.load(ada_pickle)
rf_clf = pickle.load(rf_pickle)
voting_clf = pickle.load(voting_pickle)

model_dict = {'Decision Tree': dt_clf,'AdaBoost': ada_clf,'Random Forest': rf_clf,'Soft Voting': voting_clf}
dt_pickle.close()
ada_pickle.close()
rf_pickle.close()
voting_pickle.close()

# Create a sidebar for input collection
st.sidebar.header('**Fetal Health Features Input**')
st.sidebar.write('Upload your data')
upload = st.sidebar.file_uploader("Choose a file")
st.sidebar.warning('⚠️Ensure your data strictly follows the format outlined below.')
st.sidebar.dataframe(df.drop(columns = ['fetal_health']).head())

model_select = st.sidebar.radio('Choose Model for Prediction', ('Random Forest','Decision Tree','AdaBoost','Soft Voting'))
st.sidebar.info('You selected: ' + model_select)

if upload is None:
    st.info('ℹ️Please upload data to proceed.')
elif upload is not None:
    st.success('✅CSV file uploaded successfully.')
    input = pd.read_csv(upload)
    model = model_dict[model_select]
    predictions = model.predict(input)
    probs = model.predict_proba(input) * 100
    input['Predicted Fetal Health'] = predictions
    input['Prediction Probability (%)'] = probs.max(axis=1) 

    categories = {1.0: 'Normal', 2.0: 'Suspect', 3.0: 'Pathological'}
    input['Predicted Fetal Health'] = input['Predicted Fetal Health'].map(categories)

    # Utilized generative AI to apply color coding to predictions
    color_map = {"Normal": "lime","Suspect": "yellow","Pathological": "orange"}
    def highlight_prediction(val):
        color = color_map.get(val, "white")  # default white if class not found
        return f"background-color: {color}"
    styled_df = input.style.applymap(highlight_prediction, subset=["Predicted Fetal Health"])
    st.dataframe(styled_df)
    
    con_mats = {'Decision Tree': 'confusion_mat_dt.svg','AdaBoost': 'confusion_mat_ada.svg',
                'Random Forest': 'confusion_mat_rf.svg', 'Soft Voting': 'confusion_mat_voting.svg'
    }

    feat_imps = {'Decision Tree': 'feature_imp_dt.svg','AdaBoost': 'feature_imp_ada.svg',
                'Random Forest': 'feature_imp_rf.svg', 'Soft Voting': 'feature_imp_voting.svg'
    }

    class_reports = {'Decision Tree': 'class_report_dt.csv','AdaBoost': 'class_report_ada.csv',
                'Random Forest': 'class_report_rf.csv', 'Soft Voting': 'class_report_voting.csv'
    }

    st.subheader("Prediction Performance and Insights")
    tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "Classification Report","Feature Importance"])
    with tab1:
        st.write("### Confusion Matrix")
        st.image(con_mats[model_select])
    with tab2:
        st.write("### Classification Report")
        cr = pd.read_csv(class_reports[model_select])
        st.dataframe(cr)
    with tab3:
        st.write("### Feature Importance")
        st.image(feat_imps[model_select])