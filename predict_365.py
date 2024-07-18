import pickle
import numpy as np
import streamlit as st
import time
from PIL import Image
import gzip

# st.set_page_config(layout="wide")

st.markdown(
    """
    <style>
    .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob,
    .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137,
    .viewerBadge_text__1JaDK {
        display: none;
    }
    font="sans serif"
    </style>
    """,
    unsafe_allow_html=True
)

footerText = """
<style>
#MainMenu {
visibility:hidden ;
}

footer {
visibility : hidden ;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: transparent;
color: white;
text-align: center;
}
</style>

<div class='footer'>
<p> Copyright @ 2023 Center for Digital Health <a href="mailto:iceanon1@khu.ac.kr"> iceanon1@khu.ac.kr </a></p>
</div>
"""

st.markdown(str(footerText), unsafe_allow_html=True)


@st.cache_data
def model_file():
    with gzip.open('model_rf.pkl.gz', 'rb') as f:
        model = pickle.load(f)


    return model


def prediction(X_test):
    model = model_file()
    result = model.predict([X_test])
    result = np.exp(result)
    return result

def ci_95(input):
    pre=[]
    for i in range(100):
        noise = np.random.normal(0, 0.1, size=len(input))
        input_with_noise = input + noise

        result=prediction(input_with_noise)
        pre.append(result)
    mean_prediction = np.mean(pre)
    std_prediction = np.std(pre)
    lower_bound = mean_prediction - 1.96 * std_prediction
    upper_bound = mean_prediction + 1.96 * std_prediction
    return lower_bound,upper_bound

def input_values():
    img=Image.open('img.png')
    st.image(img)
    st.markdown("""---""")

    sex = st.radio('Sex', ('Male', 'Female'), horizontal=True)
    sexDict = {'Male': 0, 'Female': 1}
    sex = sexDict[sex]

    age = st.number_input('Age', min_value=10, max_value=100, value=30)

    Marital_Status = st.radio('Marital status', ('Married', 'Single', 'Divorced'), horizontal=True)
    Marital_StatusDict = {'Married': 0, 'Single': 1, 'Divorced': 1}
    Marital_Status = Marital_StatusDict[Marital_Status]

    height = st.number_input('Height (cm)', min_value=130.0, max_value=200.0, value=163.0)
    weight = st.number_input('Weight (kg)', min_value=30.00, max_value=150.0, value=70.0)
    bmiv = weight / ((height / 100) ** 2)
    bmi=round(bmiv, 1)
    st.write('BMI: ', bmi)

    bodywater = st.number_input('Body water (l)', min_value=10.0, max_value=50.0, value=33.0)
    bodyprotein = st.number_input('Body protein (kg)', min_value=1.0, max_value=20.0, value=9.0)
    bodyminerals = st.number_input('Body minerals (kg)', min_value=0.0, max_value=10.0, value=3.0)
    bodyfat = st.number_input('Body fat mass (kg)', min_value=0.0, max_value=50.0, value=26.0)
    muscle = st.number_input('Skeletal muscle mass (kg)', min_value=0.0, max_value=80.0, value=45.0)
    muscle_fat = st.number_input('Intramuscular Fat (kg)', min_value=0.0, max_value=50.0, value=26.0)
    muscle_bodyfat = st.number_input('Muscle to Fat Ratio (kg)', min_value=0.0, max_value=150.0, value=93.0)
    before_abdomen_cm = st.number_input('Preoperative waist circumference (cm)', min_value=50.0, max_value=200.0, value=100.0)

    X_test = [age,sex,Marital_Status,weight,height,bodywater,bodyprotein,bodyminerals,bodyfat,muscle,muscle_fat,muscle_bodyfat,bmi,before_abdomen_cm]

    return X_test


def main():
    X_test = input_values()

    result = prediction(X_test)
    lower_bound, upper_bound = ci_95(X_test)

    result=round(result[0],1)
    lower_bound = round(lower_bound,1)
    upper_bound = round(upper_bound,1)
    with st.sidebar:
        img2 = Image.open('img_1.png')
        st.image(img2)

        saved_text=""
        lower_bound_text = ""
        upper_bound_text = ""
        st.markdown(f'# Liposuction volume (cc)')
        if st.button("result"):
            # 버튼이 눌렸을 때 실행할 동작
            saved_text = result
            lower_bound_text = lower_bound
            upper_bound_text = upper_bound
            st.markdown(f'# {saved_text}')
            st.markdown(f'# 95% CI ({lower_bound_text}, {upper_bound_text})')
    now = time
    print(now.strftime('%Y-%m-%d %H:%M:%S'))


if __name__ == '__main__':
    main()
