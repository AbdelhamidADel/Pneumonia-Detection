import tensorflow as tf
import numpy as np
from PIL import Image
import streamlit as st
from streamlit_option_menu import option_menu
import base64

st.set_page_config(layout="centered",page_icon="👨‍⚕️",page_title="Pneumonia Detection")
# 2. horizontal menu
selected = option_menu(None, ['Detection', 'About'], 
    icons=['house',"list-task"], 
    menu_icon="cast", default_index=0, orientation="horizontal")

#----------------------------------------------------------------

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)
set_background('Lung.jpg')
# --------------------------------------------------Prediction PAGE-----------------------------------------------------------
 
if selected =='Detection':
    hide_streamlit_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
                """
    img_style="""<style>
    img {
    margin-left: 150px;
    display:relative;
    width: 100%;}
    </style>"""

    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
    st.markdown(img_style, unsafe_allow_html=True) 
    st.markdown("<h1 style='text-align: center; color: white;'>Pneumonia Detection </h1>", unsafe_allow_html=True)
    result1_msg = st.empty()
    result2_msg = st.empty()
    st.markdown("-------------------------------------------------------------------------------")

    def pneumoniapredictPage(imgg):
      img=Image.open(imgg)
      img = img.resize((36,36))
      img = np.asarray(img)
      img = img.reshape((1,36,36,1))
      img = img / 255.0
      model = tf.keras.models.load_model("pneumonia.h5")
      pred = np.argmax(model.predict(img)[0])
      if pred == 0:
        return "Normal"
      elif pred == 1:
        return "penumonia"
    
    try:
        imageLocation = st.empty()
        imageLocation.image('Default_Image_Thumbnail.png')
        imgg = st.file_uploader(label="Choose a picture : ", type=['jpeg', 'jpg', 'png'], key="xray")

        st.markdown("<h3 style='text-align: center; color: white;'>  </h3>", unsafe_allow_html=True)

        picture = st.camera_input("Take a picture : ")
        if picture is not None:
            showed_img=Image.open(picture)
            showed_img= showed_img.resize((256, 256))
            imageLocation.image(showed_img)

            result1= pneumoniapredictPage(picture)
            with result1_msg.container():
                if result1 == "Normal" : 
                    st.success("Your Lungs are Healthy")
                elif result1 == "penumonia":
                    st.error("There is Pneumonia, You Should Go to The Doctor")
        
        if imgg is not None:
            #show to ui
            showed_img=Image.open(imgg)
            showed_img= showed_img.resize((256, 256))
            imageLocation.image(showed_img)
            # for prediction
            result2= pneumoniapredictPage(imgg)
            with result2_msg.container():
                if result2 == "Normal" : 
                    st.success("Your Lungs are Healthy")
                elif result2 == "penumonia":
                    st.error("There is Pneumonia, You Should Go to The Doctor")
    except:
        st.markdown("<h3 style='text-align: center; color: white;'>Try Another Pictuer !</h3>", unsafe_allow_html=True)



# --------------------------------------------------ABOUT PAGE-----------------------------------------------------------
if selected =='About':
    hide_streamlit_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
    st.markdown("<h1 style='text-align: center; color: white;'>How I Am ?</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: white;'>My Name is Abdelhamid Adel</h3>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: white;'>Data Scientist | Data analyst l knowledgeable in Machine learning - Deep learning - NLP - Computer Vision</h3>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: white;'> I am From Cairo, Egypt</h3>", unsafe_allow_html=True)

    button_styl="""<style>
    button {
  background: #FF4742;
  border: 1px solid #FF4742;
  border-radius: 6px;
  box-shadow: rgba(0, 0, 0, 0.1) 1px 2px 4px;
  box-sizing: border-box;
  color: #FFFFFF;
  cursor: pointer;
  display: inline-block;
  font-family: nunito,roboto,proxima-nova,"proxima nova",sans-serif;
  font-size: 16px;
  font-weight: 800;
  line-height: 16px;
  min-height: 40px;
  outline: 0;
  padding: 12px 14px;
  text-align: center;
  text-rendering: geometricprecision;
  text-transform: none;
  user-select: none;
  -webkit-user-select: none;
  touch-action: manipulation;
  vertical-align: middle;
  margin: 0;
  position: absolute;
  top: 50%;
  left: 50%;
  -ms-transform: translate(-50%, -50%);
  transform: translate(-50%, -50%);
}

button:hover,
button:active {
  background-color: initial;
  background-position: 0 0;
  color: #FF4742;
}

button:active {
  opacity: .5;
}

    </style>"""

    st.markdown(button_styl, unsafe_allow_html=True) 


    st.write(f'''
    <a target="_blank" href="https://github.com/AbdelhamidADel">
        <button>
            My GitHub
        </button>
    </a>
    ''',
    unsafe_allow_html=True)