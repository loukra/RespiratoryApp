import streamlit as st
import streamlit.components.v1 as com
import streamlit as st
import os
import numpy as np
from io import BytesIO
import io
import soundfile as sf
from matplotlib import pyplot as plt
import keras
import tensorflow as tf
import predict
import preprocess
import librosa
import pyautogui

# DESIGN implement changes to the standard streamlit UI/UX
#st.set_page_config(page_title="streamlit_audio_recorder")
# Design move app further up and remove top padding
st.markdown('''<style>.css-1egvi7u {margin-top: -3rem;}</style>''',
    unsafe_allow_html=True)
# Design change st.Audio to fixed height of 45 pixels
st.markdown('''<style>.stAudio {height: 45px;}</style>''',
    unsafe_allow_html=True)
# Design change hyperlink href link color
st.markdown('''<style>.css-v37k9u a {color: #ff4c4b;}</style>''',
    unsafe_allow_html=True)  # darkmode
st.markdown('''<style>.css-nlntq9 a {color: #ff4c4b;}</style>''',
    unsafe_allow_html=True)  # lightmode

parent_dir = os.path.dirname(os.path.abspath(__file__))
# Custom REACT-based component for recording client audio in browser
build_dir = os.path.join(parent_dir, "st_audiorec/frontend/build")
# specify directory and initialize st_audiorec object functionality
st_audiorec = com.declare_component("st_audiorec", path=build_dir)


with open('style.css') as f:
    design = f.read()
com.html(f"""
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
{
design
}
</style>
</head>
<body>

<div style="text-align:center">
  <h1 style="font-family:Arial;color:">You Breath, We Classify</h1>
  <span class="dot"></span>
</div>

</body>
""")
@st.cache(show_spinner=False, allow_output_mutation=True)
def load_model():
  return keras.models.load_model('ResNet.h5', compile=False)

with st.sidebar:
  st.title("Respiratory Health Classifier")
  """This project is designed to detect whether your lungs are diseased or healthy. To do this, please record at least 8 seconds of your lung sounds. 
  The artificial neural network will then analyse it and give you a result.\n\n
  Please remember that this is not a medical diagnosis. 
  If in doubt, it is best to seek a doctor's opinion. """


col1, col2, col3 = st.columns([1,3,1])
with col2:
  if 'is_expanded' not in st.session_state:
      st.session_state['is_expanded'] = True

  holder = st.empty()
  with holder:
    val = st_audiorec()

  st.session_state['is_expanded'] = False
  if isinstance(val, dict):
    # retrieve audio data
    ind, val = zip(*val['arr'].items())
    ind = np.array(ind, dtype=int)  # convert to np array
    val = np.array(val)             # convert to np array
    sorted_ints = val[ind]
    stream = BytesIO(b"".join([int(v).to_bytes(1, "big") for v in sorted_ints]))
    audio_bytes = stream.read()

    if audio_bytes:
      data_origin, samplerate = sf.read(io.BytesIO(audio_bytes))

      wav = data_origin[:,0]

      data = librosa.resample(y=wav, orig_sr=samplerate, target_sr=4000)  
          #create preprocessor object
      print(data.shape)
      if data.shape[0] > 16 * 4000:
        holder.empty()
        with st.spinner('Asking the Doc...'):
            
            preprocessor = preprocess.AudioPreprocessor()

            #load model
            model = load_model()

            #create predictor object
            predictor = predict.MyPredictor(model, preprocessor)
            fs_mult = np.floor(data.shape[0] / 4000)
            #predict file
            data = data[: int(4000 * fs_mult)]
            print(data.shape)
            y_pred = predictor.predict(data)
        if y_pred > 0.5:
          st.success(f"There is a higher probability of about {5 * round((y_pred * 100)/5)} % that your respiratory system is healthy")
        else:
          st.error(f"There is a higher probability of about {5 * round(((1 - y_pred) * 100)/5)} % that your respiratory system are diseased. ")
      else: 
        st.error("The recording must be at least 16 seconds long to obtain a result.")


