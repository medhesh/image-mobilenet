import streamlit as st
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input,decode_predictions
from PIL import Image 
import cv2
import numpy as np

@st.cache
def load_model():
  model = MobileNetV2()
  return model

st.title("Image Classification - Image NET")
st.write("1000 Classes")
upload = st.sidebar.file_uploader(label = 'Upload the Image')
if upload is not None:
  file_bytes = np.asarray(bytearray(upload.read()),dtype=np.uint8)
  opencv_image = cv2.imdecode(file_bytes,1)
  opencv_image = cv2.cvtColor(opencv_image,cv2.COLOR_BGR2RGB)
  img = Image.open(upload)
  st.image(img,caption='Uploade Image')
  model = load_model()
  if st.button('PREDICT'):
    st.sidebar.write("RESULT:")
    x = cv2.resize(opencv_image,(224,224))
    x = np.expand_dims(x,axis=0)
    x = preprocess_input(x)
    y = model.predict(x)
    label = decode_predictions(y)
    for i in range(3):
      out = label[0][i]
      st.sidebar.title(f'{out[1]}: {out[2]*100 :.3f}%')
