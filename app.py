import streamlit as st
import utils
import matplotlib.pyplot as plt
import io
import PIL.Image
import PIL

st.title("Monkey Pupil Size Application")

#video = st.file_uploader("Monkey Video", type=["mp4"], accept_multiple_files=False)

if "file_uploader_key" not in st.session_state:
    st.session_state["file_uploader_key"] = 0

if "uploaded_files" not in st.session_state:
    st.session_state["uploaded_files"] = []

files = st.file_uploader(
    "Monkey Videos",
    accept_multiple_files=True,
    key=st.session_state["file_uploader_key"],
)

if files:
    st.session_state["uploaded_files"] = files

if st.button("Clear uploaded files"):
    st.session_state["file_uploader_key"] += 1
    st.experimental_rerun()

st.write("Uploaded files:", st.session_state["uploaded_files"])

if files:
    video = files.pop(0)
    if video is not None:
        g = io.BytesIO(video.read())
        loc = "current_video.mp4"
        with open(loc, mode="wb") as f:
            f.write(g.read())
    
        with st.spinner("PROCESSING VIDEO") as s:
    
            model = utils.get_model()
    
            pupil_areas = utils.process_video(loc, model)
    
            graph = utils.plot_pupil_area(pupil_areas)
    
            buf = io.BytesIO()
            img = PIL.Image.open("output.png")
            img = img.convert("RGB")
            img.save(buf, format="JPEG")
            bytes_image = buf.getvalue()
    
            #st.pyplot(graph, use_container_width=True)
    
        st.download_button("Download Graphs", data=bytes_image, file_name="output.png", mime="image/jpeg", )
    
        #st.download_button("Download Values", pupil_areas)
