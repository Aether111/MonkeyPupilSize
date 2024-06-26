import streamlit as st
import utils
import loader
import matplotlib.pyplot as plt
import io
import PIL.Image
import PIL
import pandas as pd
import zipfile
import os

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

to_process = []

if files:
    to_process, st.session_state = loader.get_new_videos(st.session_state, files)

if "data" not in st.session_state:
    st.session_state["data"] = []

if "values" not in st.session_state:
    st.session_state["values"] = []

if st.button("Clear uploaded files"):
    st.session_state["file_uploader_key"] += 1
    st.session_state["data"] = []
    st.session_state["values"] = []
    st.rerun()

st.write("Uploaded files:", st.session_state["uploaded_files"])

for i, video in enumerate(to_process):
    if video is not None:
        g = io.BytesIO(video.read())
        loc = "current_video.mp4"
        with open(loc, mode="wb") as f:
            f.write(g.read())
    
        with st.spinner(f"PROCESSING {video.name}") as s:
    
            model = utils.get_model()
    
            pupil_areas = utils.process_video(loc, model)
    
            graph = utils.plot_pupil_area(pupil_areas, i)
    
            buf = io.BytesIO()
            img = PIL.Image.open(f"output{i}.png")
            img = img.convert("RGB")
            img.save(buf, format="JPEG")
            bytes_image = buf.getvalue()

            st.session_state["data"].append((video.name, bytes_image))

            # Convert the numpy array to a DataFrame
            df = pd.DataFrame(pupil_areas)
        
            # Convert DataFrame to CSV
            csv = df.to_csv(index=True).encode('utf-8')

            st.session_state["values"].append((video.name, csv))
    
            #st.pyplot(graph, use_container_width=True)

for (name_image, bytes_image), (name_csv, csv) in zip(st.session_state["data"], st.session_state["values"]):
    buffer = io.BytesIO()
    name = os.path.basename(name_csv)
    with zipfile.ZipFile(buffer, "x") as zip:
        zip.writestr(name+".csv", csv)
        zip.writestr(name+".png", bytes_image)
    st.download_button(f"Download {name} Package", data=buffer.getvalue(), file_name=f"{name}.zip", mime="application/zip", )
