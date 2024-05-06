import streamlit as st
import utils
import loader
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

to_process = []

if files:
    to_process, st.session_state = loader.get_new_videos(st.session_state, files)

if "data" not in st.session_state:
    st.session_state["data"] = []

if "values" not in st.session_state:
    st.session_state["values"] = []

if st.button("Clear uploaded files"):
    st.session_state["file_uploader_key"] += 1
    st.experimental_rerun()

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
            csv = df.to_csv(index=False).encode('utf-8')

            st.session_state["values"].append((video.name, csv))
    
            #st.pyplot(graph, use_container_width=True)
    
for name, bytes_image in st.session_state["data"]:
    st.download_button(f"Download {name} Graph", data=bytes_image, file_name=f"{name}.png", mime="image/jpeg", )

for (name, csv) in st.session_state["values"]:
    st.download_button(
        label="Download {name} CSV",
        data=csv,
        file_name=f'{name}.csv',
        mime='text/csv',
    )
