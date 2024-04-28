def get_new_videos(session_state, current_videos):
    session_state_names = [video.name for video in session_state["uploaded_files"]]
    current_video_names = [video.name for video in current_videos]
    current_video_map = {video.name: video for video in current_videos}
    diff_videos = set(session_state_names).symmetric_difference(set(current_video_names))
    diff_videos_output = [current_video_map[name] for name in diff_videos if name in current_video_map]
    session_state["uploaded_files"] = current_videos
    return diff_videos_output, session_state
