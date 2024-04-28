def get_new_videos(session_state, current_videos):
    diff_videos = set(session_state["current_videos"]).symmetric_difference(set(current_videos))
    session_state["current_videos"] = current_videos
    return list(diff_videos), session_state
