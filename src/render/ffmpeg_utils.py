import ffmpeg
import os

def build_video(config, filepath, output, frame_count):
    """
    FFMPEG Wrapper to convert rendered frames into video and mux with original audio
    """
    try:
        # Building silent video
        video_stream = build_video_from_frames(
            config.temp_dirs,
            output,
            config.samplerate,
            frame_count
        )

        video_stream.run(
            overwrite_output=True,
            capture_stdout=True,
            capture_stderr=True
        )

        # Muxing audio

        temp_output_path = output.with_name(output.stem + "_temp.mp4")

        mux_audio_video(
            temp_output_path,
            filepath,
            output
        ).run(
            overwrite_output=True,
            capture_stdout=True,
            capture_stderr=True
        )

    except ffmpeg.Error as e:
        print("stdout:", e.stdout.decode("utf8"))
        print("stderr:", e.stderr.decode("utf8"))
        raise e

    finally:
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)


def build_video_from_frames(temp_dir, output, samplerate, frame_count):
    """
    Converts rendered image sequence into silent video using ffmpeg.
    """

    pattern = f'./{temp_dir}/frame_%0{len(str(frame_count))}d.jpg'
    temp_output_path = output.with_name(output.stem + "_temp.mp4")

    return (
        ffmpeg
        .input(pattern, framerate=samplerate)
        .output(
            str(temp_output_path),
            vf="pad=ceil(iw/2)*2:ceil(ih/2)*2"
        )
    )

def mux_audio_video(video_path, audio_path, output_path):
    """
    Combines silent video with original audio track.
    """

    video = ffmpeg.input(video_path)
    audio = ffmpeg.input(audio_path)

    return (
        ffmpeg
        .concat(video, audio, v=1, a=1)
        .output(str(output_path))
    )