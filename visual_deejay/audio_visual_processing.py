import os

__all__ = ['blend_audio_and_video', 'create_video_from_images', 'crop_song']

# TODO: convert all functions to Python bindings +=> proper error handling


def blend_audio_and_video(song_file, video_file, out_video_file):
    os.system(f'ffmpeg -i {video_file} -i {song_file}.mp3 -c copy -shortest -map 0:v:0 -map 1:a:0 {out_video_file}')


def create_video_from_images(image_dir, fps, res, video_file):
    """ Inspired by http://hamelot.io/visualization/using-ffmpeg-to-convert-a-set-of-images-into-a-video/

    :param image_dir: (Path) directory in which images are stored
    :param fps: (int) frames-per-second we want to render the images as
    :param res: (int) resolution of the images in pixels
    :param video_file: (Path) file_name of output video file
    :return:
    """

    os.system(f'ffmpeg -r {fps} -f image2 -s {res}x{res} -i {image_dir}/%01d.jpg -vcodec libx264 -crf 25 -pix_fmt yuv420p {video_file}')


def crop_song(song_file, duration):
    song_minus_ext, song_ext = song_file.split('.')
    os.system(f'ffmpeg -t {duration} -i {song_file} -acodec copy {song_minus_ext}_{duration}s.{song_ext}')
