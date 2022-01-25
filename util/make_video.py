import argparse
import os
from typing import List, Tuple


def images_to_video(
    image_pattern: str,
    output_file: str,
    use_glob: bool = True,
    scale: Tuple[int, int] = (1280, 650),
) -> None:
    """This function takes images and put them into a video.

    Args:
        image_pattern: an pattern for all images.
            For example "images/%d_real_b.png" will match files like "images/5_real_b.png"
        output_file: the output file for example "videos/real_b.mp4"
        scale: scaling each image to this dimension so you can scale each image
            to format 1280x650
    """
    cmd = "ffmpeg -y "
    if use_glob:
        cmd += "-pattern_type glob "
    cmd += f'-i "{image_pattern}" '
    cmd += "-codec:v libx264 -preset veryslow -pix_fmt yuv420p -crf 28 "

    if scale is not None:
        cmd += f"-vf scale={scale[0]}:{scale[1]} "

    cmd += f"-an {output_file} "
    os.system(cmd)


def make_2x2_video_grid(
    video_paths: List[str],
    output_file: str,
):
    """This function takes 4 images and puts them into a 2x2 Grid.

    Args:
        video_paths: array of paths to the 4 videos
        output_file: output file for the resulting video
    """
    cmd = "ffmpeg -y "
    for video_path in video_paths:
        cmd += f'-i "{video_path}" '
    cmd += (
        "-filter_complex"
        '"[0:v][1:v]hstack=inputs=2[top];[2:v][3:v]hstack=inputs=2[bottom];'
        '[top][bottom]vstack=inputs=2[v]" '
    )
    cmd += '-map "[v]" '
    cmd += output_file
    os.system(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Images to Video")
    parser.add_argument(
        "--image_pattern",
        type=str,
        help="pattern for all images you want to include",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="out.mp4",
        help="the output file for the video",
    )
    args = parser.parse_args()
    images_to_video(args.image_pattern, args.output_file)
