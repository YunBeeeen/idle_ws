from glob import glob

from setuptools import find_packages, setup

package_name = "idle_vision"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/rviz", glob("rviz/*.rviz")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="su",
    maintainer_email="lsu031111@hanyang.ac.kr",
    description="Camera vision pipeline for D435, USB RGB cameras, HSV boxes, and voice/Qwen selection.",
    license="Apache-2.0",
    extras_require={
        "test": [
            "pytest",
        ],
    },
    entry_points={
        "console_scripts": [
            "box_pose_node = idle_vision.box_pose_node:main",
            "image_learning_node = idle_vision.image_learning_node:main",
            "hsv_tuner_node = idle_vision.hsv_tuner_node:main",
            "object_depth_node = idle_vision.object_depth_node:main",
            "qwen_box_selector_node = idle_vision.qwen_box_selector_node:main",
            "voice_command_node = idle_vision.voice_command_node:main",
        ],
    },
)
