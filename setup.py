from setuptools import setup, find_packages

setup(
    name="openlaunch",
    version="0.1.0",
    description="Open-source golf launch monitor and swing coach — iPhone camera + computer vision + ball flight physics",
    packages=find_packages(),
    install_requires=[
        "opencv-python-headless>=4.8",
        "mediapipe>=1.0",
        "numpy>=1.24",
        "scipy>=1.11",
        "matplotlib>=3.7",
    ],
    entry_points={
        "console_scripts": [
            "openlaunch=openlaunch.cli:main",
        ],
    },
    python_requires=">=3.10",
)
