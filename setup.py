from setuptools import setup, find_packages
import os

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='smartzoom',
    version='0.1.0',
    author="Alex M",
    author_email="alexmalone489@gmail.com",
    description="A Python utility to automatically apply a smooth, continuous zoom to a video, perfectly framing its content.",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/AMAMazing/smartzoom",
    keywords=["video", "zoom", "automation", "opencv", "ffmpeg", "cli"],
    packages=find_packages(),
    install_requires=[
        'opencv-python',
        'numpy',
    ],
    entry_points={
        'console_scripts': [
            'smartzoom=smartzoom.main:cli',
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Multimedia :: Video", 
    ],
    python_requires=">=3.6",
)