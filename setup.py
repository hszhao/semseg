import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="acosp",
    version="1.0.0",
    author="Konstantin Ditschuneit",
    author_email="konstantin.ditschuneit@merantix.com",
    description="Implementation of the ACoSP compression algorithm, including all reported experiments.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/merantix/acosp",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "fire==0.4.0",
        "matplotlib==3.4.3",
        "numpy==1.20.3",
        "opencv_python_headless==4.5.4.58",
        "Pillow==8.4.0",
        "PyYAML==6.0",
        "tensorboardX==2.4.1",
        "torch==1.10.0",
        "torchvision>=0.11.0",
        "tqdm==4.62.3",
    ]
)
