from setuptools import setup, find_packages

setup(
    name="catr",
    version="0.1.0",
    author="Sahith Reddy Thummala, Manish Yerram",
    author_email="sahithreddy.t@gmail.com",
    description="CATR: Transformer-based Image Captioning",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/sahithreddythummala/CATR",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.7.0",
        "torchvision>=0.8.0",
        "numpy>=1.19.0",
        "Pillow>=8.0.0",
        "matplotlib>=3.3.0",
        "tqdm>=4.50.0",
        "streamlit>=1.0.0",
        "transformers>=4.0.0",
        "gTTS>=2.2.0",
        "requests>=2.25.0",
    ],
)