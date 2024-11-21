from setuptools import setup, find_packages

setup(
    name="optiattack_client",
    version="0.1.0",
    description="A FastAPI decorator to track method calls and manage state.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Omur Sahin",
    author_email="omur@erciyes.edu.tr",
    packages=find_packages(),
    install_requires=[
        "fastapi",  # Dependenciess
        "uvicorn",
        "python-multipart"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
