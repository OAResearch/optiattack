from setuptools import setup, find_packages

setup(
    name="optiattack_client",
    version="0.1.6",
    description="A FastAPI decorator to track method calls and manage state.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Omur Sahin",
    author_email="omur@erciyes.edu.tr",
    url="https://github.com/OAResearch/optiattack",
    package_dir={"": "core"},
    packages=find_packages(where="core"),
    py_modules=["optiattack_client", "constants"],
    install_requires=[
        "fastapi",
        "uvicorn",
        "python-multipart"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
