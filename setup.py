import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cfis", # Replace with your own username
    version="0.0.1",
    author="Vini Dixit",
    author_email="vini01.dixit@gmail.com",
    description="Hierarchical clustering of text expressions with labelling.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vinidixit/hierarchical-labelled-clustering",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)