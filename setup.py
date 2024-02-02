import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="e2ml",
    version="0.0.0",
    author="Aleyna Albayrak",
    author_email="uk076901@student.uni-kassel.de",
    description="This package contains methods implemented during the course"
                "'Experimentation and Evaluation in Machine Learning' (E2ML)"
                "of the 'Intelligent Embedded Systems' (IES) department at "
                "the University of Kassel in Germany.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.9",
    install_requires=[
        'numpy~=1.20.1',
        'scipy~=1.6.1',
        'scikit-learn~=1.0.1',
        'matplotlib~=3.3.4',
        'iteration-utilities~=0.11.0',
        'jupyter~=1.0.0',
        'pandas~=1.3.3'
    ],
)
