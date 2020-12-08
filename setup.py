import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dbsampler", 
    version="0.0.1",
    author="Antonio Leitao",
    author_email="aml.leitao@novaims.unl.pt",
    description="Package to sample points in the decision boundary.",
    url="https://github.com/Antonio-Leitao/dbsampler",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
