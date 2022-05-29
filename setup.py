import setuptools

VERSION_STR = "0.8"

with open("README.rst", "r") as fh:
    long_description = fh.read()

# Get the required packages
with open('requirements.txt', encoding='utf-8') as f:
    install_requires = f.read().splitlines()

# Deep learning packages are optional to install
extras = ["deeplearning"]
extras_require = dict()
for e in extras:
    req_file = "requirements-{0}.txt".format(e)
    with open(req_file) as f:
        extras_require[e] = [line.strip() for line in f]

setuptools.setup(
    name="dice_ml",
    version=VERSION_STR,
    license="MIT",
    author="Ramaravind Mothilal, Amit Sharma, Chenhao Tan",
    author_email="raam.arvind93@gmail.com",
    description="Generate Diverse Counterfactual Explanations for any machine learning model.",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/interpretml/DiCE",
    download_url="https://github.com/interpretml/DiCE/archive/v"+VERSION_STR+".tar.gz",
    python_requires='>=3.6',
    packages=setuptools.find_packages(),
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords='machine-learning explanation interpretability counterfactual',
    install_requires=install_requires,
    extras_require=extras_require,
    include_package_data=True,
    package_data={
        # If any package contains *.h5 files, include them:
        '': ['*.h5',
             'counterfactual_explanations_v1.0.json',
             'counterfactual_explanations_v2.0.json']
    }
)
