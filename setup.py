from setuptools import setup

# reading long description from file
with open('DESCRIPTION.txt') as file:
    long_description = file.read()

REQUIREMENTS = [
    "transformers>=4.35",
    "torch>=1.12",
    "numpy",
    "bert_score",
    "spacy",
    "nltk",
    "openai",
]

# some more details
CLASSIFIERS = [
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
]

VERSION = {}
with open("crosscheckgpt/version.py", "r") as version_file:
    exec(version_file.read(), VERSION)

# calling the setup function
setup(
    name='crosscheckgpt',
    version=VERSION["__version__"],
    description='CrossCheckGPT: Universal Hallucination Ranking for Multimodal Foundation Models'
    long_description=long_description,
    url='https://github.com/potsawee/selfcheckgpt',
    author='Brian Sun and Potsawee Manakul',
    author_email='m.potsawee@gmail.com',
    license='MIT',
    packages=['crosscheckgpt'],
    classifiers=CLASSIFIERS,
    install_requires=REQUIREMENTS,
    keywords='crosscheckgpt',
    include_package_data=True,
)
