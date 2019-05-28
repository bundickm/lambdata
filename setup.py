""" 
lambdata - a collection of Data Science helper functions
"""

import setuptools

REQUIRED = [
    'numpy',
    'pandas'
]

with open('README.md', 'r') as fh:
    LONG_DESCRIPTION = fh.read()

setuptools.setup(
    name='lambdata-bundickm',
    version='0.0.1',
    author='bundickm',
    description='a collection of Data Science helper functions',
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    url='https://github.com/bundickm/lambdata',
    packages=setuptools.find_packages(),
    python_requires='>=3.5',
    intall_requires=REQUIRED,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ]
)