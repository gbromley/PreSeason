from setuptools import setup, find_packages


setup(
    name='preseason',
    version='0.1.0',
    packages=find_packages(where="preseason"),
    package_dir={"": "preseason"},
    install_requires=[
        # List your project's dependencies here
        # e.g., 'requests >= 2.25.1',
    ],
    # Additional metadata about your package
    author='Gabriel Bromley',
    author_email='bromlgab@gmail.com',
    description='Tools for determining seasonality metrics for precipitation',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',  # This is important for a markdown README
    url='https://github.com/gbromley/PreSeason',
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
)