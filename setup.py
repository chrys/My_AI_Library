from setuptools import setup, find_packages

setup(
    name='my_library',
    version='0.1.0',
    description='My Python library',
    author='Chrys Zampas',
    author_email='zampas@gmail.com',
    packages=find_packages(where='src'),  # Find packages within the src directory
    package_dir={'': 'src'},               # Tell setuptools packages are under src
    install_requires=[
        # List dependencies here (if any)
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
