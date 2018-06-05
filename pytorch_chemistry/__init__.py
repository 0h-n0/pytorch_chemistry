__version__ = '0.0.1'

version = get_version('pytorch_chemistry')

setuptools.setup(
    name="pytorch-chemistry",
    version=version,
    python_requires='>3.6',    
    author="Koji Ono",
    author_email="kbu94982@gmail.com",
    description="Drug Discuvory Library based on Deep Learning",
    long_description=(p.parent / 'README.md').open(encoding='utf-8').read(),
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy'
    ],
    setup_requires=['numpy', 'pytest-runner'],
    tests_require=['pytest-cov', 'pytest-html', 'pytest'],
    classifiers=[
        'Programming Language :: Python :: 3.6',
    ],
    data_files=[('html', ['html/train_template.html'])]
)
