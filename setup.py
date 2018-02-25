from setuptools import setup, find_packages


with open('README.md') as f:
    long_description = f.read()


setup(name='physprog',
    version='0.1',
    description=('Physical Programming algorithm to robustly build aggregate'
                 'objective functions for multiobjective optimization'),
    author='Nick Touran',
    author_email='physprog@partofthething.com',
    url='https://github.com/partofthething/physprog',
    packages=find_packages(),
    license='MIT',
    long_description=long_description,
    install_requires=['numpy', 'scipy'],
    keywords='physical programming multiobjective optimization aggregate',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        ],
      test_suite='tests',
      include_package_data=True

)
