from setuptools import setup, find_packages

README = 'This is the first degree final project'

requires = ['cython',
            'tornado==5.1.1',
            'tornado-cors',
            'nltk',
            'pandas',
            'numpy',
            'xgboost',
            'wordcloud',
            'sklearn',
            'keras',
            'shap',
            'gensim',
            'matplotlib']
tests_require = [
    'pytest']

setup(name='CyberBullying',
      version='0.0.1',
      description=README,
      long_description=README,
      package_dir={'': 'source'},
      classifiers=[],
      author='Ron, Sivan and Shaked',
      packages=find_packages('source'),
      include_package_data=True,
      zip_safe=False,
      extras_require={
          'testing': tests_require,
      },
      install_requires=requires,
      entry_points={},
      )
