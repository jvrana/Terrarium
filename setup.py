from distutils.core import setup

setup(name='autoplanner',
      packages=['autoplanner'],
      install_requires=[
          'pydent',
          'networkx',
          'tqdm'
      ],
      tests_require=['pytest']
      )
