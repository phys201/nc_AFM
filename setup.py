from setuptools import setup

setup(name='ncafm',
      version='1.0',
      description='An package for Harvard PHYS201',
      long_description=readme(),
      url='http://github.com/phys201/nc_AFM',
      author='Alyson Spitzig', 'Houchen Li', 'Talha Rehman',
      author_email='alyson.spitzig@gmail.com', 'houchenli@g.harvard.edu', 'talha@g.harvard.edu',
      license='GPLv3',
      packages=['ncafm'],
      install_requires=['numpy', 'pymc3', 'arviz', 'seasborn'])
