from setuptools import setup

setup(
   name='foofoo',
   version='1.0',
   description='A useful module',
   author='Man Foo',
   author_email='foomail@foo.com',
   packages=['foo'],  #same as name
   install_requires=['wheel', 'bar'], #external packages as dependencies
   scripts=[
            'scripts/cool',
            'scripts/skype',
           ]
)
