from setuptools import setup, find_namespace_packages

setup(
    name='vcm',
    version='0.1',
    packages=find_namespace_packages(),
    include_package_data=True,
    package_data={
        'vcm': ['video_call_mos_weights.pt'],
    },    
)