from setuptools import find_packages, setup

package_name = 'vicon_data_collector'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Lazar Milic',
    maintainer_email='lazarmilic2@gmail.com',
    description='Data colecting from vicon',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vicon_data_collector = vicon_data_collector.vicon_data_collector:main'
        ],
    },
)
