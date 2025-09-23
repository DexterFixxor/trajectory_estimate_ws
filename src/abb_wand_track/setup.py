from setuptools import find_packages, setup

package_name = 'abb_wand_track'

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
    maintainer_email='miliclazar@uns.ac.rs',
    description='Track active wand and send command to ABB to move after certain height is achieved.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'abb_wand = abb_wand_track.abb_wand:main'
        ],
    },
)
