from glob import glob

from setuptools import find_packages, setup

package_name = 'phy'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='su',
    maintainer_email='lsu031111@hanyang.ac.kr',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'hold_node = phy.hold_node:main',
            'ee_xyz_trajectory_node = phy.ee_xyz_trajectory_node:main',
            'joint_sweep_node = phy.joint_sweep_node:main',
            'plan_node = phy.plan_node:main',
            'plan_compute_node = phy.plan_compute_node:main',
            'send_target = phy.send_target:main',
            'gripper_node = phy.gripper_node:main',
            'task_fsm_node = phy.task_fsm_node:main',
            'contact_detector_node = phy.contact_detector_node:main',
        ],
    },
)
