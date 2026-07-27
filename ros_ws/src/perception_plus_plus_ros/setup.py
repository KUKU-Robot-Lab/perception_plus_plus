from glob import glob
from setuptools import find_packages, setup

package_name = "perception_plus_plus_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    tests_require=["pytest"],
    zip_safe=True,
    maintainer="Maintainer",
    maintainer_email="maintainer@example.com",
    description="ROS 2 adapter for FoundationPose++ cup tracking",
    license="Apache-2.0",
    entry_points={"console_scripts": [
        "cup_tracking_node = perception_plus_plus_ros.node:main",
        "anchored_cup_tracking_node = perception_plus_plus_ros.anchored_node:main",
    ]},
)
