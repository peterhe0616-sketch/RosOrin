from glob import glob
from setuptools import find_packages, setup


package_name = "rosorin_autonomy"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Peter",
    maintainer_email="peter@localhost",
    description="ROSOrin autonomy bridge and Nav2 configuration",
    license="MIT",
    entry_points={
        "console_scripts": [
            "autonomy_bridge = rosorin_autonomy.autonomy_bridge:main",
        ],
    },
)
