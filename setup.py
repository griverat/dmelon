from setuptools import setup

# CHANGE PKG NAME HERE
pkg_name = "dmelon"

setup(
    use_scm_version={
        "write_to": f"{pkg_name}/_version.py",
        "write_to_template": '__version__ = "{version}"',
        "tag_regex": r"^(?P<prefix>v)?(?P<version>[^\+]+)(?P<suffix>.*)?$",
    },
    scripts=["bin/update_dac.py"],
)
