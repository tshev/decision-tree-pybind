#!/usr/bin/env python

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools
import os
import subprocess
import platform

CPP_SRC = "src"


class get_pybind_include(object):
    """Helper class to determine the pybind11 include path

    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self, user=False):
        try:
            import pybind11
        except ImportError:
            if subprocess.call([sys.executable, "-m", "pip", "install", "pybind11"]):
                raise RuntimeError("pybind11 install failed.")

        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)


decision_tree_src_files = [str(x) for x in os.listdir(CPP_SRC)]
decision_tree_src_cc = [str(os.path.join(CPP_SRC, x)) for x in decision_tree_src_files if x.endswith(".cc")]

ext_modules = [
    Extension(
        str("decision_tree_pybind"),
        [str("python/decision_tree_module/pybind/decision_tree_pybind.cc"),] + decision_tree_src_cc,
        include_dirs=[get_pybind_include(), get_pybind_include(user=True), CPP_SRC,],
        language="c++",
        extra_compile_args=["-O3 -pthread -march=native"],
    ),
]


# As of Python 3.6, CCompiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler, flags):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".cpp") as f:
        f.write("int main (int argc, char **argv) { return 0; }")
        try:
            compiler.compile([f.name], extra_postargs=flags)
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def cpp_flag(compiler):
    standards = ["-std=c++17", "-std=c++14"]
    for standard in standards:
        if has_flag(compiler, [standard]):
            return standard
    raise RuntimeError("Unsupported compiler")


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        "msvc": ["/EHsc"],
        "unix": [],
    }

    def build_extensions(self):
        if sys.platform == "darwin":
            mac_osx_version = float(".".join(platform.mac_ver()[0].split(".")[:2]))
            os.environ["MACOSX_DEPLOYMENT_TARGET"] = str(mac_osx_version)
            all_flags = ["-stdlib=libc++", "-mmacosx-version-min=10.7"]
            if has_flag(self.compiler, [all_flags[0]]):
                self.c_opts["unix"] += [all_flags[0]]
            elif has_flag(self.compiler, all_flags):
                self.c_opts["unix"] += all_flags
            else:
                raise RuntimeError(
                    "libc++ is needed! Failed to compile with {} and {}.".
                    format(" ".join(all_flags), all_flags[0])
                )
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        extra_link_args = []

        if ct == "unix":
            opts.append("-DVERSION_INFO=\"%s\"" % self.distribution.get_version())
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, ["-fvisibility=hidden"]):
                opts.append("-fvisibility=hidden")
        for ext in self.extensions:
            ext.extra_compile_args = opts
            ext.extra_link_args = extra_link_args
        build_ext.build_extensions(self)


setup(
    name="decision_tree",
    version="0.1",
    author="Taras Shevchenko",
    description="decision_tree Python bindings",
    long_description="README",
    ext_modules=ext_modules,
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
    ],
    install_requires=["pybind11>=2.2", "setuptools >= 0.7.0", "numpy"],
    cmdclass={"build_ext": BuildExt},
    packages=[
        str("decision_tree"),
    ],
    package_dir={str(""): str(os.path.join("python", "decision_tree_module"))},
    zip_safe=False,
)
