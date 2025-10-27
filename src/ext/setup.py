from setuptools import setup, Extension
from Cython.Build import cythonize
import sys
import sysconfig

ext = Extension(
    name="shanten_dp_cy",
    sources=["shanten_dp_cy.pyx"],
    extra_compile_args=["-O3", "-ffast-math"],
)

setup(
    name="shanten_dp_cy",
    ext_modules=cythonize(
        ext,
        language_level="3",
        annotate=True,  # 生成 .html 性能标注，便于优化（可关）
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
        },
    ),
)
