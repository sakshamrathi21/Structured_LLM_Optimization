from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "chunked_hash_tree",
        [
            "chunked_hash_tree.cpp",
            "chunked_hash_tree_rl.cpp",
            "bindings.cpp"
        ],
        include_dirs=[
            pybind11.get_include(),
            "/usr/include"
        ],
        libraries=["xxhash"],
        language="c++",
        extra_compile_args=["-O3", "-std=c++17"]
    ),
]

setup(
    name="chunked_hash_tree",
    version="0.1.0",
    description="C++ Extension for Chunked Hash Tree",
    ext_modules=ext_modules,
)
