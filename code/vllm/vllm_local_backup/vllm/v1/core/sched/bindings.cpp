#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "chunked_hash_tree.hpp"

namespace py = pybind11;

PYBIND11_MODULE(chunked_hash_tree, m) {
    py::class_<ChunkedHashTree>(m, "ChunkedHashTree")
        .def(py::init<uint32_t>(), py::arg("chunk_size"))
        .def("insert", &ChunkedHashTree::insert)
        .def("find_best_request", &ChunkedHashTree::find_best_request)
        .def("activate_request", &ChunkedHashTree::activate_request)
        .def("finish_request", &ChunkedHashTree::finish_request)
        .def("remove", &ChunkedHashTree::remove);
}