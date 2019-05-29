#include "operator.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("psamask_forward", &psamask_forward_cuda, "PSAMASK forward (GPU)");
  m.def("psamask_backward", &psamask_backward_cuda, "PSAMASK backward (GPU)");
}
