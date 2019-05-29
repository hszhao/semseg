#include "operator.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("psamask_forward", &psamask_forward_cpu, "PSAMASK forward (CPU)");
  m.def("psamask_backward", &psamask_backward_cpu, "PSAMASK backward (CPU)");
}
