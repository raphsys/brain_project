#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>

// =============================
// Module 1: Trace extraction
// =============================
torch::Tensor extract_trace(torch::Tensor activation) {
    activation = activation.view({activation.size(0), -1});  // batch flatten
    torch::Tensor norms = activation.norm(2, 1, true).clamp_min(1e-8);
    return activation / norms;
}

// =============================
// Module 2: Similarity module
// =============================
torch::Tensor cosine_similarity_tensor(torch::Tensor a, torch::Tensor b) {
    a = a.flatten().contiguous().to(torch::kFloat32);
    b = b.flatten().contiguous().to(torch::kFloat32);

    auto dot = torch::dot(a, b).item<float>();
    auto norm_a = a.norm().item<float>();
    auto norm_b = b.norm().item<float>();

    float sim = (norm_a > 0 && norm_b > 0) ? dot / (norm_a * norm_b + 1e-8f) : 0.0f;
    return torch::tensor(sim);
}

// =============================
// Pybind11 interface unique
// =============================
PYBIND11_MODULE(memory_ext, m) {
    m.def("extract_trace", &extract_trace, "Extract normalized trace");
    m.def("cosine_similarity_tensor", &cosine_similarity_tensor, "Compute cosine similarity between two tensors");
}

