/*
Based on https://github.com/Maverobot/libtorch_examples/blob/master/src/simple_optimization_example.cpp
BSD 3-Clause License

Copyright (c) 2019, Zheng Qu
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <torch/torch.h>
#include <cstdlib>
#include <iostream>

constexpr double kLearningRate = 0.001;
constexpr int kMaxIterations = 100000;

void native_run(double minimal) {
  // Initial x value
  auto x = torch::randn({1, 1}, torch::requires_grad(true));

  for (size_t t = 0; t < kMaxIterations; t++) {
    // Expression/value to be minimized
    auto y = (x - minimal) * (x - minimal);
    if (y.item<double>() < 1e-3) {
      break;
    }
    // Calculate gradient
    y.backward();

    // Step x value without considering gradient
    torch::NoGradGuard no_grad_guard;
    x -= kLearningRate * x.grad();

    // Reset the gradient of variable x
    x.mutable_grad().reset();
  }

  std::cout << "[native] Actual minimal x value: " << minimal << ", calculated optimal x value: " << x.item<double>()
            << std::endl;
}

void optimizer_run(double minimal) {
  // Initial x value
  std::vector<torch::Tensor> x;
  x.push_back(torch::randn({1, 1}, torch::requires_grad(true)));
  auto opt = torch::optim::SGD(x, torch::optim::SGDOptions(kLearningRate));

  for (size_t t = 0; t < kMaxIterations; t++) {
    // Expression/value to be minimized
    auto y = (x[0] - minimal) * (x[0] - minimal);
    if (y.item<double>() < 1e-3) {
      break;
    }
    // Calculate gradient
    y.backward();

    // Step x value without considering gradient
    opt.step();
    // Reset the gradient of variable x
    opt.zero_grad();
  }

  std::cout << "[optimizer] Actual minimal x value: " << minimal
            << ", calculated optimal x value: " << x[0].item<double>() << std::endl;
}

// optimize y = (x - 10)^2
int main(int argc, char* argv[]) {
  native_run(0.01);
  optimizer_run(0.01);
  return 0;
}
