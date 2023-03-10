// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2023 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "pass_ncnn.h"

#include <limits.h>

namespace pnnx {

namespace ncnn {

class Tensor_slice_copy : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 self
pnnx.Input              input_1     0 1 src
Tensor.slice_copy       op_0        2 1 self src out dims=%dims starts=%starts ends=%ends steps=%steps
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "CopyTo";
    }

    const char* name_str() const
    {
        return "slice_copy";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        std::vector<int> axes = captured_params.at("dims").ai;
        const std::vector<int>& starts = captured_params.at("starts").ai;
        std::vector<int> ends = captured_params.at("ends").ai;
        const std::vector<int>& steps = captured_params.at("steps").ai;
        int axes_rank = axes.size();

        for (int i = 0; i < axes_rank; i++)
        {
            if (steps[i] != 1)
            {
                fprintf(stderr, "slice_copy with step %d is not supported\n", steps[i]);
                return;
            }
        }

        const int batch_index = op->inputs[0]->params["__batch_index"].i;

        {
            int input_rank = op->inputs[0]->shape.size();

            if (batch_index >= 0 && batch_index < input_rank)
                input_rank -= 1;

            if (input_rank > 4)
            {
                fprintf(stderr, "slice_copy %d-rank tensor with %d-rank axes is not possible!\n", input_rank, axes_rank);
                return;
            }
        }

        for (int i = 0; i < axes_rank; i++)
        {
            if (axes[i] == batch_index && (starts[i] != 0 || ends[i] != INT_MAX))
            {
                fprintf(stderr, "slice_copy along batch axis is not supported\n");
                return;
            }

            if (axes[i] < 0)
            {
                int input_rank = op->inputs[0]->shape.size();
                axes[i] = input_rank + axes[i];
            }

            if (axes[i] > batch_index)
                axes[i] -= 1;

            if (ends[i] == INT_MAX)
                ends[i] = -233;
        }

        op->params["9"] = starts;
        // op->params["10"] = ends; // ncnn always resolve ends from src blob
        op->params["11"] = axes;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(Tensor_slice_copy, 20)

} // namespace ncnn

} // namespace pnnx
