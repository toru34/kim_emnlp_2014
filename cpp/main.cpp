#include "dynet/nodes.h"
#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/gpu-ops.h"
#include "dynet/expr.h"
#include "dynet/model.h"
#include "dynet/grad-check.h"

#include <iostream>
#include <fstream>
#include <vector>

using namespace std;
using namespace dynet;

int main(int argc, char** argv)
{
    dynet::initialize(argc, argv);

    // Parameters
    const unsigned ITERATIONS = 30;
    ParameterCollection model;
    SimpleSGDTrainer sgd(model);

    Parameter p_W, p_b, p_V, p_a;
    const unsigned HIDDEN_SIZE = 3;
    p_W = model.add_parameters({HIDDEN_SIZE, 2});
    p_b = model.add_parameters({HIDDEN_SIZE});
    p_V = model.add_parameters({1, HIDDEN_SIZE});
    p_a = model.add_parameters({1});

    // Train the parameters
    for (unsigned iter = 0; iter < ITERATIONS; ++iter) {
        ComputationGraph cg;
        Expression W = parameter(cg, p_W);
        Expression b = parameter(cg, p_b);
        Expression V = parameter(cg, p_V);
        Expression a = parameter(cg, p_a);

        vector<Expression> losses;

        for (unsigned mi = 0; mi < 4; ++mi) {
            bool x1 = mi % 2;       // 0, 1, 0, 1
            bool x2 = (mi / 2) % 2; // 0, 0, 1, 1
            vector<dynet::real> x_values((2));
            x_values[0] = x1 ? 1 : -1; // 1, -1, 1, -1
            x_values[1] = x2 ? 1 : -1; //-1, -1, 1,  1
            float y_value = (x1 != x2) ? 1 : -1;

            Expression x = input(cg, {2}, x_values);
            Expression y = input(cg, y_value);

            Expression h = tanh(affine_transform({b, W, x}));
            Expression y_pred = affine_transform({a, V, h});
            losses.push_back(squared_distance(y_pred, y));
        }

        Expression loss_expr = sum(losses);

        float loss = as_scalar(cg.forward(loss_expr)) / 4;
        cg.backward(loss_expr);
        sgd.update();

        cout << "E = " << loss << endl;
    }

    return 0;
}
