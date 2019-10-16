import nl.jochemkuijpers.mlp.ActivationFunction;
import nl.jochemkuijpers.mlp.ErrorFunction;
import nl.jochemkuijpers.mlp.MultiLayeredPerceptron;

import java.util.Arrays;

public class Main {
    private final MultiLayeredPerceptron mlp;

    public Main() {

        /*  input   h1    h2   output
         *   (1)---(2)---(2)---(1)
         *
         *   where h1 and h2 have SIGMOID activation function
         *   output has LINEAR activation function
         *   ERROR function is MEAN_SQUARE_ERROR
         */

        this.mlp = new MultiLayeredPerceptron(
                1, 2, 2, 1,
                ActivationFunction.SIGMOID,
                ActivationFunction.LINEAR,
                ErrorFunction.MEAN_SQUARE_ERROR
        );
    }


    private void train() {
        // input set; every row has one array to assign to the input nodes
        float[][] inputs = {
                {0},
                {1}
        };
        // expected outputs, every row has one array to compare to the output nodes
        float[][] expecteds = inputs;

        float avgError = 0;
        float[] weights = new float[mlp.getVectorSize()];
        float[] avgGradient = new float[mlp.getVectorSize()];
        float[] gradient = new float[mlp.getVectorSize()];

        // learning rate.. we could make it adaptive based on avgError or change in avgError?
        float learnRate = 0.001f;

        // how many iterations over the whole training set do we need to do?
        int epochs = 1000000;

        for (int epoch = 0; epoch < epochs; epoch++) {

            // reset avg error and gradient
            avgError = 0;
            for (int j = 0; j < gradient.length; j++) {
                avgGradient[j] = 0;
            }

            // train for another epoch
            System.out.printf("epoch: %d\n", epoch);
            for (int i = 0; i < inputs.length; i++) {
                float[] input = inputs[i];
                float[] expected = expecteds[i];

                mlp.setInput(input);
                mlp.setExpected(expected);
                mlp.propagateForward();

                float[] output = mlp.getOutput();
                float error = mlp.getError();

                System.out.printf("training; expected: %s, output: %s\n", Arrays.toString(expected), Arrays.toString(output));

                mlp.propagateBackward();

                gradient = mlp.getNablaAsVector(gradient);

                // accumulate avg gradient
                for (int j = 0; j < gradient.length; j++) {
                    avgGradient[j] += gradient[j] / input.length;
                }
                avgError += error / input.length;
            }

            // print out results of this epoch
            weights = mlp.getWeightsAsVector(weights);
            System.out.printf("weights:      %s\n", Arrays.toString(weights));
            System.out.printf("avg gradient: %s\n", Arrays.toString(avgGradient));
            System.out.printf("avg error:    %8.6f\n ", avgError);
            System.out.printf("learn rate:   %8.6f\n\n", learnRate);

            // adjust weights by avg deltas and learn rate
            for (int j = 0; j < weights.length; j++) {
                weights[j] = weights[j] - learnRate * avgGradient[j];
            }

            if (avgError < 0.1) {
                break;
            }

            mlp.setWeightsFromVector(weights);
        }
    }

    public static void main(String[] args) {
        new Main().train();
    }
}
