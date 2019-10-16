package nl.jochemkuijpers.mlp;

import java.util.Random;

public class MultiLayeredPerceptron {
    /**
     * size of the serialized weight/nabla vectors
     */
    private final int vectorSize;

    private final int layers;
    private final int[] layerSizes;

    private final ActivationFunction sigmaH;
    private final ActivationFunction sigmaO;
    private final ErrorFunction errorFunction;

    /**
     * [layer][destination_node][source_node_in_prev_layer]
     * every destination node has an additional source which acts as the bias
     * nabla stores the gradient after back propagation and is serialized identically to the weights.
     */
    private final float[][][] weights;
    private final float[][][] nabla;

    /**
     * [layer][node]
     */
    private final float[][] nodeIn;
    private final float[][] nodeOut;

    /**
     * expected output for back propagation
     */
    private final float[] expected;

    /**
     * @param inputSize                number of input neurons
     * @param width                    number of hidden neurons per layer
     * @param depth                    number of hidden layers
     * @param outputSize               number of output neurons
     * @param hiddenActivationFunction activation function for the hidden layers
     * @param outputActivationFunction activation function for the output layer
     * @param errorFunction            function to minimize
     */
    public MultiLayeredPerceptron(int inputSize,
                                  int width,
                                  int depth,
                                  int outputSize,
                                  ActivationFunction hiddenActivationFunction,
                                  ActivationFunction outputActivationFunction,
                                  ErrorFunction errorFunction) {
        this.vectorSize =
                width * (inputSize + 1) +                       // input layer
                        Math.max(0, depth - 1) * width * (width + 1) +  // hidden layers
                        outputSize * (width + 1);                       // output layer

        this.layers = depth + 2;
        this.layerSizes = new int[layers];
        layerSizes[0] = inputSize;
        layerSizes[layers - 1] = outputSize;
        for (int i = 1; i < layers - 1; i++) {
            layerSizes[i] = width;
        }

        this.sigmaH = hiddenActivationFunction;
        this.sigmaO = outputActivationFunction;
        this.errorFunction = errorFunction;

        this.weights = new float[layers][][];
        this.nabla = new float[layers][][];
        this.nodeIn = new float[layers][];
        this.nodeOut = new float[layers][];
        this.expected = new float[outputSize];

        initialize(0);
    }

    public void initialize(long seed) {
        Random random = new Random(seed);

        for (int layer = 0; layer < layers; layer++) {
            nodeIn[layer] = new float[layerSizes[layer]];
            nodeOut[layer] = new float[layerSizes[layer]];

            if (layer < layers - 1) {
                weights[layer] = new float[layerSizes[layer + 1]][];
                nabla[layer] = new float[layerSizes[layer + 1]][];

                for (int dst = 0; dst < layerSizes[layer + 1]; dst++) {
                    weights[layer][dst] = new float[layerSizes[layer] + 1];
                    nabla[layer][dst] = new float[layerSizes[layer] + 1];

                    for (int src = 0; src < layerSizes[layer] + 1; src++) {
                        weights[layer][dst][src] = random.nextFloat() * 2f - 1f;
                        nabla[layer][dst][src] = 0;
                    }
                }
            }
        }
    }

    public int getVectorSize() {
        return vectorSize;
    }

    public void setInput(float[] input) {
        assert (input.length == layerSizes[0]);
        System.arraycopy(input, 0, nodeOut[0], 0, layerSizes[0]);
    }

    public void setExpected(float[] expected) {
        assert (expected.length == layerSizes[layers - 1]);
        System.arraycopy(expected, 0, this.expected, 0, layerSizes[layers - 1]);
    }

    public float[] getOutput() {
        float[] output = new float[layerSizes[layers - 1]];
        System.arraycopy(nodeOut[layers - 1], 0, output, 0, layerSizes[layers - 1]);
        return output;
    }

    public float getError() {
        return errorFunction.apply(nodeOut[layers - 1], expected);
    }

    public float[] getWeightsAsVector(float[] reuseVector) {
        float[] vector = reuseVector != null ? reuseVector : new float[vectorSize];
        assert (vector.length == vectorSize);
        return serializeNestedArrayToVector(weights, vector);
    }

    public void setWeightsFromVector(float[] vector) {
        assert (vector.length == vectorSize);
        deserializeNestedArrayFromVector(weights, vector);
    }

    public float[] getNablaAsVector(float[] reuseVector) {
        float[] vector = reuseVector != null ? reuseVector : new float[vectorSize];
        assert (vector.length == vectorSize);
        return serializeNestedArrayToVector(nabla, vector);
    }

    public void propagateForward() {
        for (int layer = 0; layer < layers; layer++) {
            ActivationFunction sigma = (layer == layers - 1) ? sigmaO : sigmaH;

            if (layer > 0) {
                for (int n = 0; n < layerSizes[layer]; n++) {
                    nodeOut[layer][n] = sigma.apply(nodeIn[layer][n]);
                }
            }

            if (layer < layers - 1) {
                for (int dst = 0; dst < layerSizes[layer + 1]; dst++) {
                    // bias
                    nodeIn[layer + 1][dst] = weights[layer][dst][layerSizes[layer]];

                    // weighted edges
                    for (int src = 0; src < layerSizes[layer]; src++) {
                        nodeIn[layer + 1][dst] += weights[layer][dst][src] * nodeOut[layer][src];
                    }
                }
            }
        }
    }

    public void propagateBackward() {
        float error = getError();
        float[] nodeOutError = errorFunction.derivative(nodeOut[layers - 1], expected);
        for (int i = 0; i < nodeOutError.length; i++) {
            nodeOutError[i] *= error;
        }

        float[] nodeInError = sigmaO.derivativeAll(nodeIn[layers - 1]);
        for (int i = 0; i < nodeInError.length; i++) {
            nodeInError[i] *= nodeOutError[i];
        }

        for (int layer = layers - 2; layer >= 0; layer--) // reversed
        {

            for (int dst = 0; dst < layerSizes[layer + 1]; dst++) {
                // bias
                nabla[layer][dst][layerSizes[layer]] = nodeInError[dst];

                // weighted edges
                for (int src = 0; src < layerSizes[layer]; src++) {
                    nabla[layer][dst][src] = nodeOut[layer][src] * nodeInError[dst];
                }
            }

            if (layer > 0) {
                // compute node errors for the next layer
                nodeOutError = new float[layerSizes[layer]];
                for (int src = 0; src < layerSizes[layer]; src++) {
                    for (int dst = 0; dst < layerSizes[layer + 1]; dst++) {
                        nodeOutError[src] += weights[layer][dst][src] * nodeInError[dst];
                    }
                }
                nodeInError = sigmaH.derivativeAll(nodeIn[layer]);
                for (int i = 0; i < nodeInError.length; i++) {
                    nodeInError[i] *= nodeOutError[i];
                }
            }
        }
    }

    private static float[] serializeNestedArrayToVector(float[][][] nested, float[] vector) {
        int pos = 0;
        for (float[][] x : nested) {
            if (x == null) continue;
            for (float[] y : x) {
                System.arraycopy(y, 0, vector, pos, y.length);
                pos += y.length;
            }
        }
        return vector;
    }

    private static void deserializeNestedArrayFromVector(float[][][] nested, float[] vector) {
        int pos = 0;
        for (float[][] x : nested) {
            if (x == null) continue;
            for (float[] y : x) {
                System.arraycopy(vector, pos, y, 0, y.length);
                pos += y.length;
            }
        }
    }
}
