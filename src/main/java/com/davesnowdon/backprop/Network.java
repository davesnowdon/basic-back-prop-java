package com.davesnowdon.backprop;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.nd4j.linalg.ops.transforms.Transforms.exp;

/**
 * This is intended to be a java equivalent of Andrew Trask's basic python implementation of a simple neural net.
 * <p>
 * Rather than attempt the same terseness of the python implementation this implementation aims for greater readability.
 */
public class Network {

    INDArray weights0;

    INDArray weights1;

    public Network(int inputWidth, int hiddenWidth, int outputWidth) {
        // initialise weights with random values (mean zero, standard deviation 1)
        weights0 = Nd4j.randn(inputWidth, hiddenWidth);
        weights1 = Nd4j.randn(hiddenWidth, outputWidth);
    }

    /**
     * Train the network
     *
     * @param x - input data
     * @param y - labels
     */
    public void train(INDArray x, INDArray y, int numIterations, double learningRate) {
        for (int i = 0; i < numIterations; ++i) {
            // forward pass
            INDArray layer1 = sigmoid(x.mmul(weights0));
            INDArray layer2 = sigmoid(layer1.mmul(weights1));

            // compute error
            INDArray layer2Error = y.sub(layer2);

            // backward pass
            INDArray delta2 = layer2Error.mul(sigmoidGradient(layer2));
            INDArray layer1Error = delta2.mmul(weights1.transpose());
            INDArray delta1 = layer1Error.mul(sigmoidGradient(layer1));
            weights1 = weights1.add(layer1.transpose().mmul(delta2).mul(learningRate));
            weights0 = weights0.add(x.transpose().mmul(delta1).mul(learningRate));
        }
    }

    public INDArray evaluate(INDArray input) {
        INDArray layer1 = sigmoid(input.mmul(weights0));
        INDArray layer2 = sigmoid(layer1.mmul(weights1));
        return layer2;
    }

    /**
     * Sigmoid activation function
     * Note: ND4J provides an implementation of sigmoid. We implement it here so all the computations are visible.
     */
    private INDArray sigmoid(INDArray input) {
         return Nd4j.ones(input.shape()).div(exp(input.neg()).add(1));
    }

    /**
     * Derivative of a sigmoid x is x * (1 - x)
     */
    private INDArray sigmoidGradient(INDArray input) {
        return input.mul(Nd4j.ones(input.shape()).sub(input));
    }

    public static void main(String args[]) {
        final double[][] inputsArray = {{0, 0, 1}, {0, 1, 1}, {1, 0, 1}, {1, 1, 1}};
        final double[] labelsArray = {0, 1, 1, 0};

        final INDArray inputs = Nd4j.create(inputsArray);
        final INDArray labels = Nd4j.create(labelsArray).transpose();

        Network network = new Network(3, 4, 1);
        network.train(inputs, labels, 60000, 0.1);
    }
}
