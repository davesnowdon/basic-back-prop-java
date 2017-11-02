import com.davesnowdon.backprop.Network;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;

public class Train {
    @Test
    public void trainNetorkAndTestResults() {
        final double[][] inputsArray = {{0, 0, 1}, {0, 1, 1}, {1, 0, 1}, {1, 1, 1}};
        final double[] labelsArray = {0, 1, 1, 0};

        final INDArray inputs = Nd4j.create(inputsArray);
        final INDArray labels = Nd4j.create(labelsArray).transpose();

        Network network = new Network(3, 4, 1);
        network.train(inputs, labels, 10_000, 0.1);

        final double[] testInputArray = {0,1 ,1};
        INDArray output = network.evaluate(Nd4j.create(testInputArray));
        int[] shape = output.shape();
        assertEquals(1, shape[0]);
        assertEquals(1, shape[1]);
        double value = output.getDouble(0);
        assertEquals(1.0, value, 0.1);
    }
}
