/*
 * Cooper Rachow
 * 10392674
 * 10/21/24
 * Assignment: Part 2: Neural Networks
 * Description: The program below implements a 3 layer (add more hidden if desired) neural network architecture that uses Stochastic Gradient Descent to 
 * train itself on the MNIST handritten digit dataset. It offers an interactive interface for the user to make multiple decisions on 
 * what they would like to experience through the network, such as tarining the network themselves, loading in a saved network (preloaded weihts, biases, etc),
 * saving a network after they have trained it, testing the network on MNIST test data, and veiwing all the results as "images" or filtered down to misclassified. 
 * The genral flow of the network itself is to load and organize the MNIST training and testing data. Seperating inputs and outputs (one-hot vectors). Once we have the 
 * I/O, we populate the bias and weight structures with random values. This is then passed through the SGD function, which is the base algorithm for the rest of the network.
 * Each epoch is began and ended within here, creating minibatches, shuffling their values, and passing each through the feedword/backprop function. Within there, our weight and bias 
 * gradients are returned to be used in updating the random values of the current minibatch. After each epoch is completed, the user is shown stats relating to 
 * each digit and a final percentage out of 60000 for the user to visualize the progress of the network during the training process. 
 */

// Shuffle algo: https://www.geeksforgeeks.org/shuffle-a-given-array-using-fisher-yates-shuffle-algorithm/
// Gaussian random: https://www.geeksforgeeks.org/random-nextgaussian-method-in-java-with-examples/

// Import necessary packages
import java.util.List;
import java.util.ArrayList;
import java.util.Random;
import java.util.Scanner; // user input
import java.io.IOException;
// File related libraries
// Wrappers for serialization
import java.io.BufferedReader;
import java.io.ObjectOutputStream;
import java.io.ObjectInputStream;
// File reading
import java.io.FileReader;
import java.io.FileOutputStream;
import java.io.FileInputStream;

public class NNP2 {
    public static void main(String[] args) {
        int[] sizes = {784, 100, 10}; // Network architecture: 784 inputs, 30 hidden neurons, 10 outputs
        // Initialize the network with proper sizes, setting up layers, filling in bias/weights with random values ready for SGD to be called
        NeuralNetwork neuralNetwork = new NeuralNetwork(sizes); 
        Scanner scanner = new Scanner(System.in);
        boolean exit = false; // running status
        boolean isNetworkInitialized = false; // used for allowing extra options

        // Load training and testing data
        CSVData trainingData = MNISTOrganizer("mnist_train.csv");
        CSVData testData = MNISTOrganizer("mnist_test.csv");

        // Run program until user ends it
        while (!exit) {
            // On start up, only show options that must be selected prior to following options
            System.out.println("Select an option:");
            System.out.println("[1] Train the network");
            System.out.println("[2] Load a pre-trained network");
            // Once either 1 / 2 are selected, then display new options to user
            if (isNetworkInitialized) {
                System.out.println("[3] Display network accuracy on TRAINING data");
                System.out.println("[4] Display network accuracy on TESTING data");
                System.out.println("[5] Run network on TESTING data showing images and labels");
                System.out.println("[6] Display the misclassified TESTING images");
                System.out.println("[7] Save the network state to file");
            }
            System.out.println("[0] Exit");
            // grabs whatever users integer choice was
            int choice = scanner.nextInt();
            // Based on input, decide next step with switch cases
            switch (choice) {
                // Train network with MNSIT loaded preloaded data
                case 1:
                    neuralNetwork.SGD(trainingData.inputs, trainingData.outputs, 30, 10, 3.0);
                    isNetworkInitialized = true; // Set to true so other options are displayed
                    break;
                // If user has a saved state of the network
                case 2:
                    // ask for file name and grab that 
                    System.out.print("Enter the filename to load the network from: ");
                    String loadFilename = scanner.next();
                    // Call loadnetwork function with file name
                    neuralNetwork.loadNetwork(loadFilename);
                    isNetworkInitialized = true; // Set to true so other options are displayed
                    break;
                // Run the network on the training data and see resulst throgh evaluate function
                case 3:
                    if (isNetworkInitialized) {
                        System.out.println("Evaluating network on training data...");
                        neuralNetwork.evaluate(trainingData.inputs, trainingData.outputs);
                    } else {
                        System.out.println("Please train or load a network first.");
                    }
                    break;
                // Run the network on the testing data and see resulst throgh evaluate function
                case 4:
                    if (isNetworkInitialized) {
                        System.out.println("Evaluating network on testing data...");
                        neuralNetwork.evaluate(testData.inputs, testData.outputs);
                    } else {
                        System.out.println("Please train or load a network first.");
                    }
                    break;
                // Pass testing data inputs and outputs to display function for images, accuracy, etc
                case 5:
                    if (isNetworkInitialized) {
                        neuralNetwork.displayResultImages(testData.inputs, testData.outputs, true);
                    } else {
                        System.out.println("Please train or load a network first.");
                    }
                    break;
                // Pass INCORRECT testing data inputs and outputs to display function for images, accuracy, etc
                case 6:
                    if (isNetworkInitialized) {
                        // Display misclassified testing images
                        neuralNetwork.displayResultImages(testData.inputs, testData.outputs, false);
                    } else {
                        System.out.println("Please train or load a network first.");
                    }
                    break;
                // Give the user the option to save the network state
                case 7:
                    if (isNetworkInitialized) {
                        // Get user provided file name
                        System.out.print("Enter the filename to save the network to: ");
                        String saveFilename = scanner.next();
                        // Pass file name trough saveNetwork function
                        neuralNetwork.saveNetwork(saveFilename);
                    } else {
                        System.out.println("Please train or load a network first.");
                    }
                    break;
                // User can exit the program
                case 0:
                    exit = true;
                    break;
                // If any other type of input, display message
                default:
                    System.out.println("Invalid option selected.");
            }
        }
        scanner.close();
    }
    // Load in CSV files and orgaize inputs and outputs retuirning the CSVData object 
    public static CSVData MNISTOrganizer(String filename) {
        try {
            // open file and wrap with BufferedReader (purpose is to offer readline functionality)
            BufferedReader br = new BufferedReader(new FileReader(filename));
            String row; // will be set to each row of file in loop below
            // Skip the header line
            row = br.readLine();

            /*
            * Starting with lists so we dont have to pre-allocate sizes, we can just expand 
            * then convert to arrays after filled successfully
            */
            List<double[]> inputsList = new ArrayList<>();
            List<double[]> outputsList = new ArrayList<>();

            // iterate through each row in the CSV until end of content
            while ((row = br.readLine()) != null) {
                String[] values = row.split(","); // split up all values in an array
                double[] input = new double[784]; // initialize input array (784 for MNIST)
                double[] output = new double[10]; // holds one-hot encodings, starts at all zeros

                // First value is the label
                int label = Integer.parseInt(values[0]);
                output[label] = 1.0; // Index = label place 1.0 there

                 // Rest are pixel values, that are scaled to values between 0-1 for normalization.
                for (int i = 1; i < values.length; i++) {
                    input[i - 1] = Double.parseDouble(values[i]) / 255.0;
                }
                // Append new arrays (rows) dynamically and onto the next
                inputsList.add(input);
                outputsList.add(output);
            }
            br.close();

            // Convert lists to arrays and return the Object of the arrays
            double[][] inputs = inputsList.toArray(new double[inputsList.size()][]);
            double[][] outputs = outputsList.toArray(new double[outputsList.size()][]);

            return new CSVData(inputs, outputs);

        } catch (IOException e) {
            // If an error occurs just return null and display error message
            System.out.println("Error loading data from file: " + filename);
            e.printStackTrace();
            return null;
        }
    }
}

// Key class containing the entire network structure once given the required parameters.
class NeuralNetwork {
    private int numLayers;
    private int[] sizes;
    private double[][] biases;
    private double[][][] weights;
    private Random random;

    // Constructor with random Gaussian initialization
    public NeuralNetwork(int[] sizes) {
        this.numLayers = sizes.length; // Typically 3 for this network
        this.sizes = sizes;  // Typically serves as [784, 30, 10]
        this.biases = new double[numLayers - 1][]; // First layer has no biases so 2D array for hidden and output [biases for hidden, biases for output]
        /*
         * [
         *      Input to hidden
         *      [
         *          hidden neuron 0
         *          [w_0_0, .... w_0_n-1],
         *          hidden neuron 1
         *          [w_0_0, .... w_0_n-1],
         *          ...
         * 
         *      ],
         *      hidden to output
         *      [
         * 
         *      ]
         * ]
         */
        this.weights = new double[numLayers - 1][][]; // 3D array [weights between 2 layers, weights for each neuron in current layer, weights for each neuron in prev layer]
        this.random = new Random();

        // Initialize biases and weights with random Gaussian values
        // Loops through each layer of network after input layer generating random values with a mean of 0 and standard deviation of 1
        for (int i = 1; i < numLayers; i++) {
            // initialize bias array in current layer based on size
            biases[i - 1] = new double[sizes[i]]; // [[30], [10]]
            for (int j = 0; j < sizes[i]; j++) {
                biases[i - 1][j] = random.nextGaussian();
            }
            // initialize weight array in current layer based on size
            // allocate space for each individual array
            weights[i - 1] = new double[sizes[i]][sizes[i - 1]]; // weights[[30[784], [10][30]]
            for (int j = 0; j < sizes[i]; j++) {
                for (int k = 0; k < sizes[i - 1]; k++) {
                    weights[i - 1][j][k] = random.nextGaussian();
                }
            }
        }
    }

    /*  Stochastic Gradient Descent
     *  General flow: Shuffle data, run through feedforward, run through backpropagation, 
     *  update the minibatch with new gradients, print each epoch and evaluate results (print).
     */
    public void SGD(double[][] inputs, double[][] outputs, int epochs, int miniBatchSize, double eta) {
        // Number of input vectors in input array
        int lenInputs = inputs.length;
        // Run epochs (outer loop)
        for (int epoch = 0; epoch < epochs; epoch++) {
            // Initialize indices array with size of lenInputs and populate with ordered numbers to be shuffled
            int[] indices = new int[lenInputs];
            for (int i = 0; i < lenInputs; i++) { // 0, 1, 2, ..., 60000
                indices[i] = i;
            }
            // Shuffle the indices array
            for (int i = indices.length - 1; i > 0; i--) {
                int index = random.nextInt(i + 1); // get random index
                int a = indices[index];
                indices[index] = indices[i]; // swap
                indices[i] = a;
            }

            // Loop over the inputs in steps of miniBatchSize
            for (int mb = 0; mb < lenInputs; mb += miniBatchSize) {
                // Grab end index for the current minibatch. Calculate the range for minibatch size
                int end = Math.min(mb + miniBatchSize, lenInputs); // Basically checks in case it exceeds amount of inputs
                // Initialize I/O pairs based on range defined minibatch range size defined above
                double[][] miniBatchInputs = new double[end - mb][];
                double[][] miniBatchOutputs = new double[end - mb][];
                // Extract I/O pairs: Output - one hot vector, Input - vector of all pixel values
                // v = start of MB, end is end of mb, [v - mb] starts from index zero onward
                for (int v = mb; v < end; v++) { // indices helps choose random I/O pair
                    miniBatchInputs[v - mb] = inputs[indices[v]]; // 10 input vectors
                    miniBatchOutputs[v - mb] = outputs[indices[v]]; // 10 one hot output vectors
                }
                // Initialize my gradient holders for weights and biases (over entire minibatch)
                double[][] gradient_b = new double[biases.length][]; // [[],[]] 
                double[][][] gradient_w = new double[weights.length][][]; // [[][], [][]]
                // Purpose: Allocate appropriate space for each gradient array matching sizes of bias/weight arrays holding random values
                for (int h = 0; h < biases.length; h++) { // biases.length == 2 in this case (number of layers except first)
                    gradient_b[h] = new double[biases[h].length]; // 1st: gradient_b = [[30]] 2nd: gradient_b = [[30], [10]]
                    gradient_w[h] = new double[weights[h].length][weights[h][0].length]; // 1st: gradient_w = [[30][784]] 2nd: gradient_w = [[30][784], [10][30]]
                }

                // Process each input sample in the minibatch (about 10 iterations for this case)
                // Keep adding up all gradients for each image
                for (int h = 0; h < miniBatchInputs.length; h++) {
                    // pass input vector with its corresponding output vector through returning the weight and bias gradients as "delta" 
                    Gradient delta = feedforwardAndBackprop(miniBatchInputs[h], miniBatchOutputs[h]);

                    // Accumulate new bias gradients for each layer and neuron
                    for (int j = 0; j < gradient_b.length; j++) { // loop through layers
                        for (int k = 0; k < gradient_b[j].length; k++) { // loop through neorons in current layer 
                            gradient_b[j][k] += delta.gradient_b[j][k]; // add bias gradient for each neuron in current layer
                        }
                    }
                    // Accumulate new weight gradients for each layer and connection
                    for (int j = 0; j < gradient_w.length; j++) { // loop each layer 
                        for (int k = 0; k < gradient_w[j].length; k++) { // loop current neurons
                            for (int l = 0; l < gradient_w[j][k].length; l++) { // loop previous neurons
                                gradient_w[j][k][l] += delta.gradient_w[j][k][l]; // add weight gradients coonecting current layer to previous layer
                            }
                        }
                    }
                }
                /*
                 * Now that we have the accumulated weights and biases, we update them.
                 * Basically the loops do the same as the previious steps where we aqccumulated the gradients.
                 * But now we apply the proper equation at the innermost level
                */
                for (int h = 0; h < biases.length; h++) {
                    for (int j = 0; j < biases[h].length; j++) {
                        biases[h][j] -= (eta / miniBatchSize) * gradient_b[h][j]; // b(old) - (Learning_rate / batch_size) * b(gradients)
                    }
                }
                for (int h = 0; h < weights.length; h++) {
                    for (int j = 0; j < weights[h].length; j++) {
                        for (int k = 0; k < weights[h][j].length; k++) {
                            weights[h][j][k] -= (eta / miniBatchSize) * gradient_w[h][j][k]; // w(old) - (Learning_rate / batch_size) * w(gradients)
                        }
                    }
                }
            }
            // After each epoch, evaluate and print statistics on the training data
            System.out.println("Epoch " + (epoch + 1) + " complete");
            /* Passes data through the trained network after each epoch. Inside this method 
             * is a feedforward call for evaluation. Since the network is already trained for
             * current epoch, doesnt need to adjust weight/biases. Statemnts revolving around accuracy 
             * will be displayed to the user
             */
            evaluate(inputs, outputs);
        }
    }

    // Computes feedwordward passes, back propgation, and returns the bias and weight gradients. 
    // Training ENTIRE network on each INDIVIDUAL image
    private Gradient feedforwardAndBackprop(double[] input, double[] output) { // input vector for image and one hot vector for output
        // Initialize holders for weights and bias gradients (over single training case)
        double[][] gradient_b = new double[biases.length][];
        double[][][] gradient_w = new double[weights.length][][];
        // Purpose: Allocate appropriate space for each gradient array matching sizes of bias/weight arrays holding random values
        for (int i = 0; i < biases.length; i++) {
            gradient_b[i] = new double[biases[i].length]; // 1st: gradient_b = [[30]] 2nd: gradient_b = [[30], [10]]
            gradient_w[i] = new double[weights[i].length][weights[i][0].length]; // 1st: gradient_w = [[30][784]] 2nd: gradient_w = [[30][784], [10][30]]
        }

        // Initialize storage of activations and z values
        double[] activation = input; // Initial input activation = input vector
        List<double[]> activations = new ArrayList<>(); // store all activations for all layers during feedforward
        // Add all activations
        activations.add(activation); // add initial input first
        List<double[]> zs = new ArrayList<>(); // store all weighted input z vectors for each layer

        // Feedforward to compute the activations for each layer. (Forward passing during training)
        for (int i = 0; i < biases.length; i++) { // loop through layers (hidden, output)
            double[] b = biases[i]; // bias vector for current layer. 1st: [30]  2nd: [10]
            double[][] w = weights[i]; // weight matrix connecting prev layer 1st: [30][784]   2nd: [10][30]
            double[] z = add(dot(w, activation), b); // perform dot product between weight matrix and activations vector, then add bias vector to get our z vector
            zs.add(z); // add new z vector to zs for backprop section during sigmnoid prime calculations
            activation = sigmoid(z); // perfrom sigmoid function  to get new activations for current layer
            activations.add(activation); // store these hidden activations for backprop
        }
        // After all iterations, activations List holds activatiosn for every layer [[784], [30], [10]]
        // zs currently holds z vectors for hidden layer and output layer [[30], [10]]

        /*
        * Backpropagation: Our backwards passes occur here We have all z values for each layer and all activations for each layer stored already.
        * back prop from layer 2 -> layer 1 in the first half. 
        * Then the second half contains the backprop from the intermediate layer to the input layer (1 -> 0)
        */
        // Calculate the error term for the output layer. (Cost derivative: (a - y)) * (sigmoid prime: a * (1 - a))
        // These delta (d) values are then stored in a gradient_b vector because both gradient bias and d are equal to each other
                                // Cost derivative (output activations, output vector) * sigmoid prime (output z vector)
        double[] delta = multiply(costDerivative(activations.get(activations.size() - 1), output), sigmoidPrime(zs.get(zs.size() - 1))); // [10]
        gradient_b[gradient_b.length - 1] = delta; // set delta to last index in gradient_b array

        // Computing the weight gradients connecting output layer and hidden layer
        int output_n = delta.length; // neurons in output layer = 10
        double[] hidden_n = activations.get(activations.size() - 2); // neurons in hidden layer before output layer = [30] in this case
        // Calculate weights 
        for (int i = 0; i < output_n; i++) { // Loop tthrough number of neurons in output layer. Length of 10
            for (int j = 0; j < hidden_n.length; j++) { // Loop tthrough activations in hidden layer. length of 30
                // Set activation * bias_gradient for index of matrix representing weights between output and hidden layer
                gradient_w[gradient_w.length - 1][i][j] = hidden_n[j] * delta[i]; 
            }
        }

        // Loop through hidden layers (SUM(w * b_Gradients) * a * (1 - a))
        for (int l = 2; l <= numLayers - 1; l++) { // in this case only 1 iteration, but can handle multiple
            double[] z = zs.get(zs.size() - l); // grabs all the z values from the layer computed during feedforward [30] = hidden layer z vector
            // (a * (1 - a))
            double[] sp = sigmoidPrime(z); // apply sigmoid prime to these values. Returns [30] but with sp's
            double[][] wTransposed = transpose(weights[weights.length - l + 1]); // transpose the matrix turning [10][30] to [30][10]

            // dot = Compute Sum(w * Gradients) then multiply with sp: 
            delta = multiply(dot(wTransposed, delta), sp); // [30][10] * [10] + [30] = 30x10 * 10x1 + 30x1 = 30x1 = [30]
            gradient_b[gradient_b.length - l] = delta; // gradient_b[0] = [30] because biasgradients = delta

            // Compute the weight gradients based off the delta/biasgradients
            int mHidden = delta.length; // number of neurons in current layer = 30
            int nHidden = activations.get(activations.size() - l - 1).length; // number of neurons in previous layer = 784
            gradient_w[gradient_w.length - l] = new double[mHidden][nHidden]; // gradient_w[0] = [30][784], clear existing gradient weights
             // Calculate weights 
            for (int i = 0; i < mHidden; i++) { // Loop tthrough number of neurons in current layer = 30
                for (int j = 0; j < nHidden; j++) { // Loop tthrough number of neurons in previous layer = 784
                    // On current connection, fill weight gradients j that connect previous layer to i (current layer)
                    gradient_w[gradient_w.length - l][i][j] =  activations.get(activations.size() - l - 1)[j] * delta[i];
                }
            }
        }
        // returns gradient_b = [30, 10] with bias gradients for each neuron and gradient_w = [[30][784], [10][30]] for each weight connecting each neuron for entire minibatch
        return new Gradient(gradient_b, gradient_w);
    }

    // Determine the accuracy of the network based on inputs and labels
    public void evaluate(double[][] inputs, double[][] outputs) {
        int totalInputs = inputs.length; // number of inputs in set (60000)
        int[] correct = new int[10]; // Each index represnts a digit, addes to index IF CORRECT. Ex: [2, 4, ...] means correct 0's is 2 and correct 1's is 4, and so on
        int[] total = new int[10]; // Same representation as above but for all possible 
        int totalCorrect = 0; // counter for how many correct
        // Loop each input and check validity
        for (int i = 0; i < totalInputs; i++) { // loop 60000 (each image) and count if image was correctly predicted
            // Call feedforward function to evaluate on trained network. 
            double[] output = feedforward(inputs[i]); // grab outputs for comparison (real values)
            // Here we utilize our one-hot vector to determine if 
            int predicted = findMax(output); // predicted answer via highest value in vector
            int actual = findMax(outputs[i]); // actual answer for current index in inputs
            // Increment index value with actual
            total[actual]++;
            // compare both indices, if === then Increment index value with actual and update correct count
            if (predicted == actual) {
                correct[actual]++;
                totalCorrect++;
            }
        }
        // iterate through total & correct arrays and display ratio for each label
        for (int i = 0; i < 10; i++) {
            // "label = correct/total"
            System.out.print(i + " = " + correct[i] + "/" + total[i] + " ");
        }
        // calculate the precentage of ALL labels. Ex: "Accuracy = correct / total (60000) = %"
        System.out.println("Accuracy = " + totalCorrect + "/" + totalInputs + " = " + String.format("%.3f", (totalCorrect * 100.0 / totalInputs)) + "%");
    }

    // Load network from filename given by the user
    public void loadNetwork(String filename) {
        try {
            // Open file and wrap for deserialization of Java objetcs 
            ObjectInputStream in = new ObjectInputStream(new FileInputStream(filename));
            // Deserialization: Grab all relevant data from the file, ie, sizes, biases, wights) and set as global variables
            this.sizes = (int[]) in.readObject();
            this.biases = (double[][]) in.readObject();
            this.weights = (double[][][]) in.readObject();
            this.numLayers = sizes.length;
            // close the stream and notify user
            in.close();
            System.out.println("Network loaded successfully.");
        // Basic error handling
        } catch (IOException e) {
            System.out.println("Failed to load the network due to an I/O error.");
            e.printStackTrace();
        } catch (ClassNotFoundException e) {
            System.out.println("Failed to load the network due to a class not being found.");
            e.printStackTrace();
        }
    }

    // Displays images to user based on how they want to filter them (total vs incorrect only). Takes in input and output arrays and passes through feedforward
    public void displayResultImages(double[][] inputs, double[][] outputs, boolean all){
        Scanner scanner = new Scanner(System.in); // gets user input on whether conintue or not
        // Loop through inputs array and basically do same functionality as evaluate
        for (int i = 0; i < inputs.length; i++) {
            double[] image = inputs[i]; // store label pixels for image
            int actual = findMax(outputs[i]); // get max index from one hot encode vectoir for actual answer
            int predicted = findMax(feedforward(image)); // get max index from output vector for network's prediction
            boolean isCorrect = actual == predicted; 
            // If user wants to see all images
            if(all){
                // Images are flattened to 784 pixels, but originally is 28x28. So we want to iterate every 28 pixels (indices) and print based on the value size
                for (int row = 0; row < 784; row += 28) { // increment by 28 for each row
                    for (int col = 0; col < 28; col++) {  // iterate through each column in that current row
                        double pixel = image[row + col]; // grab pixel based on index 
                        System.out.print(pixel > 0.5 ? "#" : " "); // if size is greater that half then print
                    }
                    // Break line to start new one
                    System.out.println();
                }
                // Display Results 
                System.out.println("Correct Classification: " + actual);
                System.out.println("Network Output: " + predicted);
                System.out.println(isCorrect ? "Correct" : "Incorrect");
                
                // Wait for user input (anything but 1 returns user)
                System.out.println("Press 1 to continue or any other key to return to menu.");
                String inputStr = scanner.nextLine();
                if (!inputStr.equals("1")) {
                    break;
                }
            // If user selected to only see misclassified, then skips image if its correct
            } else if (!isCorrect && !all) {
                // Images are flattened to 784 pixels, but originally is 28x28. So we want to iterate every 28 pixels (indices) and print based on the value size
                for (int row = 0; row < 784; row += 28) { // increment by 28 for each row
                    for (int col = 0; col < 28; col++) {  // iterate through each column in that current row
                        double pixel = image[row + col]; // grab pixel based on index 
                        System.out.print(pixel > 0.5 ? "#" : " "); // if size is greater that half then print
                    }
                    // Break line to start new one
                    System.out.println();
                }
                // Display results
                System.out.println("Actual: " + actual);
                System.out.println("Predicted: " + predicted);

                // Wait for user input (anything but 1 returns user)
                System.out.println("Press 1 to continue or any other key to return to menu.");
                String inputStr = scanner.nextLine();
                if (!inputStr.equals("1")) {
                    break;
                }
            }
        }
    }

    // Save network to filename given by the user
    public void saveNetwork(String filename) {
        try {
            // Create filestream and wrap for serialization of Java objetcs 
            ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(filename));
            // Serialization: Write the network structure and parameters to the file
            out.writeObject(this.sizes);
            out.writeObject(this.biases);
            out.writeObject(this.weights);
            
            // Close the output stream and notify the user
            out.close();
            System.out.println("Network saved successfully.");
        // Basic error handling
        } catch (IOException e) {
            System.out.println("Failed to save the network due to an I/O error.");
            e.printStackTrace();
        }
    }
    
    // Helper methods
    // Feedforward function for evaluation phases. Not used during backprop
    public double[] feedforward(double[] input) {
        double[] activation = input;
        // Same sigmoid function as previiously.
        for (int i = 0; i < biases.length; i++) {
            double[] z = add(dot(weights[i], activation), biases[i]);
            activation = sigmoid(z);
        }
        return activation; // vector of values where the max can be taken from
    }

    // Cost derivative (a - y)
    private double[] costDerivative(double[] outputActivations, double[] output) {
        double[] result = new double[outputActivations.length];
        for (int i = 0; i < outputActivations.length; i++) {
            // Derivative of computing the cost is just subtarcting the desired output from the activation
            result[i] = outputActivations[i] - output[i];
        }
        return result;
    }
    // used for determine the largest value of our one hot vector
    private int findMax(double[] array) {
        int maxIndex = 0;
        // Iterate thorugh vector comparing until finding highest value
        for (int i = 1; i < array.length; i++) {
            if (array[i] > array[maxIndex]) {
                maxIndex = i;
            }
        }
        // returns index value for highest
        return maxIndex;
    }

    // Utility functions
    // Dot product between a matrix and a vector
    public static double[] dot(double[][] matrix, double[] vector) {
        // get dimensions
        int rows = matrix.length;
        int cols = matrix[0].length;
        double[] result = new double[rows];
        // Iteration through rows on outer loop
        for (int i = 0; i < rows; i++) {
            // Multiplying the cooresponding column value and adding to the counter to be placed in the result
            double sum = 0.0;
            for (int j = 0; j < cols; j++) {
                sum += matrix[i][j] * vector[j];
            }
            result[i] = sum;
        }
        return result;
    }
    // Add two vectors
    public static double[] add(double[] a, double[] b) {
        double[] result = new double[a.length];
        // Iterate through and simply add each value and place in new array
        for (int i = 0; i < a.length; i++) {
            result[i] = a[i] + b[i];
        }
        return result;
    }
    // Multiply two vectors element-wise
    public static double[] multiply(double[] a, double[] b) {
        double[] result = new double[a.length];
        // 
        for (int i = 0; i < a.length; i++) {
            result[i] = a[i] * b[i];
        }
        return result;
    }
    // Transpose a matrix
    public static double[][] transpose(double[][] a) {
        int m = a.length;
        int n = a[0].length;
        double[][] result = new double[n][m];
        // Basically flip the columns and rows to have reverse dimensions
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                result[j][i] = a[i][j];
            }
        }
        return result;
    }
    // Sigmoid function
    public static double[] sigmoid(double[] z) {
        double[] result = new double[z.length];
        for (int i = 0; i < z.length; i++) {
            result[i] = 1.0 / (1.0 + Math.exp(-z[i])); // 1 / (1 + e^-z)
        }
        return result;
    }
    // Derivative of the sigmoid function (a * (1 - a))
    public static double[] sigmoidPrime(double[] z) {
        double[] result = new double[z.length];
        for (int i = 0; i < z.length; i++) {
            // complete derivative of sigmoid function is calculated below (1/(1+e^-z)) * (1-(1/(1+e^-z)))
            // a = (1 / (1 + e^z))
            result[i] = (1.0 / (1.0 + Math.exp(-z[i]))) * (1.0 - (1.0 / (1.0 + Math.exp(-z[i]))));
        }
        return result;
    }
}

// Structure for managing gradients of biases and weights over time. Serves the role of more type safety.
class Gradient {
    public double[][] gradient_b;
    public double[][][] gradient_w;

    public Gradient(double[][] gradient_b, double[][][] gradient_w) {
        this.gradient_b = gradient_b;
        this.gradient_w = gradient_w;
    }
}

// Structure for managing CSV data of inputs and outputs to be processed by the network.
class CSVData {
    public double[][] inputs;
    public double[][] outputs;

    public CSVData(double[][] inputs, double[][] outputs) {
        this.inputs = inputs;
        this.outputs = outputs;
    }
}