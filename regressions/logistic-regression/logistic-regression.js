// logistic regression module
const tf = require('@tensorflow/tfjs');
const _ = require('lodash');

class LogisticRegression {
    constructor(features, labels, options) {
        this.features = this.processFeatures(features);
        this.labels = tf.tensor(labels);

        //assigning default options
        this.options = Object.assign({
            learningRate: 0.1,
            iterations: 1000,
            decisionBoundary : 0.5
        }, options);

        this.lossHistory = [];
        this.weights = tf.zeros([this.features.shape[1], 1]);
    }

    gradientDescent(features, labels) {
        const logits = features.matMul(this.weights).sigmoid();
        const dW = features.transpose()
            .matMul(logits.sub(labels))
            .div(features.shape[0]);

        this.weights = this.weights.sub(dW.mul(this.options.learningRate));
    }

    train() {
        //training using batch gradient descent
        const nBatches = Math.floor(
            this.features.shape[0] / this.options.batchSize
        );
        for (let i = 0; i < this.options.iterations; i++) {
            for (let j = 0; j < nBatches; j++) {
                const featureSlice = this.features.slice(
                    [j * this.options.batchSize, 0],
                    [this.options.batchSize, -1]
                );
                const labelSlice = this.labels.slice(
                    [j * this.options.batchSize, 0],
                    [this.options.batchSize, -1]
                );

                this.gradientDescent(featureSlice, labelSlice);
            }
            //compute loss and add to the beginning of lossHistory
            const loss = this.lossFunction(this.features,this.labels);
            this.lossHistory.unshift(loss);
        }
    }

    //data preprocessing
    processFeatures(features) {
        features = tf.tensor(features);
        //standardization
        features = this.standardize(features);
        features = tf.ones([features.shape[0], 1])
            .concat(features, 1);
        return features;
    }

    standardize(features) {
        if (this.mean && this.variance) {
            return features.sub(this.mean).div(this.variance.pow(0.5));
        } else {
            const {
                mean,
                variance
            } = tf.moments(features, 0);
            this.mean = mean;
            this.variance = variance;

            return features.sub(this.mean).div(this.variance.pow(0.5));
        }
    }

    //compute loss
    lossFunction(features, labels) {
        const logits = features.matMul(this.weights).sigmoid();
        const l = labels.mul(logits.log()).sum()
            .add(tf.sub(1, labels).mul(tf.sub(1, logits).log()).sum())
            .div(features.shape[0]).mul(-1).get();
        return l;
    }

    //prediction function
    predict(features) {
        features = this.processFeatures(features);
        const preds = features.matMul(this.weights)
                        .sigmoid()
                        .greater(this.options.decisionBoundary)
                        .cast('float32');
        return preds;
    }

    test(testFeatures,testLabels){
        //testFeatures and testlABELS ARE ARRAYS
        const predictions = this.predict(testFeatures);
        testLabels = tf.tensor(testLabels);
        const err = predictions
                        .sub(testLabels)
                        .abs()
                        .sum()
                        .div(testLabels.shape[0])
                        .get();
        return 1.0-err;
    }
}

module.exports = LogisticRegression;
