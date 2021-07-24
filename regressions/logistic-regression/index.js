// predicting if a car will pass emissions test or not using logistic regression
require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('../load-csv');
const LogisticRegression = require('./logistic-regression');
const plot = require('node-remote-plot');
//loading the data
const {
    features,
    labels,
    testFeatures,
    testLabels
} = loadCSV(
    '../data/cars.csv', {
        shuffle: true,
        splitTest: 50,
        dataColumns: ['horsepower', 'displacement', 'weight'],
        labelColumns: ['passedemissions'],
        converters: {
            passedemissions: (value) => {
                return value === 'TRUE' ? 1 : 0;
            }
        }
    });

const classifier = new LogisticRegression(features, labels, {
    learningRate: 0.5,
    iterations: 100,
    batchSize: 50,
    decisionBoundary: 0.5
});

classifier.train();
console.log(classifier.test(testFeatures,testLabels));


plot({
    x : classifier.lossHistory.reverse(),
    xLabel : '#Iterations',
    YLabel : 'Cross entropy loss'
});
