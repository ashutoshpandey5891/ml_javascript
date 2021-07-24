require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('../load-csv');
const LinearRegression = require('./linear-regression');
const plot = require('node-remote-plot');


let {features,labels,testFeatures,testLabels} = loadCSV(
  '../data/cars.csv',{
    shuffle : true,
    splitTest : 50,
    dataColumns : ['horsepower','displacement','weight'],
    labelColumns : ['mpg']
  });

//console.log(features,labels);
const regression = new LinearRegression(features,labels,{
  learningRate : 0.1,
  iterations : 3,
  batchSize : 10
});

regression.train();
// console.log('MSE History : ',regression.mseHistory);
plot({
  x: regression.mseHistory.reverse(),
  xLabel: 'Iteration #',
  yLabel: 'Mean Squared Error'
});
const cod = regression.test(testFeatures, testLabels);
console.log('Coefficient of Determination : ', cod);


regression.predict([
  [120, 380, 2]
]).print();
