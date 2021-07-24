require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');

const loadCsv = require('./load-csv.js');

function knn(features,labels,predPoint,k){
  //applying standardisation;
  const { mean , variance } = tf.moments(features,0);
  const scaledPredPoint = predPoint.sub(mean).div(variance.pow(0.5));
  return features
    .sub(mean)
    .div(variance.pow(0.5))
    .sub(scaledPredPoint)
    .pow(2)
    .sum(1)
    .pow(0.5)
    .expandDims(1)
    .concat(labels,1)
    .unstack()
    .sort((a,b) => a.get(0) > b.get(0) ? 1:-1)
    .slice(0,k)
    .reduce((acc,pair) => acc + pair.get(1),0)/k;
}

let {features,labels,testFeatures,testLabels} = loadCsv('kc_house_data.csv',{
  dataColumns : ['lat','long','sqft_lot'],
  labelColumns : ['price'],
  splitTest : 10,
  shuffle : true
})

features = tf.tensor(features);
labels = tf.tensor(labels);
//testFeatures = tf.tensor(testFeatures);
//testLabels = tf.tensor(testLabels);

testFeatures.forEach((testPoint,i) => {
  const result = knn(features,labels,tf.tensor(testPoint),10);
  const err = (testLabels[i][0] - result)*100/testLabels[i][0];
  console.log('iter ',i,' Error ',err);
})
