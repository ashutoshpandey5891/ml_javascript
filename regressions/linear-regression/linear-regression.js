const tf = require('@tensorflow/tfjs');
const _ = require('lodash');

class LinearRegression {
  constructor(features,labels,options){
    this.features = this.processFeatures(features);
    this.labels = tf.tensor(labels);

    this.mseHistory = [];
    this.options = Object.assign({learningRate : 0.1,iterations : 1000},
      options);

    this.weights = tf.zeros([this.features.shape[1],1]);
  }


  gradientDescent(features,labels){
    const logits = features.matMul(this.weights);
    const dW = features.transpose()
        .matMul(logits.sub(labels))
        .div(features.shape[0]);

    this.weights = this.weights.sub(dW.mul(this.options.learningRate));

  }

  //apply gradient descent for iterations
  train(){
    const nBatches = Math.floor(
      this.features.shape[0]/this.options.batchSize
    );

    for(let i=0;i<this.options.iterations;i++){
      for(let j=0; j < nBatches ; j++){
        const featureSlice = this.features.slice(
          [j*this.options.batchSize,0],
          [this.options.batchSize,-1]
        );
        const labelSlice = this.labels.slice(
          [j*this.options.batchSize,0],
          [this.options.batchSize,-1]
        );

        this.gradientDescent(featureSlice,labelSlice);
      }
      this.recordMSE();
      this.updateLearningRate();
    }
  }

  //making presictions
  predict(predFeatures){
    predFeatures = this.processFeatures(predFeatures);
    const predictions = predFeatures.matMul(this.weights);
    return predictions;
  }

  //compute R^2 for testing data
  test(testFeatures,testLabels){
    testFeatures = this.processFeatures(testFeatures);
    testLabels = tf.tensor(testLabels);

    const predictions = testFeatures.matMul(this.weights);
    //computing coefficient of determination
    const res = testLabels.sub(predictions)
        .pow(2)
        .sum()
        .get();

    const tot = testLabels.sub(testLabels.mean())
        .pow(2)
        .sum()
        .get();

    return 1 - res/tot ;
  }

  processFeatures(features){
    //convert features to tensor
    features = tf.tensor(features);

    //standardization
    features = this.standardize(features);
    features = tf.ones([features.shape[0],1]).concat(features,1);
    return features;
  }

  standardize(features){
    if(this.mean && this.variance){
      return features.sub(this.mean).div(this.variance.pow(0.5));
    }else{
      const {mean,variance} = tf.moments(features,0);
      this.mean = mean;
      this.variance = variance;

      return features.sub(this.mean).div(this.variance.pow(0.5));
    }
  }

  recordMSE(){
    const mse = this.features.matMul(this.weights)
        .sub(this.labels)
        .pow(2)
        .sum()
        .get();
    //add mse value in front of the array
    this.mseHistory.unshift(mse);
  }
  updateLearningRate(){
    if(this.mseHistory.length < 2){
      return;
    }

    if(this.mseHistory[0] > this.mseHistory[1]){
      this.options.learningRate /= 2;
    }else{
      this.options.learningRate *= 1.05;
    }
  }

}

module.exports = LinearRegression;
