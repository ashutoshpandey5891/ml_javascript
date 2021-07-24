const fs = require('fs');
const _ = require('lodash');
const shuffleSeed = require('shuffle-seed');

function extractColumns(data,colNames){
  const headers = _.first(data);
  const indexes = _.map(colNames,column => headers.indexOf(column));

  const extracted = _.map(data,row => _.pullAt(row,indexes));

  return extracted;
}


function loadCsv(filename,
  {converters = {}
  ,dataColumns = []
  ,labels = []
  ,shuffle = true,
  splitTest = false}){
  // load the file as string
  let data = fs.readFileSync(filename,{encoding : 'utf-8'});
  // split the string into arrays and arrays of arrays
  data = data.split('\n').map(row => row.split(','));
  // drop the unnecessary commas if any
  data = data.map(row => _.dropRightWhile(row,val => val === ''));
  const headers = _.first(data);

  data = data.map((row,index) => {
    if(index === 0){
      return row;
    };

    return row.map((element,index) => {
      if(converters[headers[index]]){
        const converted = converters[headers[index]](element);
        return _.isNaN(converted)? element : converted;
      }

      let result = parseFloat(element);
       return _.isNaN(result) ? element : result;
      });

  });

let labelCols = extractColumns(data,labels);
data = extractColumns(data,dataColumns);
labelCols.shift();
data.shift();

//shuffle the data
data = shuffleSeed.shuffle(data,'shufflePhrase');
labelCols = shuffleSeed.shuffle(labelCols,'shufflePhrase');

//split the dataset if needed
if(splitTest){
  const trainSize = _.isNumber(splitTest) ? splitTest : Math.floor(data.length /2 );
  return {
    features : data.slice(0,trainSize),
    labels : labelCols.slice(0,trainSize),
    testFeatures : data.slice(trainSize),
    testLabels : labelCols.slice(trainSize)
  }
}else {
  return {features : data,labels : labelCols};
}

console.log(data);
console.log(labelCols);
}

const {features,labels,testFeatures,testLabels} = loadCsv('data.csv',{
  splitTest : false,
  shuffle : true,
  dataColumns : ['height','value'],
  labels : ['passed'],
  converters : {
    passed : val => val === 'TRUE'?true:false
}
});

console.log(features);
console.log(labels);
console.log(testFeatures);
console.log(testLabels);
