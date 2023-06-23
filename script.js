let net, webcam;
var loadingModel = true;
var customLabels = [];
const webcamElement = document.getElementById("webcam");

const classifier = knnClassifier.create();


async function app() {
  try {
    net = await mobilenet.load();
    webcam = await tf.data.webcam(webcamElement);
    loadingModel = false;

    while (true) {
      const img = await webcam.capture();
      const result = await net.classify(img);
      const activation = net.infer(img, "conv_preds");
      let knnClassificationResult;

      if (customLabels.length) {
        knnClassificationResult = await classifier.predictClass(activation);
        document.getElementById("cam-description2").innerHTML = knnClassificationResult.label;
      } 
      
      var predictionsTable = document.getElementById('predictions-table');
      for (let i = 0; i < 3; i++) {
        predictionsTable.rows[i+1].cells[0].innerHTML = result[i].className;
        predictionsTable.rows[i+1].cells[0].setAttribute("title", result[i].className);
        predictionsTable.rows[i+1].cells[1].innerHTML = (result[i].probability * 100).toFixed(2);
      }
      img.dispose();
      await tf.nextFrame();
      await sleep(2000);
    }
  } catch (error) {
    console.log(error);
  }
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function addNewLabel() {
  const inputLabel = document.getElementById("input-label").value;

  if (inputLabel != '' && !customLabels.includes(inputLabel)) {
    customLabels.push(inputLabel);
  }
  document.getElementById("list-added-labels").innerHTML = customLabels.join(", ")

  const img = await webcam.capture();
  const activation = net.infer(img, true);
  classifier.addExample(activation, inputLabel);
  img.dispose();
}

app();
