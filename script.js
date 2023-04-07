let net, webcam;
var exampleAdded = false;
var loadingModel = true;
var classes = [];
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
      let result2;

      if (exampleAdded) {
        result2 = await classifier.predictClass(activation);
        document.getElementById("cam-description2").innerHTML = result2.label;
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
  const input_label = document.getElementById("input_label").value;

  if (input_label != '' && !classes.includes(input_label)) {
    classes.push(input_label);
  }
  document.getElementById("added_classes_list").innerHTML = classes.join(", ")

  const img = await webcam.capture();
  const activation = net.infer(img, true);
  classifier.addExample(activation, input_label);
  exampleAdded = true;
  img.dispose();
}

app();
